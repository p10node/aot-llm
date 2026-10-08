#!/usr/bin/env python3
"""Reproducible benchmark harness for binaries produced by `aot-llm compile`.

For every binary it measures:
  * cold start   - the binary is copied with F_NOCACHE (macOS) so its pages are
                   not in the page cache, then run once: time to first token
                   includes paging the weights in from disk;
  * warm runs    - N runs, medians of startup, prompt eval, decode, first
                   token, peak RSS;
  * chat prompt  - N warm runs with --chat and a ~60-token system prompt, the
                   realistic "time to first token" for an assistant.

Usage:
  scripts/bench.py run --label baseline --out docs/benchmarks \
      --bin tinyllama_q4km=../out/tinyllama_bin --bin llama32_1b=../out/llama32_1b
  scripts/bench.py summary --dir docs/benchmarks     # regenerate README.md table

Only the Python standard library is used.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

PROMPT = "The quick brown fox jumps over the lazy dog because"
# Shared with `aot-llm compile --system "$(cat scripts/chat_system.txt)"` so the
# baked KV prefix matches the benchmark's chat prompt exactly.
CHAT_SYSTEM = (pathlib.Path(__file__).with_name("chat_system.txt").read_text().strip())
CHAT_PROMPT = "What does a compiler do?"

STATS = {
    "startup_ms": re.compile(r"startup ([\d.]+) ms"),
    "prompt_tokens": re.compile(r"prompt (\d+) tok"),
    "prompt_cached": re.compile(r"prompt \d+ tok \((\d+) cached\)"),
    "prompt_s": re.compile(r"prompt \d+ tok(?: \(\d+ cached\))? / ([\d.]+) s"),
    "prompt_tps": re.compile(r"prompt \d+ tok(?: \(\d+ cached\))? / [\d.]+ s \(([\d.]+) tok/s\)"),
    "gen_tokens": re.compile(r"gen (\d+) tok"),
    "gen_tps": re.compile(r"gen \d+ tok / [\d.]+ s \(([\d.]+) tok/s\)"),
    "first_token_ms": re.compile(r"first token ([\d.]+) ms"),
    "peak_rss_mib": re.compile(r"peak RSS (\d+) MB"),
    "threads": re.compile(r"\| (\d+) threads"),
}


def parse_stats(stderr: str) -> dict:
    line = next((l for l in stderr.splitlines()[::-1] if l.startswith("[aot-llm]")), "")
    out = {}
    for k, rx in STATS.items():
        m = rx.search(line)
        if m:
            v = m.group(1)
            out[k] = float(v) if "." in v else int(v)
    out["kernels"] = line.split("|")[-2].strip() if line.count("|") >= 2 else ""
    return out


def run_binary(path: str, args: list[str]) -> dict:
    p = subprocess.run([path, *args], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{path} failed: {p.stderr}")
    s = parse_stats(p.stderr)
    s["text"] = p.stdout.strip()[:120]
    return s


def nocache_copy(src: str, dst: str) -> None:
    """Copy `src` to `dst` bypassing the page cache so the copy starts cold."""
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(fi.fileno(), fcntl.F_NOCACHE, 1)
            fcntl.fcntl(fo.fileno(), fcntl.F_NOCACHE, 1)
        shutil.copyfileobj(fi, fo, 8 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.chmod(dst, 0o755)
    if sys.platform == "linux":
        # Best effort: drop the copy's pages (needs root for the global cache).
        try:
            with open(dst, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass


def median(xs):
    return statistics.median(xs) if xs else None


def bench_binary(name: str, path: str, runs: int, threads: int, ctx_extra: list[str]) -> dict:
    path = os.path.abspath(path)
    size = os.path.getsize(path)
    common = ["-t", str(threads), *ctx_extra]
    # Cold start.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path)) as td:
        cold_bin = os.path.join(td, "cold_" + os.path.basename(path))
        nocache_copy(path, cold_bin)
        cold = run_binary(cold_bin, ["-p", PROMPT, "-n", "64", *common])
    # Warm runs.
    run_binary(path, ["-p", PROMPT, "-n", "8", "-q", *common])
    warm = [run_binary(path, ["-p", PROMPT, "-n", "64", *common]) for _ in range(runs)]
    chat = [run_binary(path, ["--chat", "--system", CHAT_SYSTEM, "-p", CHAT_PROMPT, "-n", "16", *common]) for _ in range(runs)]
    med = lambda rs, k: median([r[k] for r in rs if k in r])
    return {
        "name": name,
        "path": path,
        "binary_mib": round(size / 2**20, 1),
        "threads": threads,
        "kernels": warm[0].get("kernels", ""),
        "cold": {k: cold.get(k) for k in ("startup_ms", "first_token_ms", "prompt_tokens", "prompt_tps", "gen_tps", "peak_rss_mib")},
        "warm": {
            "startup_ms": med(warm, "startup_ms"),
            "first_token_ms": med(warm, "first_token_ms"),
            "prompt_tokens": warm[0].get("prompt_tokens"),
            "prompt_tps": med(warm, "prompt_tps"),
            "gen_tps": med(warm, "gen_tps"),
            "gen_tps_min": min(r["gen_tps"] for r in warm),
            "gen_tps_max": max(r["gen_tps"] for r in warm),
            "peak_rss_mib": med(warm, "peak_rss_mib"),
        },
        "chat": {
            "prompt_tokens": chat[0].get("prompt_tokens"),
            "prompt_cached": chat[0].get("prompt_cached", 0),
            "first_token_ms": med(chat, "first_token_ms"),
            "first_token_ms_min": min(r["first_token_ms"] for r in chat),
            "prompt_tps": med(chat, "prompt_tps"),
            "prompt_tps_max": max(r["prompt_tps"] for r in chat),
        },
        "sample_output": warm[0]["text"],
    }


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def machine_info() -> dict:
    info = {"platform": sys.platform, "rustc": sh(["rustc", "--version"])}
    if sys.platform == "darwin":
        info["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["cores"] = sh(["sysctl", "-n", "hw.ncpu"])
        info["ram_gib"] = round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30)
        info["os"] = "macOS " + sh(["sw_vers", "-productVersion"])
    else:
        info["cpu"] = sh(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip()
        info["cores"] = str(os.cpu_count())
        info["os"] = sh(["uname", "-sr"])
    try:
        info["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    return info


def git_info() -> dict:
    return {
        "commit": sh(["git", "rev-parse", "--short", "HEAD"]),
        "subject": sh(["git", "log", "-1", "--format=%s"]),
        "dirty": bool(sh(["git", "status", "--porcelain"])),
    }


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown(res: dict, path: pathlib.Path) -> None:
    m, g = res["machine"], res["git"]
    L = []
    L.append(f"# Benchmark: {res['label']}\n")
    L.append(f"* Date: {res['date']}")
    L.append(f"* Commit: `{g['commit']}` {g['subject']}{' (dirty tree)' if g['dirty'] else ''}")
    L.append(f"* Machine: {m.get('cpu','?')}, {m.get('cores','?')} cores, {m.get('ram_gib','?')} GiB, {m.get('os','?')}, {m.get('rustc','?')}")
    L.append(f"* Load average at start: {m.get('load_avg','?')}")
    L.append(f"* Method: `scripts/bench.py` — prompt `{PROMPT!r}`, 64 generated tokens, greedy, {res['runs']} warm runs (medians), "
             f"cold = F_NOCACHE copy of the binary run once. Chat = `--chat --system <~60 tokens>`, 16 tokens.\n")
    L.append("## Startup and time to first token\n")
    L.append("| binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tokens (cached) |")
    L.append("|---|---|---|---|---|---|")
    for b in res["binaries"]:
        L.append(f"| {b['name']} | {fmt(b['warm']['startup_ms'], 3)} | {fmt(b['cold']['first_token_ms'], 0)} | {fmt(b['warm']['first_token_ms'], 0)} | {fmt(b['chat']['first_token_ms'], 0)} | {fmt(b['chat']['prompt_tokens'])} ({fmt(b['chat'].get('prompt_cached', 0))}) |")
    L.append("\n## Throughput and memory\n")
    L.append("| binary | prompt tok/s (warm) | chat prompt tok/s | decode tok/s (median) | decode min–max | peak RSS (MiB) | binary (MiB) | kernels / threads |")
    L.append("|---|---|---|---|---|---|---|---|")
    for b in res["binaries"]:
        w = b["warm"]
        L.append(f"| {b['name']} | {fmt(w['prompt_tps'])} | {fmt(b['chat']['prompt_tps'])} | {fmt(w['gen_tps'])} | {fmt(w['gen_tps_min'])}–{fmt(w['gen_tps_max'])} | {fmt(w['peak_rss_mib'], 0)} | {b['binary_mib']} | {b['kernels']} / {b['threads']} |")
    L.append("\n## Cold run detail\n")
    L.append("| binary | cold startup (ms) | cold prompt tok/s | cold decode tok/s |")
    L.append("|---|---|---|---|")
    for b in res["binaries"]:
        c = b["cold"]
        L.append(f"| {b['name']} | {fmt(c['startup_ms'], 3)} | {fmt(c['prompt_tps'])} | {fmt(c['gen_tps'])} |")
    if res.get("notes"):
        L.append("\n## Notes\n")
        L.append(res["notes"])
    L.append("\nSample output (first warm run): " + " / ".join(f"`{b['sample_output'][:60]}`" for b in res["binaries"]))
    L.append(f"\nRaw data: `{path.with_suffix('.json').name}`. Reference numbers for Ollama/llama.cpp: `ollama-reference.md`.\n")
    path.write_text("\n".join(L))


def cmd_run(a) -> None:
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in out.glob("[0-9][0-9]-*.json"))
    n = len(existing) + 1
    stem = f"{n:02d}-{a.label}"
    res = {
        "label": a.label,
        "date": dt.date.today().isoformat(),
        "git": git_info(),
        "machine": machine_info(),
        "runs": a.runs,
        "notes": a.notes or "",
        "binaries": [],
    }
    extra = []
    if a.extra:
        extra = a.extra.split()
    for spec in a.bin:
        name, path = spec.split("=", 1)
        print(f"== {name}: {path}", file=sys.stderr)
        res["binaries"].append(bench_binary(name, path, a.runs, a.threads, extra))
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, out / f"{stem}.md")
    print(f"wrote {out / stem}.md", file=sys.stderr)
    cmd_summary(argparse.Namespace(dir=a.out))


def cmd_summary(a) -> None:
    d = pathlib.Path(a.dir)
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("[0-9][0-9]-*.json"))]
    L = ["# Benchmark history\n",
         "One file per measurement, newest last. Every entry is produced by `scripts/bench.py run` on the same machine "
         "with the same prompts; see each file for the exact method. `ollama-reference.md` holds the llama.cpp/Ollama "
         "numbers the baseline was compared against.\n",
         "Throughput columns show the median of the warm runs and, in parentheses, the best run; the best run is the "
         "better estimate when other processes were competing for the CPU (see the load average in each file).\n",
         "| # | label | commit | binary | startup (ms) | cold first token (ms) | warm first token (ms) | chat first token (ms) | chat prompt tok/s | decode tok/s | peak RSS (MiB) | load |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs, 1):
        load = r["machine"].get("load_avg", ["?"])[0]
        for b in r["binaries"]:
            L.append(f"| {i} | [{r['label']}]({i:02d}-{r['label']}.md) | `{r['git']['commit']}` | {b['name']} | {fmt(b['warm']['startup_ms'], 3)} | "
                     f"{fmt(b['cold']['first_token_ms'], 0)} | {fmt(b['warm']['first_token_ms'], 0)} | {fmt(b['chat']['first_token_ms'], 0)} ({fmt(b['chat'].get('first_token_ms_min'), 0)}) | "
                     f"{fmt(b['chat']['prompt_tps'])} ({fmt(b['chat'].get('prompt_tps_max'))}) | {fmt(b['warm']['gen_tps'])} ({fmt(b['warm']['gen_tps_max'])}) | {fmt(b['warm']['peak_rss_mib'], 0)} | {load} |")
    (d / "README.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="benchmark binaries and write docs")
    r.add_argument("--label", required=True, help="short slug, e.g. baseline, batched-prompt")
    r.add_argument("--out", default="docs/benchmarks")
    r.add_argument("--bin", action="append", required=True, help="name=path (repeatable)")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--extra", default="", help="extra flags passed to every run")
    r.add_argument("--notes", default="")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summary", help="regenerate the history table")
    s.add_argument("--dir", default="docs/benchmarks")
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
