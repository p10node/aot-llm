//! Command-line driver shared by every generated binary.
//!
//! The generated `main.rs` hands this module a [`Runtime`] describing the
//! compiled model (dimensions, the static `forward` function, the embedded
//! tokenizer blob and the weights accessor). Everything else - argument
//! parsing, prompt encoding, the decode loop, sampling and statistics - is
//! model independent.

use super::chat;
use super::constrain::{unescape, Choices, StopMatcher};
use super::matvec;
use super::pool::Pool;
use super::prefix::Prefix;
use super::sampler::{Sampler, SamplerConfig};
use super::state::{Dims, ModelInfo, State, Tensor, Weights};
use super::tokenizer::Tokenizer;
use super::weights;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(target_os = "macos")]
pub use super::metal::Gpu;
/// Placeholder on platforms without a GPU backend (`Runtime::forward_gpu`
/// is always `None` there, so this is never constructed).
#[cfg(not(target_os = "macos"))]
pub struct Gpu {
    pub name: String,
    pub init_ms: f64,
}

/// Static forward pass: `(state, weights, pool, token, position, want_logits)`.
pub type ForwardFn = fn(&mut State, &Weights, &Pool, u32, usize, bool);
/// Batched forward pass over consecutive prompt tokens:
/// `(state, weights, pool, tokens, first_position, want_logits)`.
pub type ForwardBatchFn = fn(&mut State, &Weights, &Pool, &[u32], usize, bool);
/// GPU variants of the two (binaries built with `--gpu`).
pub type ForwardGpuFn = fn(&mut State, &Weights, &mut Gpu, u32, usize, bool);
pub type ForwardBatchGpuFn = fn(&mut State, &Weights, &mut Gpu, &[u32], usize, bool);

/// Default number of prompt tokens per batched step.
pub const DEFAULT_BATCH: usize = 64;

/// Everything the driver needs from the generated model module.
pub struct Runtime {
    pub dims: &'static Dims,
    pub info: &'static ModelInfo,
    pub forward: ForwardFn,
    pub forward_batch: ForwardBatchFn,
    /// GPU forward passes; `None` when the binary was built without `--gpu`.
    pub forward_gpu: Option<ForwardGpuFn>,
    pub forward_batch_gpu: Option<ForwardBatchGpuFn>,
    /// Optional per-dimension RoPE frequency factors (Llama 3.x scaling).
    pub rope_freqs: Option<Tensor>,
    pub tokenizer_blob: &'static [u8],
    /// Compile-time KV prefix cache (empty when none was baked).
    pub prefix_kv: &'static [u8],
    /// Resolves the weights blob (embedded data or a sidecar file).
    pub weights: fn(Option<&Path>) -> std::io::Result<&'static [u8]>,
    /// `true` when weights live in a separate file next to the executable.
    pub sidecar: bool,
}

#[derive(Debug)]
struct Args {
    prompt: Option<String>,
    max_tokens: usize,
    sampler: SamplerConfig,
    ctx: Option<usize>,
    threads: Option<usize>,
    batch: usize,
    chat: bool,
    think: bool,
    system: Option<String>,
    no_bos: bool,
    no_prefetch: bool,
    kernels: Option<String>,
    device: Option<String>,
    weights_path: Option<PathBuf>,
    quiet: bool,
    info: bool,
    show_special: bool,
    tokens_only: bool,
    dump_logits: bool,
    no_prefix_cache: bool,
    bake_prefix_kv: Option<PathBuf>,
    stop: Vec<String>,
    choices: Vec<String>,
    save_kv: Option<PathBuf>,
    load_kv: Option<PathBuf>,
}

const USAGE: &str = "\
Usage: {bin} [OPTIONS] [--prompt TEXT]

Options:
  -p, --prompt TEXT        Prompt text (reads stdin when omitted)
  -n, --max-tokens N       Maximum tokens to generate [default: 128]
      --temp F             Sampling temperature, 0 = greedy [default: 0]
      --top-k N            Top-k filter, 0 = off [default: 40]
      --top-p F            Nucleus sampling threshold [default: 0.95]
      --repeat-penalty F   Repetition penalty, 1 = off [default: 1.0]
      --seed N             RNG seed [default: 42]
  -c, --ctx N              Context length [default: prompt + max-tokens]
  -t, --threads N          Worker threads [default: min(cores, 8)]
  -b, --batch N            Prompt tokens per batched step, 1 = token by token [default: 64]
      --chat               Wrap the prompt in the model's chat template
      --system TEXT        System prompt for --chat
      --think              Let a Qwen3 model think before answering (--chat)
      --stop TEXT          Stop when the output contains TEXT (repeatable; \n escapes allowed)
      --choices A,B,C      Answer with exactly one of the given strings (also --choice TEXT, repeatable)
      --save-kv FILE       After generating, write the tokens and KV cache of the whole run to FILE
      --load-kv FILE       Continue a saved run: the prompt is appended (--chat: as the next user turn)
      --no-bos             Do not prepend the BOS token
      --no-prefetch        Do not page in the weights from background threads at startup
      --kernels NAME       auto|scalar|neon|neon-dotprod|avx2
      --device NAME        auto|cpu|gpu (gpu needs a binary built with --gpu) [default: auto]
      --weights PATH       Sidecar weights file (sidecar builds only)
      --show-special       Print special tokens
      --tokens             Print token ids instead of text
      --dump-logits        Print the logits of the last prompt token and exit
      --no-prefix-cache    Ignore the compile-time KV prefix cache
      --bake-prefix-kv P   (compiler use) evaluate the prompt and write its KV cache to P
  -q, --quiet              Do not print statistics to stderr
      --info               Print model information and exit
  -h, --help               Show this help
";

fn parse_args(rt: &Runtime) -> Result<Args, String> {
    let mut a = Args {
        prompt: None,
        max_tokens: 128,
        sampler: SamplerConfig::default(),
        ctx: None,
        threads: None,
        batch: DEFAULT_BATCH,
        chat: false,
        think: false,
        system: None,
        no_bos: false,
        no_prefetch: false,
        kernels: None,
        device: None,
        weights_path: None,
        quiet: false,
        info: false,
        show_special: false,
        tokens_only: false,
        dump_logits: false,
        no_prefix_cache: false,
        bake_prefix_kv: None,
        stop: Vec::new(),
        choices: Vec::new(),
        save_kv: None,
        load_kv: None,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let next = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
    };
    while i < argv.len() {
        let f = argv[i].as_str();
        match f {
            "-p" | "--prompt" => a.prompt = Some(next(&mut i, f)?),
            "-n" | "--max-tokens" => a.max_tokens = next(&mut i, f)?.parse().map_err(|_| "bad --max-tokens")?,
            "--temp" | "--temperature" => a.sampler.temperature = next(&mut i, f)?.parse().map_err(|_| "bad --temp")?,
            "--top-k" => a.sampler.top_k = next(&mut i, f)?.parse().map_err(|_| "bad --top-k")?,
            "--top-p" => a.sampler.top_p = next(&mut i, f)?.parse().map_err(|_| "bad --top-p")?,
            "--repeat-penalty" => a.sampler.repeat_penalty = next(&mut i, f)?.parse().map_err(|_| "bad --repeat-penalty")?,
            "--seed" => a.sampler.seed = next(&mut i, f)?.parse().map_err(|_| "bad --seed")?,
            "-c" | "--ctx" => a.ctx = Some(next(&mut i, f)?.parse().map_err(|_| "bad --ctx")?),
            "-t" | "--threads" => a.threads = Some(next(&mut i, f)?.parse().map_err(|_| "bad --threads")?),
            "-b" | "--batch" => a.batch = next(&mut i, f)?.parse::<usize>().map_err(|_| "bad --batch")?.max(1),
            "--chat" => a.chat = true,
            "--think" => a.think = true,
            "--system" => a.system = Some(next(&mut i, f)?),
            "--no-bos" => a.no_bos = true,
            "--no-prefetch" => a.no_prefetch = true,
            "--prefetch" => {}
            "--kernels" => a.kernels = Some(next(&mut i, f)?),
            "--device" => a.device = Some(next(&mut i, f)?),
            "--weights" => a.weights_path = Some(PathBuf::from(next(&mut i, f)?)),
            "--show-special" => a.show_special = true,
            "--tokens" => a.tokens_only = true,
            "--dump-logits" => a.dump_logits = true,
            "--no-prefix-cache" => a.no_prefix_cache = true,
            "--bake-prefix-kv" => a.bake_prefix_kv = Some(PathBuf::from(next(&mut i, f)?)),
            "--stop" => a.stop.push(unescape(&next(&mut i, f)?)),
            "--choice" => a.choices.push(next(&mut i, f)?),
            "--choices" => a.choices.extend(next(&mut i, f)?.split(',').filter(|c| !c.is_empty()).map(str::to_string)),
            "--save-kv" => a.save_kv = Some(PathBuf::from(next(&mut i, f)?)),
            "--load-kv" => a.load_kv = Some(PathBuf::from(next(&mut i, f)?)),
            "-q" | "--quiet" => a.quiet = true,
            "--info" => a.info = true,
            "-h" | "--help" => {
                print!("{}", USAGE.replace("{bin}", &bin_name()));
                println!("\nModel: {} ({}, {})", rt.info.name, rt.info.arch, rt.info.file_type);
                std::process::exit(0);
            }
            _ if a.prompt.is_none() && !f.starts_with('-') => a.prompt = Some(f.to_string()),
            _ => return Err(format!("unknown option {f} (try --help)")),
        }
        i += 1;
    }
    Ok(a)
}

fn bin_name() -> String {
    std::env::args().next().and_then(|p| Path::new(&p).file_name().map(|s| s.to_string_lossy().into_owned())).unwrap_or_else(|| "model".into())
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8)
}

fn mb(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn print_info(rt: &Runtime) {
    // Ignore write errors (e.g. a closed pipe) instead of panicking.
    let mut o = std::io::stdout().lock();
    let d = rt.dims;
    let tok = Tokenizer::new(rt.tokenizer_blob).ok();
    let _ = writeln!(o, "model:        {}", rt.info.name);
    let _ = writeln!(o, "source:       {}", rt.info.source);
    let _ = writeln!(o, "architecture: {}", rt.info.arch);
    let _ = writeln!(o, "file type:    {}", rt.info.file_type);
    let _ = writeln!(o, "parameters:   {:.3} B", rt.info.n_params as f64 / 1e9);
    let _ = writeln!(o, "tensors:      {} ({})", rt.info.n_tensors, rt.info.quant_mix);
    let _ = writeln!(o, "weights:      {:.1} MB ({})", mb(rt.info.weights_len as u64), if rt.sidecar { "sidecar file" } else { "embedded in executable" });
    let _ = writeln!(o, "dim={} hidden={} layers={} heads={} kv_heads={} head_dim={} vocab={} ctx_train={}", d.dim, d.hidden, d.n_layer, d.n_head, d.n_kv_head, d.head_dim, d.vocab, d.n_ctx_train);
    let _ = writeln!(o, "rope: base={} rot_dim={} scale={} freq_factors={}", d.rope_base, d.rot_dim, d.rope_scale, rt.rope_freqs.is_some());
    if let Some(t) = tok {
        let _ =
            writeln!(o, "tokenizer:    {} ({} tokens, {} merges, bos={:?} eos={:?} eot={:?}, chat template: {})", if t.is_spm() { "spm" } else { "bpe" }, t.n_tokens(), t.n_merges(), t.bos(), t.eos(), t.eot(), chat::name(t.chat_format()));
    }
    let _ = writeln!(o, "kernels:      {} ({} cores)", matvec::detect_kernels().name, default_threads());
    let _ = writeln!(o, "gpu:          {}", if rt.forward_gpu.is_some() { "metal (--device auto|cpu|gpu)" } else { "none (compile with --gpu metal)" });
    let _ = writeln!(o, "compiler:     aot-llm {}", rt.info.compiler_version);
}

/// Write complete UTF-8 characters from `buf` to `out`, keeping any trailing
/// partial sequence for the next token.
fn flush_utf8(buf: &mut Vec<u8>, out: &mut impl Write) {
    let valid = match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid > 0 {
        let _ = out.write_all(&buf[..valid]);
        let _ = out.flush();
        buf.drain(..valid);
    }
    // Drop hopeless garbage (a partial sequence longer than 4 bytes).
    if buf.len() > 4 {
        buf.clear();
    }
}

#[cfg(target_os = "macos")]
fn gpu_new(weights: &'static [u8], state: &State, d: &Dims) -> Result<Gpu, String> {
    Gpu::new(weights, state, d)
}

#[cfg(not(target_os = "macos"))]
fn gpu_new(_weights: &'static [u8], _state: &State, _d: &Dims) -> Result<Gpu, String> {
    Err("no GPU backend on this platform".into())
}

/// One forward step on whichever device is active.
#[allow(clippy::too_many_arguments)]
fn step(rt: &Runtime, state: &mut State, weights: &Weights, pool: &Pool, gpu: &mut Option<Gpu>, token: u32, pos: usize, want_logits: bool) {
    match gpu {
        Some(g) => (rt.forward_gpu.expect("gpu forward"))(state, weights, g, token, pos, want_logits),
        None => (rt.forward)(state, weights, pool, token, pos, want_logits),
    }
}

/// One batched prompt step on whichever device is active.
#[allow(clippy::too_many_arguments)]
fn step_batch(rt: &Runtime, state: &mut State, weights: &Weights, pool: &Pool, gpu: &mut Option<Gpu>, tokens: &[u32], pos0: usize, want_logits: bool) {
    match gpu {
        Some(g) => (rt.forward_batch_gpu.expect("gpu batched forward"))(state, weights, g, tokens, pos0, want_logits),
        None => (rt.forward_batch)(state, weights, pool, tokens, pos0, want_logits),
    }
}

/// Entry point for generated binaries. Returns the process exit code.
pub fn main(rt: Runtime) -> i32 {
    let t0 = Instant::now();
    let args = match parse_args(&rt) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    if args.info {
        print_info(&rt);
        return 0;
    }
    let tok = match Tokenizer::new(rt.tokenizer_blob) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let wbytes = match (rt.weights)(args.weights_path.as_deref()) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: cannot open weights: {e}");
            return 1;
        }
    };
    if wbytes.len() < rt.info.weights_len {
        eprintln!("error: weights blob is {} bytes, expected {}", wbytes.len(), rt.info.weights_len);
        return 1;
    }
    let weights = Weights::new(wbytes);
    let threads = args.threads.unwrap_or_else(default_threads).max(1);
    if !args.no_prefetch && !args.info {
        // Overlaps page-in with tokenization and pool start-up; see
        // `weights::prefault_background`.
        weights::prefault_background(&wbytes[..rt.info.weights_len], threads);
    }
    if let Some(name) = &args.kernels {
        match matvec::kernels_by_name(name) {
            Some(k) => {
                matvec::set_kernels(k);
            }
            None => {
                eprintln!("error: unknown kernel set {name}");
                return 2;
            }
        }
    }
    let device = args.device.as_deref().unwrap_or("auto");
    let use_gpu = match device {
        "auto" => rt.forward_gpu.is_some(),
        "gpu" | "metal" => {
            if rt.forward_gpu.is_none() {
                eprintln!("error: this binary was built without GPU support (aot-llm compile --gpu metal)");
                return 2;
            }
            true
        }
        "cpu" => false,
        other => {
            eprintln!("error: unknown device {other} (auto | cpu | gpu)");
            return 2;
        }
    };
    let t_pool0 = Instant::now();
    let pool = Pool::new(threads);
    let t_pool = t_pool0.elapsed();

    // Prompt.
    let raw = match &args.prompt {
        Some(p) => p.clone(),
        None if args.bake_prefix_kv.is_some() && args.chat => String::new(),
        None => {
            let mut s = String::new();
            if std::io::stdin().read_to_string(&mut s).is_err() || s.trim().is_empty() {
                eprintln!("error: no prompt given (use --prompt or pipe text on stdin)");
                return 2;
            }
            s.trim_end_matches('\n').to_string()
        }
    };
    let text = if args.chat {
        let fmt = if args.think && tok.chat_format() == super::tokenizer::chat_format::QWEN3 { super::tokenizer::chat_format::CHATML } else { tok.chat_format() };
        let built = if args.bake_prefix_kv.is_some() {
            // Baking: only the part of the template that precedes the user turn.
            chat::system_prefix(fmt, args.system.as_deref()).filter(|s| !s.is_empty())
        } else if args.load_kv.is_some() {
            // Continuing a saved conversation: close the last reply, open the next turn.
            chat::continuation(fmt, &raw)
        } else {
            chat::format(fmt, args.system.as_deref(), &raw)
        };
        match built {
            Some(t) => t,
            None => {
                eprintln!("error: this model has no known chat template (or nothing to bake); use a raw prompt");
                return 2;
            }
        }
    } else {
        raw
    };
    let d = rt.dims;
    // A saved run (--load-kv): its tokens come first, the prompt continues them.
    let saved: Option<Prefix> = match &args.load_kv {
        None => None,
        Some(path) => match weights::mmap_file(path) {
            Ok(blob) => match Prefix::parse(blob, d) {
                Some(p) => Some(p),
                None => {
                    eprintln!("error: {} is not a KV cache of this model", path.display());
                    return 2;
                }
            },
            Err(e) => {
                eprintln!("error: cannot open {}: {e}", path.display());
                return 2;
            }
        },
    };
    let add_bos = !args.no_bos && tok.add_bos() && saved.is_none();
    let t_enc0 = Instant::now();
    let mut prompt = tok.encode(&text, add_bos, true);
    if let Some(pre) = &saved {
        let mut all = pre.tokens.clone();
        all.append(&mut prompt);
        prompt = all;
    }
    let t_encode = t_enc0.elapsed();
    if prompt.is_empty() {
        eprintln!("error: empty prompt");
        return 2;
    }
    let ctx = args.ctx.unwrap_or_else(|| (prompt.len() + args.max_tokens).min(d.n_ctx_train.max(prompt.len() + 1)));
    if ctx > d.n_ctx_train && !args.quiet {
        eprintln!("warning: context {ctx} exceeds the training context {}", d.n_ctx_train);
    }
    if prompt.len() > ctx {
        eprintln!("error: prompt has {} tokens but the context is {ctx}; raise --ctx", prompt.len());
        return 2;
    }
    let rope_ff = rt.rope_freqs.as_ref().map(|t| weights.f32s(t));
    let batch = args.batch.min(prompt.len()).max(1);
    let mut state = State::new(d, ctx, rope_ff, threads, batch);
    // GPU: maps the weights and the state's buffers (no copies), compiles or
    // loads the kernels. Falls back to the CPU unless --device gpu was given.
    let t_gpu0 = Instant::now();
    let mut gpu: Option<Gpu> = None;
    if use_gpu {
        match gpu_new(wbytes, &state, d) {
            Ok(g) => gpu = Some(g),
            Err(e) if device == "auto" => {
                if !args.quiet {
                    eprintln!("warning: GPU unavailable ({e}); using the CPU");
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                return 1;
            }
        }
    }
    let t_gpu = t_gpu0.elapsed();
    let t_ready = t0.elapsed();

    // KV rows that need no evaluation: a saved run (--load-kv) or the
    // compile-time prefix embedded in the binary.
    let mut cached = 0usize;
    if let Some(pre) = &saved {
        pre.load_into(&mut state);
        cached = pre.cached();
    } else if !args.no_prefix_cache && args.bake_prefix_kv.is_none() {
        if let Some(pre) = Prefix::parse(rt.prefix_kv, d) {
            if prompt.len() > pre.len() && prompt.starts_with(&pre.tokens) {
                pre.load_into(&mut state);
                cached = pre.cached();
            }
        }
    }

    // Prompt evaluation in batches (logits are only needed for the last token).
    let t1 = Instant::now();
    let mut done = cached;
    while done < prompt.len() {
        let end = (done + batch).min(prompt.len());
        let last = end == prompt.len();
        if end - done == 1 {
            step(&rt, &mut state, &weights, &pool, &mut gpu, prompt[done], done, last);
        } else {
            step_batch(&rt, &mut state, &weights, &pool, &mut gpu, &prompt[done..end], done, last);
        }
        done = end;
    }
    let t_prompt = t1.elapsed();

    if let Some(path) = &args.bake_prefix_kv {
        let blob = super::prefix::serialize(&state, d, &prompt, prompt.len());
        if let Err(e) = std::fs::write(path, &blob) {
            eprintln!("error: cannot write {}: {e}", path.display());
            return 1;
        }
        eprintln!("[aot-llm] baked KV cache for {} prompt tokens ({:.1} MB) into {}", prompt.len(), mb(blob.len() as u64), path.display());
        return 0;
    }

    if std::env::var_os("AOT_DEBUG_COMPARE").is_some() && batch > 1 && gpu.is_none() {
        // Debug aid: re-run the prompt token by token in a fresh state and
        // report the first (layer, position) where the KV caches diverge.
        let mut s2 = State::new(d, ctx, rope_ff, threads, 1);
        for (i, &t) in prompt.iter().enumerate() {
            (rt.forward)(&mut s2, &weights, &pool, t, i, i + 1 == prompt.len());
        }
        let kv = d.kv_dim();
        let mut first: Option<(usize, usize, f32, &str)> = None;
        for layer in 0..d.n_layer {
            for pos in 0..prompt.len() {
                let at = (layer * ctx + pos) * kv;
                let dk = state.k_cache[at..at + kv].iter().zip(&s2.k_cache[at..at + kv]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let dv = state.v_cache[at..at + kv].iter().zip(&s2.v_cache[at..at + kv]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if (dk > 0.0 || dv > 0.0) && first.is_none() {
                    let nk = state.k_cache[at..at + kv].iter().zip(&s2.k_cache[at..at + kv]).filter(|(a, b)| a != b).count();
                    eprintln!("[debug] {nk} of {kv} k values differ at layer {layer} pos {pos}");
                    first = Some((layer, pos, dk.max(dv), if dk > 0.0 { "k" } else { "v" }));
                }
            }
        }
        let dl = state.logits.iter().zip(&s2.logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        match first {
            Some((l, p, dd, which)) => eprintln!("[debug] first KV divergence at layer {l} pos {p} ({which}, max diff {dd}); logits max diff {dl}"),
            None => eprintln!("[debug] KV caches identical; logits max diff {dl}"),
        }
    }

    if args.dump_logits {
        let mut line = String::with_capacity(state.logits.len() * 10);
        for v in &state.logits {
            line.push_str(&format!("{v:.6} "));
        }
        println!("{}", line.trim_end());
        return 0;
    }

    // Generation.
    let mut sampler = Sampler::new(args.sampler.clone());
    let mut stopper = StopMatcher::new(&args.stop);
    let mut choices = if args.choices.is_empty() { None } else { Some(Choices::new(&tok, &args.choices)) };
    let stop_ids: Vec<u32> = (0..d.vocab as u32).filter(|&t| tok.is_stop(t)).collect();
    let mut all = prompt.clone();
    let mut generated = 0usize;
    let mut out_buf = Vec::new();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut pos = prompt.len();
    let t2 = Instant::now();
    let mut t_first = None;
    let mut why = "max tokens";
    // With --choices the pieces are collected and the chosen string is
    // printed verbatim at the end (the tokenizer's spelling of a choice can
    // carry a leading space).
    let mut choice_buf: Vec<u8> = Vec::new();
    while generated < args.max_tokens && pos < ctx {
        if let Some(ch) = &mut choices {
            ch.mask(&mut state.logits, &stop_ids);
        }
        let next = sampler.sample(&mut state.logits, &all);
        if t_first.is_none() {
            t_first = Some(t2.elapsed());
        }
        let stop = tok.is_stop(next);
        if stop && !args.show_special {
            why = "stop token";
            break;
        }
        // Constrained answers: follow the trie; a stop token at a terminal
        // node (allowed by the mask) ends the shorter choice.
        let choice_done = match &mut choices {
            Some(ch) => {
                if !ch.advance(next) {
                    why = "choice";
                    break;
                }
                ch.done()
            }
            None => false,
        };
        all.push(next);
        generated += 1;
        if args.tokens_only {
            let _ = write!(out, "{next} ");
            let _ = out.flush();
        } else if choices.is_some() {
            tok.decode_piece(next, &mut choice_buf, args.show_special);
        } else {
            let mut piece = Vec::new();
            tok.decode_piece(next, &mut piece, args.show_special);
            let emit = stopper.feed(&piece);
            out_buf.extend_from_slice(&emit);
            flush_utf8(&mut out_buf, &mut out);
        }
        if stopper.matched.is_some() {
            why = "stop string";
            break;
        }
        if choice_done {
            why = "choice";
            break;
        }
        if stop || generated >= args.max_tokens || pos >= ctx {
            if stop {
                why = "stop token";
            } else if pos >= ctx {
                why = "context full";
            }
            break;
        }
        step(&rt, &mut state, &weights, &pool, &mut gpu, next, pos, true);
        pos += 1;
    }
    let t_gen = t2.elapsed();
    if stopper.matched.is_none() {
        out_buf.extend(stopper.finish());
    }
    if let Some(ch) = &choices {
        if !args.tokens_only {
            match ch.selected() {
                Some(t) => out_buf.extend_from_slice(t.as_bytes()),
                None => out_buf.extend_from_slice(&choice_buf),
            }
        }
    }
    if !out_buf.is_empty() {
        let _ = out.write_all(String::from_utf8_lossy(&out_buf).as_bytes());
    }
    let _ = out.write_all(b"\n");
    let _ = out.flush();

    // --save-kv: every token of the run, with cache rows for the evaluated
    // ones (the last generated token was never fed back).
    if let Some(path) = &args.save_kv {
        let blob = super::prefix::serialize(&state, d, &all, pos);
        if let Err(e) = std::fs::write(path, &blob) {
            eprintln!("error: cannot write {}: {e}", path.display());
            return 1;
        }
        if !args.quiet {
            eprintln!("[aot-llm] saved {} tokens ({} with KV rows, {:.1} MB) to {}", all.len(), pos, mb(blob.len() as u64), path.display());
        }
    }

    if !args.quiet {
        let ps = t_prompt.as_secs_f64();
        let gs = t_gen.as_secs_f64();
        let gen_forwards = generated.saturating_sub(1).max(if generated > 0 { 1 } else { 0 });
        let evaluated = prompt.len() - cached;
        let mut line = format!(
            "[aot-llm] startup {:.3} ms (threads {:.3}, tokenize {:.3}{}) | prompt {} tok{} / {:.3} s ({:.1} tok/s) | gen {} tok / {:.3} s ({:.1} tok/s)",
            t_ready.as_secs_f64() * 1e3,
            t_pool.as_secs_f64() * 1e3,
            t_encode.as_secs_f64() * 1e3,
            if gpu.is_some() { format!(", gpu {:.3}", t_gpu.as_secs_f64() * 1e3) } else { String::new() },
            prompt.len(),
            if cached > 0 { format!(" ({cached} cached)") } else { String::new() },
            ps,
            evaluated as f64 / ps.max(1e-9),
            generated,
            gs,
            gen_forwards as f64 / gs.max(1e-9),
        );
        if let Some(tf) = t_first {
            line.push_str(&format!(" | first token {:.1} ms", (t_prompt + tf).as_secs_f64() * 1e3));
        }
        if let Some(rss) = weights::peak_rss_bytes() {
            line.push_str(&format!(" | peak RSS {:.0} MB", mb(rss)));
        }
        let backend = match &gpu {
            Some(g) => format!("metal ({})", g.name),
            None => matvec::active_kernels().name.to_string(),
        };
        line.push_str(&format!(" | {} threads | {} | ctx {} | end: {why}", threads, backend, ctx));
        eprintln!("{line}");
    }
    0
}
