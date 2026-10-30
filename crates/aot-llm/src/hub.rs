//! `hf://` model references: download a GGUF from the Hugging Face Hub into
//! a local cache. Uses the system `curl` so the compiler stays dependency
//! free; `HF_TOKEN` is forwarded for gated repositories, `HF_ENDPOINT`
//! overrides the host, `AOT_LLM_CACHE` the cache directory.
//!
//! Forms:
//! * `hf://org/repo:Q4_K_M`   - the GGUF of the repo whose name contains the quant tag
//! * `hf://org/repo`          - same with `Q4_K_M`
//! * `hf://org/repo/file.gguf` - an explicit file

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_QUANT: &str = "Q4_K_M";

/// Turn a model argument into a local path, downloading `hf://` references.
pub fn resolve_model(spec: &str, quiet: bool) -> Result<PathBuf> {
    let Some(rest) = spec.strip_prefix("hf://") else {
        return Ok(PathBuf::from(spec));
    };
    let rest = rest.trim_end_matches('/');
    let (repo, file) = if rest.ends_with(".gguf") {
        let (repo, file) = rest.rsplit_once('/').context("hf:// reference needs org/repo/file.gguf")?;
        (repo.to_string(), file.to_string())
    } else {
        let (repo, tag) = match rest.split_once(':') {
            Some((r, t)) => (r.to_string(), t.to_string()),
            None => (rest.to_string(), DEFAULT_QUANT.to_string()),
        };
        let file = pick_file(&repo, &tag)?;
        (repo, file)
    };
    if repo.matches('/').count() != 1 {
        bail!("hf:// reference needs the form hf://org/repo[:QUANT|/file.gguf], got {spec:?}");
    }
    let dir = cache_dir()?.join(&repo);
    let path = dir.join(&file);
    if path.exists() {
        if !quiet {
            eprintln!("      using cached {}", path.display());
        }
        return Ok(path);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let url = format!("{}/{repo}/resolve/main/{file}", endpoint());
    if !quiet {
        eprintln!("      downloading {url}");
    }
    let tmp = dir.join(format!("{file}.part"));
    let mut cmd = Command::new("curl");
    cmd.args(["-L", "--fail", "--retry", "3", "-C", "-", "-o"]).arg(&tmp).arg(&url);
    if quiet {
        cmd.arg("-s");
    } else {
        cmd.arg("-#");
    }
    for a in auth_args() {
        cmd.arg(a);
    }
    let status = cmd.status().context("running curl (is it installed?)")?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!("download of {url} failed ({status}); for gated repositories set HF_TOKEN");
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Choose the GGUF of `repo` whose file name contains `tag` (case-insensitive).
fn pick_file(repo: &str, tag: &str) -> Result<String> {
    let url = format!("{}/api/models/{repo}", endpoint());
    let mut cmd = Command::new("curl");
    cmd.args(["-sL", "--fail"]).arg(&url);
    for a in auth_args() {
        cmd.arg(a);
    }
    let out = cmd.output().context("running curl (is it installed?)")?;
    if !out.status.success() {
        bail!("cannot list {repo} on the Hub ({}); check the repository name, or set HF_TOKEN for gated models", out.status);
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let files = rfilenames(&body);
    let ggufs: Vec<&str> = files.iter().map(String::as_str).filter(|f| f.to_ascii_lowercase().ends_with(".gguf")).filter(|f| !f.to_ascii_lowercase().contains("mmproj")).collect();
    if ggufs.is_empty() {
        bail!("{repo} has no GGUF files");
    }
    let tag_l = tag.to_ascii_lowercase();
    let mut matches: Vec<&str> = ggufs.iter().copied().filter(|f| f.to_ascii_lowercase().contains(&tag_l)).collect();
    // Prefer the plain file over sharded parts and over longer variants.
    matches.sort_by_key(|f| (f.contains("-of-"), f.len()));
    match matches.first() {
        Some(f) if !f.contains("-of-") => Ok((*f).to_string()),
        Some(f) => bail!("{f} is a multi-part GGUF; sharded files are not supported, pick another quantization"),
        None => bail!("no GGUF in {repo} matches {tag:?}; available: {}", ggufs.join(", ")),
    }
}

/// Values of every `"rfilename": "..."` in the Hub's JSON (no JSON library needed).
fn rfilenames(json: &str) -> Vec<String> {
    let mut out = Vec::new();
    let key = "\"rfilename\"";
    let mut rest = json;
    while let Some(i) = rest.find(key) {
        rest = &rest[i + key.len()..];
        let Some(q) = rest.find('"') else { break };
        let after = &rest[q + 1..];
        let Some(end) = after.find('"') else { break };
        out.push(after[..end].replace("\\/", "/"));
        rest = &after[end + 1..];
    }
    out
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

fn auth_args() -> Vec<String> {
    match std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        Ok(t) if !t.is_empty() => vec!["-H".into(), format!("Authorization: Bearer {t}")],
        _ => Vec::new(),
    }
}

fn cache_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var("AOT_LLM_CACHE") {
        return Ok(PathBuf::from(d));
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).context("no HOME directory for the model cache (set AOT_LLM_CACHE)")?;
    Ok(Path::new(&home).join(".cache").join("aot-llm").join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rfilenames() {
        let json = r#"{"siblings":[{"rfilename":"README.md"},{"rfilename":"Qwen3-0.6B-Q4_K_M.gguf"},{"rfilename":"a\/b.gguf"}]}"#;
        assert_eq!(rfilenames(json), vec!["README.md", "Qwen3-0.6B-Q4_K_M.gguf", "a/b.gguf"]);
    }

    #[test]
    fn plain_paths_pass_through() {
        assert_eq!(resolve_model("./m.gguf", true).unwrap(), PathBuf::from("./m.gguf"));
    }
}
