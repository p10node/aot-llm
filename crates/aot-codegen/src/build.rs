//! Drives `cargo build --release` on a generated project.

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    /// Rust target triple (e.g. `x86_64-unknown-linux-musl`).
    pub target: Option<String>,
    /// Pass `-j N` to cargo.
    pub jobs: Option<usize>,
    /// Hide cargo output unless the build fails.
    pub quiet: bool,
    /// Cargo executable.
    pub cargo: Option<PathBuf>,
}

/// Result of a successful build.
#[derive(Debug)]
pub struct BuildOutput {
    pub binary: PathBuf,
    pub elapsed: std::time::Duration,
}

/// Build the project in `dir` and return the path of the release binary.
pub fn cargo_build(dir: &Path, crate_name: &str, o: &BuildOptions) -> Result<BuildOutput> {
    let cargo = o.cargo.clone().unwrap_or_else(|| PathBuf::from(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())));
    // Cargo resolves a relative CARGO_TARGET_DIR against its own cwd.
    let dir = dir.canonicalize().with_context(|| format!("project directory {} not found", dir.display()))?;
    let target_dir = dir.join("target");
    let mut cmd = Command::new(&cargo);
    cmd.arg("build").arg("--release").current_dir(&dir).env("CARGO_TARGET_DIR", &target_dir);
    if let Some(t) = &o.target {
        cmd.arg("--target").arg(t);
    }
    if let Some(j) = o.jobs {
        cmd.arg("-j").arg(j.to_string());
    }
    if o.quiet {
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    } else {
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    let start = Instant::now();
    let out = cmd.output().with_context(|| format!("running {} (is Rust installed? https://rustup.rs)", cargo.display()))?;
    if !out.status.success() {
        if o.quiet {
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        }
        bail!("cargo build failed with {}", out.status);
    }
    let mut bin = target_dir.clone();
    if let Some(t) = &o.target {
        bin.push(t);
    }
    bin.push("release");
    bin.push(crate_name);
    if o.target.as_deref().is_some_and(|t| t.contains("windows")) {
        bin.set_extension("exe");
    }
    if !bin.exists() {
        return Err(anyhow!("build succeeded but {} was not produced", bin.display()));
    }
    Ok(BuildOutput { binary: bin, elapsed: start.elapsed() })
}
