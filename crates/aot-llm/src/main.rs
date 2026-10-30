//! `aot-llm` command line: compile a GGUF model into a standalone binary.

use anyhow::{bail, Context, Result};
use aot_codegen::build::{cargo_build, BuildOptions};
use aot_codegen::emit::{crate_name_from, write_project, EmitOptions, GpuBackend, LibKind};
use aot_codegen::model::ModelSpec;
use aot_codegen::tokenizer::{build_blob, chat_format_name};
use aot_gguf::{Gguf, MetaValue};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "aot-llm", version, about = "Ahead-of-time compiler for GGUF LLMs: one model in, one native executable out")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Compile a GGUF model into a standalone executable.
    Compile {
        /// Path to the quantized GGUF model.
        #[arg(short, long)]
        model: PathBuf,
        /// Output executable path.
        #[arg(short, long)]
        output: PathBuf,
        /// Directory for the generated Rust project [default: <output>.build].
        #[arg(long)]
        build_dir: Option<PathBuf>,
        /// Rust target triple, e.g. x86_64-unknown-linux-musl.
        #[arg(long)]
        target: Option<String>,
        /// Optimise for the build machine's CPU (-C target-cpu=native). The
        /// binary then only runs on CPUs with the same features.
        #[arg(long)]
        native: bool,
        /// Keep weights in a sidecar file (<output>.weights) mapped at run
        /// time instead of embedding them into the executable.
        #[arg(long)]
        sidecar: bool,
        /// Only generate the project; do not run cargo.
        #[arg(long)]
        emit_only: bool,
        /// Parallel cargo jobs.
        #[arg(short, long)]
        jobs: Option<usize>,
        /// Hide cargo output unless the build fails.
        #[arg(short, long)]
        quiet: bool,
        /// Crate / binary name inside the generated project [default: from output name].
        #[arg(long)]
        name: Option<String>,
        /// Bake the KV cache of this system prompt (chat template) into the
        /// binary; `--chat --system <same text>` then skips re-evaluating it.
        #[arg(long)]
        system: Option<String>,
        /// Bake the KV cache of this raw text prefix into the binary; prompts
        /// that start with it (same tokenization) skip re-evaluating it.
        #[arg(long, conflicts_with = "system")]
        prefix: Option<String>,
        /// Also compile a GPU backend into the binary: `metal` (macOS,
        /// Apple Silicon). The binary picks the GPU by default and keeps the
        /// CPU path (`--device cpu`).
        #[arg(long, value_name = "BACKEND")]
        gpu: Option<String>,
        /// Build a model library with a C API (`aot_llm.h`) instead of an
        /// executable: `static` (lib<name>.a, default) or `shared`
        /// (lib<name>.dylib / .so). <output> is then a directory.
        #[arg(long, value_name = "KIND", num_args = 0..=1, default_missing_value = "static")]
        lib: Option<String>,
        /// With --lib: prefix for the exported C symbols and header names
        /// (e.g. `qwen3_` gives `qwen3_aot_open`), so that several model
        /// libraries can be linked into one program.
        #[arg(long, value_name = "PREFIX", requires = "lib")]
        symbol_prefix: Option<String>,
    },
    /// Print metadata and tensors of a GGUF file.
    Inspect {
        model: PathBuf,
        /// List every tensor.
        #[arg(long)]
        tensors: bool,
        /// Print all metadata values (arrays are summarised).
        #[arg(long)]
        metadata: bool,
    },
    /// Tokenize text with a model's vocabulary (debugging aid).
    Tokenize {
        model: PathBuf,
        #[arg(short, long)]
        text: String,
        /// Do not prepend BOS.
        #[arg(long)]
        no_bos: bool,
        /// Wrap in the model's chat template.
        #[arg(long)]
        chat: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compile { model, output, build_dir, target, native, sidecar, emit_only, jobs, quiet, name, system, prefix, gpu, lib, symbol_prefix } => {
            compile(&model, &output, build_dir, target, native, sidecar, emit_only, jobs, quiet, name, system, prefix, gpu, lib, symbol_prefix)
        }
        Cmd::Inspect { model, tensors, metadata } => inspect(&model, tensors, metadata),
        Cmd::Tokenize { model, text, no_bos, chat } => tokenize(&model, &text, no_bos, chat),
    }
}

/// Binary units (MiB-based), matching the statistics printed by generated binaries.
fn human(bytes: u64) -> String {
    let b = bytes as f64;
    const K: f64 = 1024.0;
    if b >= K * K * K {
        format!("{:.2} GB", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.1} MB", b / (K * K))
    } else if b >= K {
        format!("{:.1} KB", b / K)
    } else {
        format!("{bytes} B")
    }
}

#[allow(clippy::too_many_arguments)]
fn compile(
    model: &Path,
    output: &Path,
    build_dir: Option<PathBuf>,
    target: Option<String>,
    native: bool,
    sidecar: bool,
    emit_only: bool,
    jobs: Option<usize>,
    quiet: bool,
    name: Option<String>,
    system: Option<String>,
    prefix: Option<String>,
    gpu: Option<String>,
    lib: Option<String>,
    symbol_prefix: Option<String>,
) -> Result<()> {
    let t0 = Instant::now();
    let symbol_prefix = symbol_prefix.unwrap_or_default();
    if !symbol_prefix.is_empty() && !(symbol_prefix.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !symbol_prefix.starts_with(|c: char| c.is_ascii_digit())) {
        bail!("--symbol-prefix must be a C identifier prefix (letters, digits, underscores)");
    }
    let lib = match lib.as_deref() {
        None => None,
        Some(k) => Some(LibKind::parse(k).ok_or_else(|| anyhow::anyhow!("unknown library kind {k:?} (static | shared)"))?),
    };
    let gpu = match gpu.as_deref() {
        None => None,
        Some("metal") => {
            let for_macos = match &target {
                Some(t) => t.contains("apple-darwin"),
                None => cfg!(target_os = "macos"),
            };
            if !for_macos {
                bail!("--gpu metal needs a macOS target (host or --target aarch64-apple-darwin)");
            }
            Some(GpuBackend::Metal)
        }
        Some(other) => bail!("unknown GPU backend {other:?} (supported: metal)"),
    };
    let model = model.canonicalize().with_context(|| format!("model {} not found", model.display()))?;
    let g = Gguf::open(&model).with_context(|| format!("parsing {}", model.display()))?;
    eprintln!("[1/4] parsed {} (GGUF v{}, {} tensors, {} metadata keys)", model.display(), g.version, g.tensors.len(), g.metadata.len());
    let spec = ModelSpec::from_gguf(&g, &model).context("analysing model architecture")?;
    let d = &spec.dims;
    eprintln!(
        "      {} | {} | {:.3}B params | {} | dim {} hidden {} layers {} heads {}/{} vocab {} ctx {}",
        spec.name,
        spec.arch,
        spec.n_params as f64 / 1e9,
        spec.file_type,
        d.dim,
        d.hidden,
        d.n_layer,
        d.n_head,
        d.n_kv_head,
        d.vocab,
        d.n_ctx_train
    );
    eprintln!("      quantization: {} | weights {}", spec.quant_mix, human(spec.data_len));
    for w in &spec.warnings {
        eprintln!("      warning: {w}");
    }

    let (blob, tok) = build_blob(&g).context("building tokenizer")?;
    eprintln!(
        "[2/4] tokenizer: {} ({}) {} tokens, {} merges, {} special, bos={:?} eos={:?} eot={:?}, chat template: {}, blob {}",
        tok.kind,
        tok.pre,
        tok.n_tokens,
        tok.n_merges,
        tok.n_special,
        tok.bos,
        tok.eos,
        tok.eot,
        chat_format_name(tok.chat_format),
        human(blob.len() as u64)
    );
    for w in &tok.warnings {
        eprintln!("      warning: {w}");
    }

    let dir = build_dir.unwrap_or_else(|| {
        let mut s = output.as_os_str().to_owned();
        s.push(".build");
        PathBuf::from(s)
    });
    let crate_name = name.unwrap_or_else(|| crate_name_from(&output.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()));
    let bake = system.is_some() || prefix.is_some();
    // With --lib and a baked prefix, pass 1 is a plain executable (it has to
    // run to evaluate the prefix); pass 2 emits the library.
    let pass1_lib = if bake { None } else { lib };
    let mut opts = EmitOptions {
        crate_name: crate_name.clone(),
        sidecar,
        native,
        gguf_path: model.clone(),
        compiler_version: aot_codegen::VERSION.to_string(),
        target: target.clone(),
        prefix_kv_path: None,
        gpu,
        lib: pass1_lib,
        symbol_prefix: symbol_prefix.clone(),
    };
    write_project(&dir, &spec, &blob, &opts).with_context(|| format!("writing project to {}", dir.display()))?;
    eprintln!(
        "[3/4] generated project in {} (crate `{crate_name}`, {} layers unrolled, weights {}{}{})",
        dir.display(),
        d.n_layer,
        if sidecar { "sidecar" } else { "embedded" },
        gpu.map(|g| format!(", gpu: {}", g.name())).unwrap_or_default(),
        lib.map(|k| format!(", {} library + C API", if k == LibKind::Static { "static" } else { "shared" })).unwrap_or_default()
    );
    if emit_only {
        if bake && lib.is_some() {
            opts.lib = lib;
            write_project(&dir, &spec, &blob, &opts)?;
        }
        eprintln!("      --emit-only: skipping cargo build");
        return Ok(());
    }

    eprintln!("[4/4] cargo build --release{}{}{} ...", target.as_ref().map(|t| format!(" --target {t}")).unwrap_or_default(), if native { " (target-cpu=native)" } else { "" }, if bake { " (pass 1 of 2)" } else { "" });
    let mut built = cargo_build(&dir, &crate_name, &BuildOptions { target: target.clone(), jobs, quiet, cargo: None, lib: pass1_lib })?;

    if bake {
        // Pass 1 produced a binary without a prefix cache; use it to evaluate
        // the prefix with the exact kernels of the final binary, then rebuild
        // with the resulting KV blob embedded.
        if target.is_some() {
            bail!("--system/--prefix need to run the compiled binary, which is not possible with --target");
        }
        let kv_path = dir.canonicalize()?.join("prefix_kv.bin");
        let mut cmd = std::process::Command::new(&built.binary);
        cmd.arg("--bake-prefix-kv").arg(&kv_path);
        match (&system, &prefix) {
            (Some(s), _) => {
                cmd.arg("--chat").arg("--system").arg(s);
            }
            (_, Some(p)) => {
                cmd.arg("--prompt").arg(p);
            }
            _ => unreachable!(),
        }
        if sidecar {
            let wp = g.data_section();
            let tmp = dir.canonicalize()?.join("weights.tmp");
            std::fs::write(&tmp, wp)?;
            cmd.arg("--weights").arg(&tmp);
        }
        let status = cmd.status().context("running the pass-1 binary to bake the prefix")?;
        if !status.success() {
            bail!("baking the KV prefix failed ({status})");
        }
        let _ = std::fs::remove_file(dir.join("weights.tmp"));
        opts.prefix_kv_path = Some(kv_path.clone());
        opts.lib = lib;
        write_project(&dir, &spec, &blob, &opts)?;
        eprintln!("      cargo build --release (pass 2 of 2, embedding {}{}) ...", human(std::fs::metadata(&kv_path)?.len()), if lib.is_some() { ", library" } else { "" });
        built = cargo_build(&dir, &crate_name, &BuildOptions { target: target.clone(), jobs, quiet, cargo: None, lib })?;
    }

    if let Some(kind) = lib {
        // Library: <output>/ holds the library, the header and (sidecar) the weights.
        std::fs::create_dir_all(output).with_context(|| format!("creating {}", output.display()))?;
        let lib_name = kind.file_name(&crate_name, target.as_deref());
        let lib_path = output.join(&lib_name);
        std::fs::copy(&built.binary, &lib_path).with_context(|| format!("copying library to {}", lib_path.display()))?;
        std::fs::write(output.join("aot_llm.h"), aot_codegen::emit::c_header(&symbol_prefix))?;
        if sidecar {
            let wp = output.join(format!("{crate_name}.weights"));
            std::fs::write(&wp, g.data_section()).with_context(|| format!("writing {}", wp.display()))?;
            eprintln!("      wrote sidecar weights {} ({}); pass its path in aot_model_params.weights_path", wp.display(), human(spec.data_len));
        }
        let size = std::fs::metadata(&lib_path)?.len();
        let linux = target.as_deref().map(|t| t.contains("linux")).unwrap_or(cfg!(target_os = "linux"));
        let extra = if linux { " -lpthread -ldl -lm" } else { "" };
        let link = match kind {
            LibKind::Static => format!("cc app.c -I{out} {lib}{extra}", out = output.display(), lib = lib_path.display()),
            LibKind::Shared => format!("cc app.c -I{out} -L{out} -l{crate_name}{extra}", out = output.display()),
        };
        eprintln!("done: {} ({}) + {}/aot_llm.h in {:.1}s (cargo {:.1}s)\nlink: {link}", lib_path.display(), human(size), output.display(), t0.elapsed().as_secs_f64(), built.elapsed.as_secs_f64());
        return Ok(());
    }

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::copy(&built.binary, output).with_context(|| format!("copying binary to {}", output.display()))?;
    if sidecar {
        let mut wp = output.as_os_str().to_owned();
        wp.push(".weights");
        let wp = PathBuf::from(wp);
        std::fs::write(&wp, g.data_section()).with_context(|| format!("writing {}", wp.display()))?;
        eprintln!("      wrote sidecar weights {} ({})", wp.display(), human(spec.data_len));
    }
    let size = std::fs::metadata(output)?.len();
    eprintln!(
        "done: {} ({}) in {:.1}s (cargo {:.1}s)\nrun:  {} --prompt \"Hello world\"",
        output.display(),
        human(size),
        t0.elapsed().as_secs_f64(),
        built.elapsed.as_secs_f64(),
        if output.is_absolute() || output.starts_with(".") { output.display().to_string() } else { format!("./{}", output.display()) }
    );
    Ok(())
}

fn summarise(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) => {
            let s: String = s.chars().take(80).collect();
            format!("{s:?}")
        }
        MetaValue::Array(a) => {
            let head: Vec<String> = a.iter().take(4).map(summarise).collect();
            format!("[{}{}] ({} items)", head.join(", "), if a.len() > 4 { ", ..." } else { "" }, a.len())
        }
        MetaValue::F32(f) => format!("{f}"),
        MetaValue::F64(f) => format!("{f}"),
        MetaValue::Bool(b) => format!("{b}"),
        other => other.as_u64().map(|v| v.to_string()).unwrap_or_else(|| format!("{other:?}")).to_string(),
    }
}

fn inspect(model: &Path, tensors: bool, metadata: bool) -> Result<()> {
    let g = Gguf::open(model).with_context(|| format!("parsing {}", model.display()))?;
    println!("file:        {}", model.display());
    println!("gguf:        v{} alignment {} data offset {} data size {}", g.version, g.alignment, g.data_offset, human(g.data_len()));
    println!("tensors:     {} ({:.3}B params)", g.tensors.len(), g.n_params() as f64 / 1e9);
    let mut mix = std::collections::BTreeMap::new();
    for t in &g.tensors {
        *mix.entry(t.ty.name()).or_insert(0usize) += 1;
    }
    println!("types:       {}", mix.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", "));
    match ModelSpec::from_gguf(&g, model) {
        Ok(spec) => {
            let d = &spec.dims;
            println!("model:       {} ({}, {})", spec.name, spec.arch, spec.file_type);
            println!(
                "dims:        dim {} hidden {} layers {} heads {} kv {} head_dim {} rot {} vocab {} ctx {} eps {} rope {}{}",
                d.dim,
                d.hidden,
                d.n_layer,
                d.n_head,
                d.n_kv_head,
                d.head_dim,
                d.rot_dim,
                d.vocab,
                d.n_ctx_train,
                d.eps,
                d.rope_base,
                if spec.rope_freqs.is_some() { " (+freq factors)" } else { "" }
            );
            println!("output head: {}", if spec.output.is_some() { "separate" } else { "tied to token_embd" });
            for w in &spec.warnings {
                println!("warning:     {w}");
            }
        }
        Err(e) => println!("model:       not compilable: {e:#}"),
    }
    match build_blob(&g) {
        Ok((blob, t)) => println!(
            "tokenizer:   {} ({}) {} tokens, {} merges, {} special, bos={:?} eos={:?} eot={:?}, stop={:?}, chat={}, blob {}",
            t.kind,
            t.pre,
            t.n_tokens,
            t.n_merges,
            t.n_special,
            t.bos,
            t.eos,
            t.eot,
            t.stop_tokens,
            chat_format_name(t.chat_format),
            human(blob.len() as u64)
        ),
        Err(e) => println!("tokenizer:   unsupported: {e:#}"),
    }
    if metadata {
        println!("\nmetadata:");
        for (k, v) in &g.metadata {
            println!("  {k} = {}", summarise(v));
        }
    } else {
        println!("\nmetadata (use --metadata for all):");
        for (k, v) in &g.metadata {
            if !matches!(v, MetaValue::Array(_)) {
                println!("  {k} = {}", summarise(v));
            }
        }
    }
    if tensors {
        println!("\ntensors:");
        for t in &g.tensors {
            println!("  {:<32} {:>7} {:?} offset {} ({})", t.name, t.ty.name(), t.dims, t.offset, human(t.n_bytes()));
        }
    }
    Ok(())
}

fn tokenize(model: &Path, text: &str, no_bos: bool, chat: bool) -> Result<()> {
    let g = Gguf::open(model)?;
    let (blob, _) = build_blob(&g)?;
    let blob: &'static [u8] = Box::leak(blob.into_boxed_slice());
    let tok = aot_kernels::tokenizer::Tokenizer::new(blob).map_err(|e| anyhow::anyhow!(e))?;
    let text = if chat {
        match aot_kernels::chat::format(tok.chat_format(), None, text) {
            Some(t) => t,
            None => bail!("model has no known chat template"),
        }
    } else {
        text.to_string()
    };
    let ids = tok.encode(&text, !no_bos && tok.add_bos(), true);
    println!("{} tokens: {:?}", ids.len(), ids);
    for &id in &ids {
        let mut piece = Vec::new();
        tok.decode_piece(id, &mut piece, true);
        println!("  {id:>7}  {:?}  (type {})", String::from_utf8_lossy(&piece), tok.token_type(id));
    }
    println!("decoded: {:?}", tok.decode(&ids, false));
    Ok(())
}
