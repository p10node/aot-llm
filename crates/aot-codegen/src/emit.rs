//! Emits the standalone Rust project for a model: `Cargo.toml`, `main.rs`,
//! the generated `model.rs` (static forward pass), `payload.rs` (how the
//! weights and tokenizer are embedded) and the runtime sources.

use crate::model::{Activation, Kind, LayerSpec, ModelSpec, TensorSpec};
use crate::rt_sources::RT_SOURCES;
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Options controlling the generated project.
#[derive(Debug, Clone)]
pub struct EmitOptions {
    /// Cargo package / binary name.
    pub crate_name: String,
    /// Keep weights in a sidecar file instead of embedding them.
    pub sidecar: bool,
    /// Add `-C target-cpu=native` to the project's cargo config.
    pub native: bool,
    /// Absolute path of the source GGUF (embedded with `.incbin`).
    pub gguf_path: PathBuf,
    pub compiler_version: String,
}

/// Turn an arbitrary string into a valid crate name.
pub fn crate_name_from(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "model".into()
    } else if out.chars().next().unwrap().is_ascii_digit() {
        format!("m_{out}")
    } else {
        out
    }
}

fn tensor_lit(t: &TensorSpec) -> String {
    format!(
        "Tensor {{ name: {:?}, kind: Kind::{}, off: {}, len: {}, rows: {}, cols: {} }}",
        t.name,
        t.kind.rust_name(),
        t.off,
        t.len,
        t.rows,
        t.cols
    )
}

fn act_expr(a: Activation, src: &str) -> String {
    match a {
        Activation::Q8_0 => "&s.aq0".into(),
        Activation::Q8_K => "&s.aqk".into(),
        Activation::F32 => format!("&s.{src}"),
    }
}

/// Emit the activation quantization needed by `kinds` for input `src`.
fn emit_quantize(out: &mut String, src: &str, kinds: impl Iterator<Item = Kind>) {
    let acts: BTreeSet<Activation> = kinds.map(Kind::activation).collect();
    for a in acts {
        match a {
            Activation::Q8_0 => writeln!(out, "    quantize_q8_0(&s.{src}, &mut s.aq0);").unwrap(),
            Activation::Q8_K => writeln!(out, "    quantize_q8_k(&s.{src}, &mut s.aqk);").unwrap(),
            Activation::F32 => {}
        }
    }
}

fn emit_matvec(out: &mut String, t: &TensorSpec, texpr: &str, dst: &str, src: &str) {
    writeln!(
        out,
        "    matvec_{}(pool, &mut s.{dst}, w.bytes(&{texpr}), {}, {}, {});",
        t.kind.fn_suffix(),
        act_expr(t.kind.activation(), src),
        t.rows,
        t.cols
    )
    .unwrap();
}

fn emit_layer(out: &mut String, i: usize, l: &LayerSpec, spec: &ModelSpec) {
    let d = &spec.dims;
    writeln!(out, "/// Transformer block {i}.").unwrap();
    writeln!(out, "#[inline(never)]").unwrap();
    writeln!(out, "fn layer_{i}(s: &mut State, w: &Weights, pool: &Pool, pos: usize) {{").unwrap();
    writeln!(out, "    let l = &LAYERS[{i}];").unwrap();
    // Attention.
    writeln!(out, "    rmsnorm(&mut s.xb, &s.x, w.f32s(&l.attn_norm), DIMS.eps);").unwrap();
    emit_quantize(out, "xb", [l.wq.kind, l.wk.kind, l.wv.kind].into_iter());
    emit_matvec(out, &l.wq, "l.wq", "q", "xb");
    emit_matvec(out, &l.wk, "l.wk", "k", "xb");
    emit_matvec(out, &l.wv, "l.wv", "v", "xb");
    writeln!(out, "    rope(&mut s.q, {}, {}, {}, &s.rope_cs);", d.n_head, d.head_dim, d.rot_dim).unwrap();
    writeln!(out, "    rope(&mut s.k, {}, {}, {}, &s.rope_cs);", d.n_kv_head, d.head_dim, d.rot_dim).unwrap();
    writeln!(out, "    store_kv(&mut s.k_cache, &mut s.v_cache, &s.k, &s.v, &DIMS, s.ctx, {i}, pos);").unwrap();
    writeln!(out, "    attention(pool, &mut s.xb, &s.q, &s.k_cache, &s.v_cache, &mut s.scores, &DIMS, s.ctx, {i}, pos);").unwrap();
    emit_quantize(out, "xb", [l.wo.kind].into_iter());
    emit_matvec(out, &l.wo, "l.wo", "xb2", "xb");
    writeln!(out, "    add_inplace(&mut s.x, &s.xb2);").unwrap();
    // Feed-forward.
    writeln!(out, "    rmsnorm(&mut s.xb, &s.x, w.f32s(&l.ffn_norm), DIMS.eps);").unwrap();
    emit_quantize(out, "xb", [l.gate.kind, l.up.kind].into_iter());
    let fused = l.gate.kind == l.up.kind && l.gate.kind.activation() != Activation::F32;
    if fused {
        let (fn_name, dot_fn) = match l.gate.kind.activation() {
            Activation::Q8_K => ("matvec2_q8_k_input", "dot_fn_q8_k"),
            _ => ("matvec2_q8_0_input", "dot_fn_q8_0"),
        };
        writeln!(
            out,
            "    {fn_name}(pool, &mut s.hb, &mut s.hb2, w.bytes(&l.gate), w.bytes(&l.up), {}, {}, Kind::{}.row_bytes({}), {dot_fn}(Kind::{}));",
            act_expr(l.gate.kind.activation(), "xb"),
            l.gate.rows,
            l.gate.kind.rust_name(),
            l.gate.cols,
            l.gate.kind.rust_name()
        )
        .unwrap();
    } else {
        emit_matvec(out, &l.gate, "l.gate", "hb", "xb");
        emit_matvec(out, &l.up, "l.up", "hb2", "xb");
    }
    writeln!(out, "    silu_mul(&mut s.hb, &s.hb2);").unwrap();
    emit_quantize(out, "hb", [l.down.kind].into_iter());
    emit_matvec(out, &l.down, "l.down", "xb2", "hb");
    writeln!(out, "    add_inplace(&mut s.x, &s.xb2);").unwrap();
    writeln!(out, "}}\n").unwrap();
}

/// Generate `model.rs`: constants, the tensor table and the static forward pass.
pub fn emit_model_rs(spec: &ModelSpec, opts: &EmitOptions) -> String {
    let d = &spec.dims;
    let mut o = String::new();
    writeln!(o, "//! Generated by aot-llm {} from `{}`. Do not edit.", opts.compiler_version, spec.source).unwrap();
    writeln!(o, "//!").unwrap();
    writeln!(o, "//! Model: {} ({}, {})", spec.name, spec.arch, spec.file_type).unwrap();
    writeln!(o, "//! Static pipeline: embedding -> [rmsnorm -> attention -> residual -> rmsnorm -> SwiGLU FFN -> residual] x {} -> rmsnorm -> logits", d.n_layer).unwrap();
    writeln!(o, "#![allow(clippy::all, dead_code)]\n").unwrap();
    writeln!(o, "use crate::rt::matvec::*;").unwrap();
    writeln!(o, "use crate::rt::ops::*;").unwrap();
    writeln!(o, "use crate::rt::pool::Pool;").unwrap();
    writeln!(o, "use crate::rt::quant::*;").unwrap();
    writeln!(o, "use crate::rt::state::*;\n").unwrap();

    writeln!(o, "pub const DIMS: Dims = Dims {{").unwrap();
    writeln!(o, "    dim: {},", d.dim).unwrap();
    writeln!(o, "    hidden: {},", d.hidden).unwrap();
    writeln!(o, "    n_layer: {},", d.n_layer).unwrap();
    writeln!(o, "    n_head: {},", d.n_head).unwrap();
    writeln!(o, "    n_kv_head: {},", d.n_kv_head).unwrap();
    writeln!(o, "    head_dim: {},", d.head_dim).unwrap();
    writeln!(o, "    rot_dim: {},", d.rot_dim).unwrap();
    writeln!(o, "    vocab: {},", d.vocab).unwrap();
    writeln!(o, "    n_ctx_train: {},", d.n_ctx_train).unwrap();
    writeln!(o, "    eps: {:?},", d.eps).unwrap();
    writeln!(o, "    rope_base: {:?},", d.rope_base).unwrap();
    writeln!(o, "    rope_scale: {:?},", d.rope_scale).unwrap();
    writeln!(o, "}};\n").unwrap();

    writeln!(o, "pub const INFO: ModelInfo = ModelInfo {{").unwrap();
    writeln!(o, "    name: {:?},", spec.name).unwrap();
    writeln!(o, "    arch: {:?},", spec.arch).unwrap();
    writeln!(o, "    file_type: {:?},", spec.file_type).unwrap();
    writeln!(o, "    source: {:?},", spec.source).unwrap();
    writeln!(o, "    n_params: {},", spec.n_params).unwrap();
    writeln!(o, "    weights_len: {},", spec.data_len).unwrap();
    writeln!(o, "    n_tensors: {},", spec.n_tensors).unwrap();
    writeln!(o, "    quant_mix: {:?},", spec.quant_mix).unwrap();
    writeln!(o, "    compiler_version: {:?},", opts.compiler_version).unwrap();
    writeln!(o, "}};\n").unwrap();

    writeln!(o, "pub const TOKEN_EMBD: Tensor = {};", tensor_lit(&spec.token_embd)).unwrap();
    writeln!(o, "pub const OUTPUT_NORM: Tensor = {};", tensor_lit(&spec.output_norm)).unwrap();
    match &spec.output {
        Some(t) => writeln!(o, "pub const OUTPUT: Tensor = {};", tensor_lit(t)).unwrap(),
        None => writeln!(o, "/// Logits head tied to the token embedding.\npub const OUTPUT: Tensor = TOKEN_EMBD;").unwrap(),
    }
    match &spec.rope_freqs {
        Some(t) => writeln!(o, "pub const ROPE_FREQS: Option<Tensor> = Some({});", tensor_lit(t)).unwrap(),
        None => writeln!(o, "pub const ROPE_FREQS: Option<Tensor> = None;").unwrap(),
    }
    writeln!(o).unwrap();
    writeln!(o, "pub struct Layer {{").unwrap();
    for f in ["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "gate", "up", "down"] {
        writeln!(o, "    pub {f}: Tensor,").unwrap();
    }
    writeln!(o, "}}\n").unwrap();
    writeln!(o, "pub static LAYERS: [Layer; {}] = [", d.n_layer).unwrap();
    for l in &spec.layers {
        writeln!(o, "    Layer {{").unwrap();
        for (f, t) in [
            ("attn_norm", &l.attn_norm),
            ("wq", &l.wq),
            ("wk", &l.wk),
            ("wv", &l.wv),
            ("wo", &l.wo),
            ("ffn_norm", &l.ffn_norm),
            ("gate", &l.gate),
            ("up", &l.up),
            ("down", &l.down),
        ] {
            writeln!(o, "        {f}: {},", tensor_lit(t)).unwrap();
        }
        writeln!(o, "    }},").unwrap();
    }
    writeln!(o, "];\n").unwrap();

    for (i, l) in spec.layers.iter().enumerate() {
        emit_layer(&mut o, i, l, spec);
    }

    let out_t = spec.output.as_ref().unwrap_or(&spec.token_embd);
    writeln!(o, "/// One decoding step: feeds `token` at `pos` through every block and,").unwrap();
    writeln!(o, "/// when `want_logits` is set, computes the vocabulary logits into `s.logits`.").unwrap();
    writeln!(o, "pub fn forward(s: &mut State, w: &Weights, pool: &Pool, token: u32, pos: usize, want_logits: bool) {{").unwrap();
    writeln!(o, "    assert!((token as usize) < DIMS.vocab, \"token id out of range\");").unwrap();
    writeln!(o, "    assert!(pos < s.ctx, \"position exceeds the context\");").unwrap();
    writeln!(o, "    dequant_row_{}(w.row(&TOKEN_EMBD, token as usize), &mut s.x);", spec.token_embd.kind.fn_suffix()).unwrap();
    writeln!(o, "    rope_cos_sin(&mut s.rope_cs, &s.inv_freq, pos as f32 / DIMS.rope_scale);").unwrap();
    for i in 0..d.n_layer {
        writeln!(o, "    layer_{i}(s, w, pool, pos);").unwrap();
    }
    writeln!(o, "    if !want_logits {{").unwrap();
    writeln!(o, "        return;").unwrap();
    writeln!(o, "    }}").unwrap();
    writeln!(o, "    rmsnorm(&mut s.xb, &s.x, w.f32s(&OUTPUT_NORM), DIMS.eps);").unwrap();
    emit_quantize(&mut o, "xb", [out_t.kind].into_iter());
    emit_matvec(&mut o, out_t, "OUTPUT", "logits", "xb");
    writeln!(o, "}}").unwrap();
    o
}

/// Generate `payload.rs`: weight + tokenizer embedding.
pub fn emit_payload_rs(spec: &ModelSpec, opts: &EmitOptions) -> String {
    let mut o = String::new();
    writeln!(o, "//! Generated by aot-llm: how the weights and tokenizer reach the runtime.\n").unwrap();
    writeln!(o, "use std::path::Path;\n").unwrap();
    writeln!(o, "/// Tokenizer tables, embedded as read-only data.").unwrap();
    writeln!(o, "static TOKENIZER: &[u8] = include_bytes!(\"../tokenizer.bin\");\n").unwrap();
    writeln!(o, "pub fn tokenizer() -> &'static [u8] {{\n    TOKENIZER\n}}\n").unwrap();
    if opts.sidecar {
        writeln!(o, "pub const SIDECAR: bool = true;\n").unwrap();
        writeln!(o, "/// Memory-map the sidecar weights file (`<exe>.weights` unless `--weights` is given).").unwrap();
        writeln!(o, "pub fn weights(path: Option<&Path>) -> std::io::Result<&'static [u8]> {{").unwrap();
        writeln!(o, "    let p = match path {{").unwrap();
        writeln!(o, "        Some(p) => p.to_path_buf(),").unwrap();
        writeln!(o, "        None => {{").unwrap();
        writeln!(o, "            let exe = std::env::current_exe()?;").unwrap();
        writeln!(o, "            let mut s = exe.into_os_string();").unwrap();
        writeln!(o, "            s.push(\".weights\");").unwrap();
        writeln!(o, "            std::path::PathBuf::from(s)").unwrap();
        writeln!(o, "        }}").unwrap();
        writeln!(o, "    }};").unwrap();
        writeln!(o, "    crate::rt::weights::mmap_file(&p)").unwrap();
        writeln!(o, "}}").unwrap();
        return o;
    }
    let path = opts.gguf_path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
    let skip = spec.data_offset;
    let len = spec.data_len;
    writeln!(o, "pub const SIDECAR: bool = false;\n").unwrap();
    writeln!(o, "/// The tensor data section of the GGUF file is placed verbatim in a").unwrap();
    writeln!(o, "/// read-only section of this executable with the assembler's `.incbin`").unwrap();
    writeln!(o, "/// directive. The OS loader maps the section lazily: starting the binary").unwrap();
    writeln!(o, "/// costs no I/O and weights are paged in on first touch (zero-copy).").unwrap();
    writeln!(o, "pub const WEIGHTS_LEN: usize = {len};\n").unwrap();
    writeln!(o, "#[cfg(target_os = \"macos\")]").unwrap();
    writeln!(o, "std::arch::global_asm!(").unwrap();
    // A private section: Mach-O arm64 relocation addends are 24-bit, so the
    // compiler's own constants must not share a section with the blob.
    writeln!(o, "    \".section __TEXT,__aot_weights\",").unwrap();
    writeln!(o, "    \".balign 64\",").unwrap();
    writeln!(o, "    \".globl _aot_weights_start\",").unwrap();
    writeln!(o, "    \"_aot_weights_start:\",").unwrap();
    writeln!(o, "    \".incbin \\\"{path}\\\", {skip}, {len}\",").unwrap();
    writeln!(o, "    \".balign 64\",").unwrap();
    writeln!(o, ");\n").unwrap();
    writeln!(o, "#[cfg(not(target_os = \"macos\"))]").unwrap();
    writeln!(o, "std::arch::global_asm!(").unwrap();
    writeln!(o, "    \".section .rodata.aot_weights,\\\"a\\\",@progbits\",").unwrap();
    writeln!(o, "    \".balign 64\",").unwrap();
    writeln!(o, "    \".globl aot_weights_start\",").unwrap();
    writeln!(o, "    \"aot_weights_start:\",").unwrap();
    writeln!(o, "    \".incbin \\\"{path}\\\", {skip}, {len}\",").unwrap();
    writeln!(o, "    \".balign 64\",").unwrap();
    writeln!(o, "    \".text\",").unwrap();
    writeln!(o, ");\n").unwrap();
    writeln!(o, "extern \"C\" {{").unwrap();
    writeln!(o, "    static aot_weights_start: u8;").unwrap();
    writeln!(o, "}}\n").unwrap();
    writeln!(o, "pub fn weights(_path: Option<&Path>) -> std::io::Result<&'static [u8]> {{").unwrap();
    writeln!(o, "    // SAFETY: the symbol marks WEIGHTS_LEN bytes of read-only data placed").unwrap();
    writeln!(o, "    // by the assembler directive above; it lives for the whole process.").unwrap();
    writeln!(o, "    Ok(unsafe {{ std::slice::from_raw_parts(std::ptr::addr_of!(aot_weights_start), WEIGHTS_LEN) }})").unwrap();
    writeln!(o, "}}").unwrap();
    o
}

pub fn emit_main_rs(spec: &ModelSpec) -> String {
    format!(
        "//! Generated by aot-llm: standalone inference binary for `{}`.\n\
         #![allow(dead_code, unused_imports, clippy::all)]\n\n\
         mod model;\n\
         mod payload;\n\
         mod rt;\n\n\
         fn main() {{\n\
         \x20   let runtime = rt::app::Runtime {{\n\
         \x20       dims: &model::DIMS,\n\
         \x20       info: &model::INFO,\n\
         \x20       forward: model::forward,\n\
         \x20       rope_freqs: model::ROPE_FREQS,\n\
         \x20       tokenizer_blob: payload::tokenizer(),\n\
         \x20       weights: payload::weights,\n\
         \x20       sidecar: payload::SIDECAR,\n\
         \x20   }};\n\
         \x20   std::process::exit(rt::app::main(runtime));\n\
         }}\n",
        spec.name
    )
}

pub fn emit_cargo_toml(opts: &EmitOptions) -> String {
    format!(
        "[package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\
         publish = false\n\n\
         [[bin]]\n\
         name = \"{name}\"\n\
         path = \"src/main.rs\"\n\n\
         [dependencies]\n\n\
         [profile.release]\n\
         opt-level = 3\n\
         lto = \"fat\"\n\
         codegen-units = 1\n\
         panic = \"abort\"\n\
         strip = true\n\
         debug = false\n\
         incremental = false\n",
        name = opts.crate_name
    )
}

pub fn emit_readme(spec: &ModelSpec, opts: &EmitOptions) -> String {
    format!(
        "# {name}\n\n\
         Standalone inference binary generated by aot-llm {ver} from `{src}`.\n\n\
         * `src/model.rs`   - static forward pass and tensor table (generated)\n\
         * `src/payload.rs` - weight / tokenizer embedding (generated)\n\
         * `src/rt/`        - runtime kernels (copied from aot-kernels)\n\
         * `tokenizer.bin`  - serialised vocabulary\n\n\
         Rebuild with `cargo build --release`. {note}\n",
        name = opts.crate_name,
        ver = opts.compiler_version,
        src = spec.source,
        note = if opts.sidecar {
            "Weights are read from `<binary>.weights` at run time."
        } else {
            "The weights are read from the original GGUF file at build time (see `.incbin` in `src/payload.rs`), so keep it in place while rebuilding."
        }
    )
}

/// Write the whole project into `dir` (created if needed).
pub fn write_project(dir: &Path, spec: &ModelSpec, tokenizer_blob: &[u8], opts: &EmitOptions) -> Result<()> {
    let src = dir.join("src");
    let rt = src.join("rt");
    fs::create_dir_all(&rt).with_context(|| format!("creating {}", rt.display()))?;
    fs::write(dir.join("Cargo.toml"), emit_cargo_toml(opts))?;
    fs::write(dir.join("README.md"), emit_readme(spec, opts))?;
    fs::write(dir.join("tokenizer.bin"), tokenizer_blob)?;
    fs::write(src.join("main.rs"), emit_main_rs(spec))?;
    fs::write(src.join("model.rs"), emit_model_rs(spec, opts))?;
    fs::write(src.join("payload.rs"), emit_payload_rs(spec, opts))?;
    for (name, contents) in RT_SOURCES {
        fs::write(rt.join(name), contents)?;
    }
    let cargo_dir = dir.join(".cargo");
    if opts.native {
        fs::create_dir_all(&cargo_dir)?;
        fs::write(cargo_dir.join("config.toml"), "[build]\nrustflags = [\"-C\", \"target-cpu=native\"]\n")?;
    } else if cargo_dir.exists() {
        let _ = fs::remove_file(cargo_dir.join("config.toml"));
    }
    // A stale lock file from an older toolchain is harmless but confusing.
    let _ = fs::remove_file(dir.join("Cargo.lock"));
    Ok(())
}
