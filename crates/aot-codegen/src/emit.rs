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
    /// Target triple the project will be built for (affects cargo config).
    pub target: Option<String>,
    /// Absolute path of a baked KV prefix blob to embed (see `rt/prefix.rs`).
    pub prefix_kv_path: Option<PathBuf>,
    /// GPU backend to compile in (the binary can still run on the CPU).
    pub gpu: Option<GpuBackend>,
    /// Produce a model library with the C API of `aot_llm.h` instead of an
    /// executable.
    pub lib: Option<LibKind>,
    /// Prefix of the exported C symbols and header identifiers (`--lib`
    /// only): `qwen3_` turns `aot_open` into `qwen3_aot_open`, so several
    /// model libraries can be linked into one program.
    pub symbol_prefix: String,
}

/// Library flavours of `aot-llm compile --lib`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibKind {
    /// `lib<name>.a`, linked into the application.
    Static,
    /// `lib<name>.dylib` / `lib<name>.so`.
    Shared,
}

impl LibKind {
    pub fn parse(s: &str) -> Option<LibKind> {
        match s {
            "static" | "staticlib" | "a" => Some(LibKind::Static),
            "shared" | "dynamic" | "cdylib" | "dylib" | "so" => Some(LibKind::Shared),
            _ => None,
        }
    }

    pub fn crate_type(self) -> &'static str {
        match self {
            LibKind::Static => "staticlib",
            LibKind::Shared => "cdylib",
        }
    }

    /// File name of the library cargo produces for `target` (host when `None`).
    pub fn file_name(self, crate_name: &str, target: Option<&str>) -> String {
        let windows = target.map(|t| t.contains("windows")).unwrap_or(cfg!(target_os = "windows"));
        let macos = target.map(|t| t.contains("apple")).unwrap_or(cfg!(target_vendor = "apple"));
        match (self, windows, macos) {
            (LibKind::Static, true, _) => format!("{crate_name}.lib"),
            (LibKind::Static, false, _) => format!("lib{crate_name}.a"),
            (LibKind::Shared, true, _) => format!("{crate_name}.dll"),
            (LibKind::Shared, false, true) => format!("lib{crate_name}.dylib"),
            (LibKind::Shared, false, false) => format!("lib{crate_name}.so"),
        }
    }
}

/// GPU backends the generated binary can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// Apple Metal (macOS only; see `rt/metal.rs`).
    Metal,
}

impl GpuBackend {
    pub fn name(self) -> &'static str {
        match self {
            GpuBackend::Metal => "metal",
        }
    }
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
    format!("Tensor {{ name: {:?}, kind: Kind::{}, off: {}, len: {}, rows: {}, cols: {} }}", t.name, t.kind.rust_name(), t.off, t.len, t.rows, t.cols)
}

/// The activation operand of a matrix-vector product whose input is the
/// first `n` floats of `s.{src}`.
fn act_expr(a: Activation, src: &str, n: usize) -> String {
    match a {
        Activation::Q8_0 => "&s.aq0".into(),
        Activation::Q8_K => "&s.aqk".into(),
        Activation::F32 => format!("&s.{src}[..{n}]"),
    }
}

/// Emit the activation quantization needed by `kinds` for the first `n`
/// floats of `src` (buffers can be longer than the vector they hold: `xb`
/// also receives the attention output of `q_dim` floats).
fn emit_quantize(out: &mut String, src: &str, n: usize, kinds: impl Iterator<Item = Kind>) {
    let acts: BTreeSet<Activation> = kinds.map(Kind::activation).collect();
    for a in acts {
        match a {
            Activation::Q8_0 => writeln!(out, "    quantize_q8_0(&s.{src}[..{n}], &mut s.aq0);").unwrap(),
            Activation::Q8_K => writeln!(out, "    quantize_q8_k(&s.{src}[..{n}], &mut s.aqk);").unwrap(),
            Activation::F32 => {}
        }
    }
}

/// Batched activation quantization over `m` rows of `src` (row length `n`),
/// spread over the pool.
fn emit_quantize_rows(out: &mut String, src: &str, n: usize, kinds: impl Iterator<Item = Kind>) {
    let acts: BTreeSet<Activation> = kinds.map(Kind::activation).collect();
    for a in acts {
        match a {
            Activation::Q8_0 => writeln!(out, "    quantize_q8_0_par(pool, &s.{src}[..m * {n}], &mut s.baq0);").unwrap(),
            Activation::Q8_K => writeln!(out, "    quantize_q8_k_par(pool, &s.{src}[..m * {n}], &mut s.baqk);").unwrap(),
            Activation::F32 => {}
        }
    }
}

fn bact_expr(a: Activation, src: &str, n: usize) -> String {
    match a {
        Activation::Q8_0 => "&s.baq0".into(),
        Activation::Q8_K => "&s.baqk".into(),
        Activation::F32 => format!("&s.{src}[..m * {n}]"),
    }
}

fn emit_matmul(out: &mut String, t: &TensorSpec, texpr: &str, dst: &str, src: &str) {
    writeln!(out, "    matmul_{}(pool, &mut s.{dst}, w.bytes(&{texpr}), {}, m, {}, {});", t.kind.fn_suffix(), bact_expr(t.kind.activation(), src, t.cols), t.rows, t.cols).unwrap();
}

fn emit_layer_batch(out: &mut String, i: usize, l: &LayerSpec, spec: &ModelSpec) {
    let d = &spec.dims;
    let f = &spec.flags;
    let (dim, hidden, qd) = (d.dim, d.hidden, d.n_head * d.head_dim);
    let cs = if l.local_attn { "s.brope_cs_local" } else { "s.brope_cs" };
    let window = if l.local_attn { d.swa_window } else { 0 };
    writeln!(out, "/// Transformer block {i} over `m` consecutive prompt tokens starting at `pos0`.").unwrap();
    writeln!(out, "#[inline(never)]").unwrap();
    writeln!(out, "fn layer_{i}_batch(s: &mut State, w: &Weights, pool: &Pool, pos0: usize, m: usize) {{").unwrap();
    writeln!(out, "    let l = &LAYERS[{i}];").unwrap();
    writeln!(out, "    rmsnorm_rows_par(pool, &mut s.bxb, &s.bx, w.f32s(&l.attn_norm), DIMS.eps, m);").unwrap();
    emit_quantize_rows(out, "bxb", dim, [l.wq.kind, l.wk.kind, l.wv.kind].into_iter());
    emit_matmul(out, &l.wq, "l.wq", "bq", "bxb");
    emit_matmul(out, &l.wk, "l.wk", "bk", "bxb");
    emit_matmul(out, &l.wv, "l.wv", "bv", "bxb");
    if f.qkv_bias {
        writeln!(out, "    add_bias_rows(&mut s.bq, w.f32s(&l.bq), m);").unwrap();
        writeln!(out, "    add_bias_rows(&mut s.bk, w.f32s(&l.bk), m);").unwrap();
        writeln!(out, "    add_bias_rows(&mut s.bv, w.f32s(&l.bv), m);").unwrap();
    }
    if f.qk_norm {
        writeln!(out, "    rmsnorm_heads_rows(&mut s.bq, w.f32s(&l.q_norm), m, {}, {}, DIMS.eps);", d.n_head, d.head_dim).unwrap();
        writeln!(out, "    rmsnorm_heads_rows(&mut s.bk, w.f32s(&l.k_norm), m, {}, {}, DIMS.eps);", d.n_kv_head, d.head_dim).unwrap();
    }
    let rope_fn = if f.rope_neox { "rope_neox_rows" } else { "rope_rows" };
    writeln!(out, "    {rope_fn}(&mut s.bq, m, {}, {}, {}, &{cs});", d.n_head, d.head_dim, d.rot_dim).unwrap();
    writeln!(out, "    {rope_fn}(&mut s.bk, m, {}, {}, {}, &{cs});", d.n_kv_head, d.head_dim, d.rot_dim).unwrap();
    writeln!(out, "    store_kv_rows(&mut s.k_cache, &mut s.v_cache, &s.bk, &s.bv, &DIMS, s.ctx, {i}, pos0, m);").unwrap();
    writeln!(out, "    attention_rows(pool, &mut s.bxb, &s.bq, &s.k_cache, &s.v_cache, &mut s.scores, &DIMS, s.ctx, {i}, pos0, m, {window});").unwrap();
    emit_quantize_rows(out, "bxb", qd, [l.wo.kind].into_iter());
    emit_matmul(out, &l.wo, "l.wo", "bxb2", "bxb");
    if f.post_norms {
        // Gemma: the block output is normalised before it joins the residual stream.
        writeln!(out, "    rmsnorm_rows_par(pool, &mut s.bxb, &s.bxb2, w.f32s(&l.attn_post_norm), DIMS.eps, m);").unwrap();
        writeln!(out, "    add_inplace_par(pool, &mut s.bx[..m * {dim}], &s.bxb[..m * {dim}]);").unwrap();
    } else {
        writeln!(out, "    add_inplace_par(pool, &mut s.bx[..m * {dim}], &s.bxb2[..m * {dim}]);").unwrap();
    }
    writeln!(out, "    rmsnorm_rows_par(pool, &mut s.bxb, &s.bx, w.f32s(&l.ffn_norm), DIMS.eps, m);").unwrap();
    emit_quantize_rows(out, "bxb", dim, [l.gate.kind, l.up.kind].into_iter());
    let fused = l.gate.kind == l.up.kind && l.gate.kind.activation() != Activation::F32;
    if fused {
        let (fn_name, dot_fn, dot4_fn) = match l.gate.kind.activation() {
            Activation::Q8_K => ("matmul2_q8_k_input", "dot_fn_q8_k", "dot4_fn_q8_k"),
            _ => ("matmul2_q8_0_input", "dot_fn_q8_0", "dot4_fn_q8_0"),
        };
        let kn = l.gate.kind.rust_name();
        writeln!(
            out,
            "    {fn_name}(pool, &mut s.bhb, &mut s.bhb2, w.bytes(&l.gate), w.bytes(&l.up), {}, m, {}, Kind::{kn}.row_bytes({}), {dot_fn}(Kind::{kn}), {dot4_fn}(Kind::{kn}));",
            bact_expr(l.gate.kind.activation(), "bxb", dim),
            l.gate.rows,
            l.gate.cols,
        )
        .unwrap();
    } else {
        emit_matmul(out, &l.gate, "l.gate", "bhb", "bxb");
        emit_matmul(out, &l.up, "l.up", "bhb2", "bxb");
    }
    let act = if f.gelu { "gelu_mul_par" } else { "silu_mul_par" };
    writeln!(out, "    {act}(pool, &mut s.bhb[..m * {hidden}], &s.bhb2[..m * {hidden}]);").unwrap();
    emit_quantize_rows(out, "bhb", hidden, [l.down.kind].into_iter());
    emit_matmul(out, &l.down, "l.down", "bxb2", "bhb");
    if f.post_norms {
        writeln!(out, "    rmsnorm_rows_par(pool, &mut s.bxb, &s.bxb2, w.f32s(&l.ffn_post_norm), DIMS.eps, m);").unwrap();
        writeln!(out, "    add_inplace_par(pool, &mut s.bx[..m * {dim}], &s.bxb[..m * {dim}]);").unwrap();
    } else {
        writeln!(out, "    add_inplace_par(pool, &mut s.bx[..m * {dim}], &s.bxb2[..m * {dim}]);").unwrap();
    }
    writeln!(out, "}}\n").unwrap();
}

fn emit_matvec(out: &mut String, t: &TensorSpec, texpr: &str, dst: &str, src: &str) {
    writeln!(out, "    matvec_{}(pool, &mut s.{dst}, w.bytes(&{texpr}), {}, {}, {});", t.kind.fn_suffix(), act_expr(t.kind.activation(), src, t.cols), t.rows, t.cols).unwrap();
}

fn emit_layer(out: &mut String, i: usize, l: &LayerSpec, spec: &ModelSpec) {
    let d = &spec.dims;
    let f = &spec.flags;
    let (dim, qd) = (d.dim, d.n_head * d.head_dim);
    let cs = if l.local_attn { "s.rope_cs_local" } else { "s.rope_cs" };
    let window = if l.local_attn { d.swa_window } else { 0 };
    writeln!(out, "/// Transformer block {i}{}.", if l.local_attn { " (sliding-window attention)" } else { "" }).unwrap();
    writeln!(out, "#[inline(never)]").unwrap();
    writeln!(out, "fn layer_{i}(s: &mut State, w: &Weights, pool: &Pool, pos: usize) {{").unwrap();
    writeln!(out, "    let l = &LAYERS[{i}];").unwrap();
    // Attention.
    writeln!(out, "    rmsnorm(&mut s.xb, &s.x, w.f32s(&l.attn_norm), DIMS.eps);").unwrap();
    emit_quantize(out, "xb", dim, [l.wq.kind, l.wk.kind, l.wv.kind].into_iter());
    let same = l.wq.kind == l.wk.kind && l.wk.kind == l.wv.kind && l.wq.kind.activation() != Activation::F32;
    if same {
        let (fn_name, dot_fn) = match l.wq.kind.activation() {
            Activation::Q8_K => ("matvec3_q8_k_input", "dot_fn_q8_k"),
            _ => ("matvec3_q8_0_input", "dot_fn_q8_0"),
        };
        let kn = l.wq.kind.rust_name();
        writeln!(
            out,
            "    {fn_name}(pool, [&mut s.q, &mut s.k, &mut s.v], [w.bytes(&l.wq), w.bytes(&l.wk), w.bytes(&l.wv)], {}, [{}, {}, {}], Kind::{kn}.row_bytes({}), {dot_fn}(Kind::{kn}));",
            act_expr(l.wq.kind.activation(), "xb", dim),
            l.wq.rows,
            l.wk.rows,
            l.wv.rows,
            l.wq.cols
        )
        .unwrap();
    } else {
        emit_matvec(out, &l.wq, "l.wq", "q", "xb");
        emit_matvec(out, &l.wk, "l.wk", "k", "xb");
        emit_matvec(out, &l.wv, "l.wv", "v", "xb");
    }
    if f.qkv_bias {
        writeln!(out, "    add_inplace(&mut s.q, w.f32s(&l.bq));").unwrap();
        writeln!(out, "    add_inplace(&mut s.k, w.f32s(&l.bk));").unwrap();
        writeln!(out, "    add_inplace(&mut s.v, w.f32s(&l.bv));").unwrap();
    }
    if f.qk_norm {
        writeln!(out, "    rmsnorm_heads(&mut s.q, w.f32s(&l.q_norm), {}, {}, DIMS.eps);", d.n_head, d.head_dim).unwrap();
        writeln!(out, "    rmsnorm_heads(&mut s.k, w.f32s(&l.k_norm), {}, {}, DIMS.eps);", d.n_kv_head, d.head_dim).unwrap();
    }
    let rope_fn = if f.rope_neox { "rope_neox" } else { "rope" };
    writeln!(out, "    {rope_fn}(&mut s.q, {}, {}, {}, &{cs});", d.n_head, d.head_dim, d.rot_dim).unwrap();
    writeln!(out, "    {rope_fn}(&mut s.k, {}, {}, {}, &{cs});", d.n_kv_head, d.head_dim, d.rot_dim).unwrap();
    writeln!(out, "    store_kv(&mut s.k_cache, &mut s.v_cache, &s.k, &s.v, &DIMS, s.ctx, {i}, pos);").unwrap();
    writeln!(out, "    attention(pool, &mut s.xb, &s.q, &s.k_cache, &s.v_cache, &mut s.scores, &DIMS, s.ctx, {i}, pos, {window});").unwrap();
    emit_quantize(out, "xb", qd, [l.wo.kind].into_iter());
    emit_matvec(out, &l.wo, "l.wo", "xb2", "xb");
    if f.post_norms {
        writeln!(out, "    rmsnorm(&mut s.xb, &s.xb2[..{dim}], w.f32s(&l.attn_post_norm), DIMS.eps);").unwrap();
        writeln!(out, "    add_inplace(&mut s.x, &s.xb[..{dim}]);").unwrap();
    } else {
        writeln!(out, "    add_inplace(&mut s.x, &s.xb2[..{dim}]);").unwrap();
    }
    // Feed-forward.
    writeln!(out, "    rmsnorm(&mut s.xb, &s.x, w.f32s(&l.ffn_norm), DIMS.eps);").unwrap();
    emit_quantize(out, "xb", dim, [l.gate.kind, l.up.kind].into_iter());
    let fused = l.gate.kind == l.up.kind && l.gate.kind.activation() != Activation::F32;
    if fused {
        let (fn_name, dot_fn) = match l.gate.kind.activation() {
            Activation::Q8_K => ("matvec2_q8_k_input", "dot_fn_q8_k"),
            _ => ("matvec2_q8_0_input", "dot_fn_q8_0"),
        };
        writeln!(
            out,
            "    {fn_name}(pool, &mut s.hb, &mut s.hb2, w.bytes(&l.gate), w.bytes(&l.up), {}, {}, Kind::{}.row_bytes({}), {dot_fn}(Kind::{}));",
            act_expr(l.gate.kind.activation(), "xb", dim),
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
    writeln!(out, "    {}(&mut s.hb, &s.hb2);", if f.gelu { "gelu_mul" } else { "silu_mul" }).unwrap();
    emit_quantize(out, "hb", d.hidden, [l.down.kind].into_iter());
    emit_matvec(out, &l.down, "l.down", "xb2", "hb");
    if f.post_norms {
        writeln!(out, "    rmsnorm(&mut s.xb, &s.xb2[..{dim}], w.f32s(&l.ffn_post_norm), DIMS.eps);").unwrap();
        writeln!(out, "    add_inplace(&mut s.x, &s.xb[..{dim}]);").unwrap();
    } else {
        writeln!(out, "    add_inplace(&mut s.x, &s.xb2[..{dim}]);").unwrap();
    }
    writeln!(out, "}}\n").unwrap();
}

/// Names of the per-layer tensor fields of the generated `Layer` struct.
fn layer_fields(spec: &ModelSpec) -> Vec<&'static str> {
    let mut v = vec!["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "gate", "up", "down"];
    if spec.flags.qkv_bias {
        v.extend(["bq", "bk", "bv"]);
    }
    if spec.flags.qk_norm {
        v.extend(["q_norm", "k_norm"]);
    }
    if spec.flags.post_norms {
        v.extend(["attn_post_norm", "ffn_post_norm"]);
    }
    v
}

fn layer_tensor<'a>(l: &'a LayerSpec, field: &str) -> &'a TensorSpec {
    match field {
        "attn_norm" => &l.attn_norm,
        "wq" => &l.wq,
        "wk" => &l.wk,
        "wv" => &l.wv,
        "wo" => &l.wo,
        "ffn_norm" => &l.ffn_norm,
        "gate" => &l.gate,
        "up" => &l.up,
        "down" => &l.down,
        "bq" => l.bq.as_ref().expect("bias"),
        "bk" => l.bk.as_ref().expect("bias"),
        "bv" => l.bv.as_ref().expect("bias"),
        "q_norm" => l.q_norm.as_ref().expect("q_norm"),
        "k_norm" => l.k_norm.as_ref().expect("k_norm"),
        "attn_post_norm" => l.attn_post_norm.as_ref().expect("attn_post_norm"),
        "ffn_post_norm" => l.ffn_post_norm.as_ref().expect("ffn_post_norm"),
        _ => unreachable!(),
    }
}

/// Generate `model.rs`: constants, the tensor table and the static forward pass.
pub fn emit_model_rs(spec: &ModelSpec, opts: &EmitOptions) -> String {
    let d = &spec.dims;
    let mut o = String::new();
    writeln!(o, "//! Generated by aot-llm {} from `{}`. Do not edit.", opts.compiler_version, spec.source).unwrap();
    writeln!(o, "//!").unwrap();
    writeln!(o, "//! Model: {} ({}, {})", spec.name, spec.arch, spec.file_type).unwrap();
    let attn = match (spec.flags.qkv_bias, spec.flags.qk_norm, spec.flags.rope_neox) {
        (true, _, true) => "attention(q/k/v + bias, NEOX RoPE)",
        (_, true, true) if spec.flags.post_norms => "attention(per-head q/k RMSNorm, NEOX RoPE, sliding window) -> post-norm",
        (_, true, true) => "attention(per-head q/k RMSNorm, NEOX RoPE)",
        (_, _, true) => "attention(NEOX RoPE)",
        _ => "attention",
    };
    writeln!(o, "//! Static pipeline: embedding -> [rmsnorm -> {attn} -> residual -> rmsnorm -> SwiGLU FFN -> residual] x {} -> rmsnorm -> logits", d.n_layer).unwrap();
    writeln!(o, "#![allow(clippy::all, dead_code)]\n").unwrap();
    writeln!(o, "use crate::rt::matvec::*;").unwrap();
    writeln!(o, "use crate::rt::ops::*;").unwrap();
    writeln!(o, "use crate::rt::pool::Pool;").unwrap();
    writeln!(o, "use crate::rt::quant::*;").unwrap();
    writeln!(o, "use crate::rt::state::*;").unwrap();
    if opts.gpu.is_some() {
        writeln!(o, "#[cfg(target_os = \"macos\")]").unwrap();
        writeln!(o, "use crate::rt::metal::{{BufId, Gpu, MvOpts}};").unwrap();
    }
    writeln!(o).unwrap();

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
    writeln!(o, "    swa_window: {},", d.swa_window).unwrap();
    writeln!(o, "    rope_base_local: {:?},", d.rope_base_local).unwrap();
    writeln!(o, "    rope_scale_local: {:?},", d.rope_scale_local).unwrap();
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
    let fields = layer_fields(spec);
    writeln!(o, "pub struct Layer {{").unwrap();
    for f in &fields {
        writeln!(o, "    pub {f}: Tensor,").unwrap();
    }
    writeln!(o, "}}\n").unwrap();
    writeln!(o, "pub static LAYERS: [Layer; {}] = [", d.n_layer).unwrap();
    for l in &spec.layers {
        writeln!(o, "    Layer {{").unwrap();
        for f in &fields {
            writeln!(o, "        {f}: {},", tensor_lit(layer_tensor(l, f))).unwrap();
        }
        writeln!(o, "    }},").unwrap();
    }
    writeln!(o, "];\n").unwrap();

    for (i, l) in spec.layers.iter().enumerate() {
        emit_layer(&mut o, i, l, spec);
    }
    for (i, l) in spec.layers.iter().enumerate() {
        emit_layer_batch(&mut o, i, l, spec);
    }

    let out_t = spec.output.as_ref().unwrap_or(&spec.token_embd);
    writeln!(o, "/// One decoding step: feeds `token` at `pos` through every block and,").unwrap();
    writeln!(o, "/// when `want_logits` is set, computes the vocabulary logits into `s.logits`.").unwrap();
    writeln!(o, "pub fn forward(s: &mut State, w: &Weights, pool: &Pool, token: u32, pos: usize, want_logits: bool) {{").unwrap();
    writeln!(o, "    assert!((token as usize) < DIMS.vocab, \"token id out of range\");").unwrap();
    writeln!(o, "    assert!(pos < s.ctx, \"position exceeds the context\");").unwrap();
    writeln!(o, "    dequant_row_{}(w.row(&TOKEN_EMBD, token as usize), &mut s.x);", spec.token_embd.kind.fn_suffix()).unwrap();
    if spec.flags.embd_scale {
        writeln!(o, "    scale_inplace(&mut s.x, {:?});", (d.dim as f32).sqrt()).unwrap();
    }
    writeln!(o, "    rope_cos_sin(&mut s.rope_cs, &s.inv_freq, pos as f32 / DIMS.rope_scale);").unwrap();
    if d.swa_window > 0 {
        writeln!(o, "    rope_cos_sin(&mut s.rope_cs_local, &s.inv_freq_local, pos as f32 / DIMS.rope_scale_local);").unwrap();
    }
    for i in 0..d.n_layer {
        writeln!(o, "    layer_{i}(s, w, pool, pos);").unwrap();
    }
    writeln!(o, "    if !want_logits {{").unwrap();
    writeln!(o, "        return;").unwrap();
    writeln!(o, "    }}").unwrap();
    writeln!(o, "    rmsnorm(&mut s.xb, &s.x, w.f32s(&OUTPUT_NORM), DIMS.eps);").unwrap();
    emit_quantize(&mut o, "xb", d.dim, [out_t.kind].into_iter());
    emit_matvec(&mut o, out_t, "OUTPUT", "logits", "xb");
    writeln!(o, "}}\n").unwrap();

    writeln!(o, "/// Batched prompt step: processes `tokens` at positions `pos0 ..` through every").unwrap();
    writeln!(o, "/// block, reading each weight row once for all of them, and computes the").unwrap();
    writeln!(o, "/// logits of the last token when `want_logits` is set.").unwrap();
    writeln!(o, "pub fn forward_batch(s: &mut State, w: &Weights, pool: &Pool, tokens: &[u32], pos0: usize, want_logits: bool) {{").unwrap();
    writeln!(o, "    let m = tokens.len();").unwrap();
    writeln!(o, "    assert!(m >= 1 && m <= s.max_batch, \"batch size out of range\");").unwrap();
    writeln!(o, "    assert!(pos0 + m <= s.ctx, \"positions exceed the context\");").unwrap();
    writeln!(o, "    for (i, &t) in tokens.iter().enumerate() {{").unwrap();
    writeln!(o, "        assert!((t as usize) < DIMS.vocab, \"token id out of range\");").unwrap();
    writeln!(o, "        dequant_row_{}(w.row(&TOKEN_EMBD, t as usize), &mut s.bx[i * {dim}..(i + 1) * {dim}]);", spec.token_embd.kind.fn_suffix(), dim = d.dim).unwrap();
    writeln!(o, "    }}").unwrap();
    if spec.flags.embd_scale {
        writeln!(o, "    scale_inplace(&mut s.bx[..m * {}], {:?});", d.dim, (d.dim as f32).sqrt()).unwrap();
    }
    writeln!(o, "    rope_cos_sin_rows(&mut s.brope_cs, &s.inv_freq, pos0, m, DIMS.rope_scale);").unwrap();
    if d.swa_window > 0 {
        writeln!(o, "    rope_cos_sin_rows(&mut s.brope_cs_local, &s.inv_freq_local, pos0, m, DIMS.rope_scale_local);").unwrap();
    }
    for i in 0..d.n_layer {
        writeln!(o, "    layer_{i}_batch(s, w, pool, pos0, m);").unwrap();
    }
    writeln!(o, "    if !want_logits {{").unwrap();
    writeln!(o, "        return;").unwrap();
    writeln!(o, "    }}").unwrap();
    writeln!(o, "    let last = (m - 1) * {dim};", dim = d.dim).unwrap();
    writeln!(o, "    rmsnorm(&mut s.xb, &s.bx[last..last + {dim}], w.f32s(&OUTPUT_NORM), DIMS.eps);", dim = d.dim).unwrap();
    emit_quantize(&mut o, "xb", d.dim, [out_t.kind].into_iter());
    emit_matvec(&mut o, out_t, "OUTPUT", "logits", "xb");
    writeln!(o, "}}").unwrap();
    if opts.gpu.is_some() {
        emit_gpu_forwards(&mut o, spec);
    }
    o
}

/// The GPU forward passes: the same operator sequence as the CPU ones,
/// encoded into one Metal command buffer per step. Only the embedding lookup
/// and the RoPE tables are computed on the CPU (they are tiny and the buffers
/// are shared). The weight format of every matrix is fixed in the tensor
/// table, so `Gpu::matmul` picks the kernel for it without any graph
/// interpretation at run time.
fn emit_gpu_forwards(o: &mut String, spec: &ModelSpec) {
    let d = &spec.dims;
    let (dim, hidden) = (d.dim, d.hidden);
    let embd = spec.token_embd.kind.fn_suffix();
    // Decoding: 8 dispatches per block. RoPE is applied in the epilogue of
    // the q / k projections (k and v are written straight into the cache),
    // the gate / up projections and the gated product are one kernel when
    // both matrices share a format, and the residual adds are folded into
    // the wo / down products. RMSNorm stays a separate (12 us) dispatch:
    // folding it into the products was measured 2-3x slower. Models with
    // NEOX RoPE rotate in a separate dispatch (the pairs are half a head
    // apart); Gemma's post-norms accumulate into the residual stream.
    let f = &spec.flags;
    let (bq, bk, bv) = if f.qkv_bias { ("Some(&l.bq)", "Some(&l.bk)", "Some(&l.bv)") } else { ("None", "None", "None") };
    let gelu = f.gelu;
    // Per-head q / k RMSNorm (Qwen3, Gemma 3), in place, before RoPE.
    let qk_norm = |m: &str| -> String {
        if f.qk_norm {
            format!(
                "        g.rmsnorm_at(BufId::{q}, 0, BufId::{q}, 0, &l.q_norm, {hd}, {m} * {nh}, DIMS.eps);
        g.rmsnorm_at(BufId::KCache, kv_at, BufId::KCache, kv_at, &l.k_norm, {hd}, {m} * {nkv}, DIMS.eps);\n",
                q = if m == "1" { "Q" } else { "Bq" },
                hd = d.head_dim,
                nh = d.n_head,
                nkv = d.n_kv_head,
            )
        } else {
            String::new()
        }
    };
    let decode_block = |l: &LayerSpec| -> String {
        let (cs, window) = if l.local_attn { ("RopeCsLocal", d.swa_window) } else { ("RopeCs", 0) };
        let gate = if l.gate.kind == l.up.kind && l.gate.rows == l.up.rows {
            format!("        g.gate_up(BufId::Hb, &l.gate, &l.up, BufId::Xb, {gelu});\n")
        } else {
            format!(
                "        g.matmul(BufId::Hb, 0, &l.gate, BufId::Xb, 0, 1, false);
        g.matmul(BufId::Hb2, 0, &l.up, BufId::Xb, 0, 1, false);
        g.act_mul(BufId::Hb, BufId::Hb2, {hidden}, {gelu});\n"
            )
        };
        // The three projections share the input: one dispatch when they share
        // a format (`Gpu::mv3` falls back to three otherwise).
        let qkv = if f.rope_neox {
            format!(
                "        g.mv3([(BufId::Q, 0), (BufId::KCache, kv_at), (BufId::VCache, kv_at)], [&l.wq, &l.wk, &l.wv], BufId::Xb, None, None, [{bq}, {bk}, {bv}]);
{norm}        g.rope_ex(BufId::Q, BufId::KCache, kv_at, 1, BufId::{cs}, &DIMS, true);\n",
                norm = qk_norm("1")
            )
        } else {
            format!("        g.mv3([(BufId::Q, 0), (BufId::KCache, kv_at), (BufId::VCache, kv_at)], [&l.wq, &l.wk, &l.wv], BufId::Xb, Some(&DIMS), Some(BufId::{cs}), [{bq}, {bk}, {bv}]);\n")
        };
        // Residual joins: fused into the products, or normalised first (Gemma).
        let (wo, down) = if f.post_norms {
            (
                format!(
                    "        g.mv(BufId::Xb2, 0, &l.wo, BufId::Xb, 0, None, MvOpts::default());
        g.rmsnorm_add(BufId::X, 0, BufId::Xb2, 0, &l.attn_post_norm, {dim}, 1, DIMS.eps);\n"
                ),
                format!(
                    "        g.mv(BufId::Xb2, 0, &l.down, BufId::Hb, 0, None, MvOpts::default());
        g.rmsnorm_add(BufId::X, 0, BufId::Xb2, 0, &l.ffn_post_norm, {dim}, 1, DIMS.eps);\n"
                ),
            )
        } else {
            (
                "        g.mv(BufId::X, 0, &l.wo, BufId::Xb, 0, None, MvOpts { accumulate: true, ..MvOpts::default() });\n".to_string(),
                "        g.mv(BufId::X, 0, &l.down, BufId::Hb, 0, None, MvOpts { accumulate: true, ..MvOpts::default() });\n".to_string(),
            )
        };
        format!(
            "        let kv_at = (i * s.ctx + pos) * kv;
        g.rmsnorm(BufId::Xb, BufId::X, 0, &l.attn_norm, {dim}, 1, DIMS.eps);
{qkv}        g.attention(BufId::Xb, BufId::Q, i, pos, 1, &DIMS, s.ctx, {window});
{wo}        g.rmsnorm(BufId::Xb, BufId::X, 0, &l.ffn_norm, {dim}, 1, DIMS.eps);
{gate}{down}"
        )
    };
    // Prompt batches: the plain sequence; the batched kernels are compute
    // bound, so fusion matters less there.
    let batch_block = |l: &LayerSpec| -> String {
        let (cs, window) = if l.local_attn { ("BropeCsLocal", d.swa_window) } else { ("BropeCs", 0) };
        let (wo, down) = if f.post_norms {
            (
                format!(
                    "        g.matmul(BufId::Bxb2, 0, &l.wo, BufId::Bxb, 0, m, false);
        g.rmsnorm_add(BufId::Bx, 0, BufId::Bxb2, 0, &l.attn_post_norm, {dim}, m, DIMS.eps);\n"
                ),
                format!(
                    "        g.matmul(BufId::Bxb2, 0, &l.down, BufId::Bhb, 0, m, false);
        g.rmsnorm_add(BufId::Bx, 0, BufId::Bxb2, 0, &l.ffn_post_norm, {dim}, m, DIMS.eps);\n"
                ),
            )
        } else {
            ("        g.matmul(BufId::Bx, 0, &l.wo, BufId::Bxb, 0, m, true);\n".to_string(), "        g.matmul(BufId::Bx, 0, &l.down, BufId::Bhb, 0, m, true);\n".to_string())
        };
        format!(
            "        let kv_at = (i * s.ctx + pos0) * kv;
        g.rmsnorm(BufId::Bxb, BufId::Bx, 0, &l.attn_norm, {dim}, m, DIMS.eps);
        g.matmul_bias(BufId::Bq, 0, &l.wq, BufId::Bxb, 0, m, false, {bq});
        g.matmul_bias(BufId::KCache, kv_at, &l.wk, BufId::Bxb, 0, m, false, {bk});
        g.matmul_bias(BufId::VCache, kv_at, &l.wv, BufId::Bxb, 0, m, false, {bv});
{norm}        g.rope_ex(BufId::Bq, BufId::KCache, kv_at, m, BufId::{cs}, &DIMS, {neox});
        g.attention(BufId::Bxb, BufId::Bq, i, pos0, m, &DIMS, s.ctx, {window});
{wo}        g.rmsnorm(BufId::Bxb, BufId::Bx, 0, &l.ffn_norm, {dim}, m, DIMS.eps);
        g.matmul(BufId::Bhb, 0, &l.gate, BufId::Bxb, 0, m, false);
        g.matmul(BufId::Bhb2, 0, &l.up, BufId::Bxb, 0, m, false);
        g.act_mul(BufId::Bhb, BufId::Bhb2, m * {hidden}, {gelu});
{down}",
            norm = qk_norm("m"),
            neox = f.rope_neox,
        )
    };
    // Every layer has the same tensor layout, so one loop body serves all
    // of them unless layers differ (gate / up format mix, local vs global
    // attention); then the blocks are emitted per layer.
    let shape = |l: &LayerSpec| ((l.gate.kind == l.up.kind && l.gate.rows == l.up.rows), l.local_attn);
    let uniform = spec.layers.iter().all(|l| shape(l) == shape(&spec.layers[0]));
    let body = |block: &dyn Fn(&LayerSpec) -> String| -> String {
        if uniform {
            format!("    for (i, l) in LAYERS.iter().enumerate() {{\n{}    }}\n", block(&spec.layers[0]))
        } else {
            let mut b = String::new();
            for (i, l) in spec.layers.iter().enumerate() {
                b.push_str(&format!("    {{\n        let (i, l) = ({i}, &LAYERS[{i}]);\n{}    }}\n", block(l)));
            }
            b
        }
    };
    let decode_body = body(&decode_block);
    let batch_body = body(&batch_block);
    let scale = if f.embd_scale { format!("    scale_inplace(&mut s.x, {:?});\n", (d.dim as f32).sqrt()) } else { String::new() };
    let scale_rows = if f.embd_scale { format!("    scale_inplace(&mut s.bx[..m * {dim}], {:?});\n", (d.dim as f32).sqrt()) } else { String::new() };
    let local = if d.swa_window > 0 { "    rope_cos_sin(&mut s.rope_cs_local, &s.inv_freq_local, pos as f32 / DIMS.rope_scale_local);\n" } else { "" };
    let local_rows = if d.swa_window > 0 { "    rope_cos_sin_rows(&mut s.brope_cs_local, &s.inv_freq_local, pos0, m, DIMS.rope_scale_local);\n" } else { "" };
    writeln!(o, "\n/// [`forward`] on the GPU: one command buffer for the whole step.").unwrap();
    writeln!(o, "#[cfg(target_os = \"macos\")]").unwrap();
    writeln!(o, "pub fn forward_gpu(s: &mut State, w: &Weights, g: &mut Gpu, token: u32, pos: usize, want_logits: bool) {{").unwrap();
    writeln!(o, "    assert!((token as usize) < DIMS.vocab, \"token id out of range\");").unwrap();
    writeln!(o, "    assert!(pos < s.ctx, \"position exceeds the context\");").unwrap();
    writeln!(o, "    dequant_row_{embd}(w.row(&TOKEN_EMBD, token as usize), &mut s.x);").unwrap();
    o.push_str(&scale);
    writeln!(o, "    rope_cos_sin(&mut s.rope_cs, &s.inv_freq, pos as f32 / DIMS.rope_scale);").unwrap();
    o.push_str(local);
    writeln!(o, "    let kv = DIMS.kv_dim();").unwrap();
    writeln!(o, "    g.begin();").unwrap();
    o.push_str(&decode_body);
    writeln!(o, "    if want_logits {{").unwrap();
    writeln!(o, "        g.rmsnorm(BufId::Xb, BufId::X, 0, &OUTPUT_NORM, {dim}, 1, DIMS.eps);").unwrap();
    writeln!(o, "        g.mv(BufId::Logits, 0, &OUTPUT, BufId::Xb, 0, None, MvOpts::default());").unwrap();
    writeln!(o, "    }}").unwrap();
    writeln!(o, "    g.finish();").unwrap();
    writeln!(o, "}}\n").unwrap();

    writeln!(o, "/// [`forward_batch`] on the GPU.").unwrap();
    writeln!(o, "#[cfg(target_os = \"macos\")]").unwrap();
    writeln!(o, "pub fn forward_batch_gpu(s: &mut State, w: &Weights, g: &mut Gpu, tokens: &[u32], pos0: usize, want_logits: bool) {{").unwrap();
    writeln!(o, "    let m = tokens.len();").unwrap();
    writeln!(o, "    assert!(m >= 1 && m <= s.max_batch, \"batch size out of range\");").unwrap();
    writeln!(o, "    assert!(pos0 + m <= s.ctx, \"positions exceed the context\");").unwrap();
    writeln!(o, "    for (i, &t) in tokens.iter().enumerate() {{").unwrap();
    writeln!(o, "        assert!((t as usize) < DIMS.vocab, \"token id out of range\");").unwrap();
    writeln!(o, "        dequant_row_{embd}(w.row(&TOKEN_EMBD, t as usize), &mut s.bx[i * {dim}..(i + 1) * {dim}]);").unwrap();
    writeln!(o, "    }}").unwrap();
    o.push_str(&scale_rows);
    writeln!(o, "    rope_cos_sin_rows(&mut s.brope_cs, &s.inv_freq, pos0, m, DIMS.rope_scale);").unwrap();
    o.push_str(local_rows);
    writeln!(o, "    let kv = DIMS.kv_dim();").unwrap();
    writeln!(o, "    g.begin();").unwrap();
    o.push_str(&batch_body);
    writeln!(o, "    if want_logits {{").unwrap();
    writeln!(o, "        g.rmsnorm(BufId::Xb, BufId::Bx, (m - 1) * {dim}, &OUTPUT_NORM, {dim}, 1, DIMS.eps);").unwrap();
    writeln!(o, "        g.matmul(BufId::Logits, 0, &OUTPUT, BufId::Xb, 0, 1, false);").unwrap();
    writeln!(o, "    }}").unwrap();
    writeln!(o, "    g.finish();").unwrap();
    writeln!(o, "}}").unwrap();
}

/// Generate `payload.rs`: weight + tokenizer embedding.
pub fn emit_payload_rs(spec: &ModelSpec, opts: &EmitOptions) -> String {
    let mut o = String::new();
    writeln!(o, "//! Generated by aot-llm: how the weights and tokenizer reach the runtime.\n").unwrap();
    writeln!(o, "use std::path::Path;\n").unwrap();
    writeln!(o, "/// Tokenizer tables, embedded as read-only data.").unwrap();
    writeln!(o, "static TOKENIZER: &[u8] = include_bytes!(\"../tokenizer.bin\");\n").unwrap();
    writeln!(o, "pub fn tokenizer() -> &'static [u8] {{\n    TOKENIZER\n}}\n").unwrap();
    emit_prefix_kv(&mut o, opts);
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
    // A private section: Mach-O arm64 relocation addends are 24-bit, so the
    // compiler's own constants must not share a section with the blob.
    // GPU builds place it in __DATA, page aligned: Metal wires the pages of
    // the weights, and wired copies of __TEXT pages fail code-signature
    // validation when the CPU touches them afterwards (see rt/weights.rs).
    let (segment, align) = if opts.gpu.is_some() { ("__DATA", 16384) } else { ("__TEXT", 64) };
    writeln!(o, "#[cfg(target_os = \"macos\")]").unwrap();
    writeln!(o, "std::arch::global_asm!(").unwrap();
    writeln!(o, "    \".section {segment},__aot_weights\",").unwrap();
    writeln!(o, "    \".balign {align}\",").unwrap();
    writeln!(o, "    \".globl _aot_weights_start\",").unwrap();
    writeln!(o, "    \"_aot_weights_start:\",").unwrap();
    writeln!(o, "    \".incbin \\\"{path}\\\", {skip}, {len}\",").unwrap();
    writeln!(o, "    \".balign {align}\",").unwrap();
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
    if opts.gpu.is_some() {
        writeln!(o, "    // Re-map the section from the executable file (MAP_SHARED) so the GPU").unwrap();
        writeln!(o, "    // can wire its pages without copying them; the loader's own private").unwrap();
        writeln!(o, "    // mapping below is the fallback (the GPU then copies on first use).").unwrap();
        writeln!(o, "    #[cfg(target_os = \"macos\")]").unwrap();
        writeln!(o, "    if let Ok(m) = crate::rt::weights::map_own_section(\"{segment}\", \"__aot_weights\", WEIGHTS_LEN) {{").unwrap();
        writeln!(o, "        return Ok(m);").unwrap();
        writeln!(o, "    }}").unwrap();
    }
    writeln!(o, "    // SAFETY: the symbol marks WEIGHTS_LEN bytes of read-only data placed").unwrap();
    writeln!(o, "    // by the assembler directive above; it lives for the whole process.").unwrap();
    writeln!(o, "    Ok(unsafe {{ std::slice::from_raw_parts(std::ptr::addr_of!(aot_weights_start), WEIGHTS_LEN) }})").unwrap();
    writeln!(o, "}}").unwrap();
    o
}

/// Emit the compile-time KV prefix accessor (`.incbin` of the baked blob or
/// an empty slice).
fn emit_prefix_kv(o: &mut String, opts: &EmitOptions) {
    match &opts.prefix_kv_path {
        None => {
            writeln!(o, "/// No KV prefix was baked into this binary.").unwrap();
            writeln!(o, "pub fn prefix_kv() -> &'static [u8] {{\n    &[]\n}}\n").unwrap();
        }
        Some(path) => {
            let len = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let p = path.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\"");
            writeln!(o, "/// Compile-time KV cache of the baked prompt prefix (see `rt/prefix.rs`).").unwrap();
            writeln!(o, "pub const PREFIX_KV_LEN: usize = {len};\n").unwrap();
            writeln!(o, "#[cfg(target_os = \"macos\")]").unwrap();
            writeln!(o, "std::arch::global_asm!(").unwrap();
            writeln!(o, "    \".section __TEXT,__aot_prefix\",").unwrap();
            writeln!(o, "    \".balign 64\",").unwrap();
            writeln!(o, "    \".globl _aot_prefix_start\",").unwrap();
            writeln!(o, "    \"_aot_prefix_start:\",").unwrap();
            writeln!(o, "    \".incbin \\\"{p}\\\"\",").unwrap();
            writeln!(o, "    \".balign 64\",").unwrap();
            writeln!(o, ");\n").unwrap();
            writeln!(o, "#[cfg(not(target_os = \"macos\"))]").unwrap();
            writeln!(o, "std::arch::global_asm!(").unwrap();
            writeln!(o, "    \".section .rodata.aot_prefix,\\\"a\\\",@progbits\",").unwrap();
            writeln!(o, "    \".balign 64\",").unwrap();
            writeln!(o, "    \".globl aot_prefix_start\",").unwrap();
            writeln!(o, "    \"aot_prefix_start:\",").unwrap();
            writeln!(o, "    \".incbin \\\"{p}\\\"\",").unwrap();
            writeln!(o, "    \".balign 64\",").unwrap();
            writeln!(o, "    \".text\",").unwrap();
            writeln!(o, ");\n").unwrap();
            writeln!(o, "extern \"C\" {{\n    static aot_prefix_start: u8;\n}}\n").unwrap();
            writeln!(o, "pub fn prefix_kv() -> &'static [u8] {{").unwrap();
            writeln!(o, "    // SAFETY: PREFIX_KV_LEN read-only bytes placed by the directive above.").unwrap();
            writeln!(o, "    unsafe {{ std::slice::from_raw_parts(std::ptr::addr_of!(aot_prefix_start), PREFIX_KV_LEN) }}").unwrap();
            writeln!(o, "}}\n").unwrap();
        }
    }
}

/// The `Runtime` literal shared by `main.rs` and `lib.rs`.
fn runtime_literal(opts: &EmitOptions, indent: &str) -> String {
    let (gpu_fwd, gpu_batch) = if opts.gpu.is_some() { ("Some(model::forward_gpu)", "Some(model::forward_batch_gpu)") } else { ("None", "None") };
    format!(
        "rt::app::Runtime {{
{indent}    dims: &model::DIMS,
{indent}    info: &model::INFO,
{indent}    forward: model::forward,
{indent}    forward_batch: model::forward_batch,
{indent}    forward_gpu: {gpu_fwd},
{indent}    forward_batch_gpu: {gpu_batch},
{indent}    rope_freqs: model::ROPE_FREQS,
{indent}    tokenizer_blob: payload::tokenizer(),
{indent}    prefix_kv: payload::prefix_kv(),
{indent}    weights: payload::weights,
{indent}    sidecar: payload::SIDECAR,
{indent}}}"
    )
}

pub fn emit_main_rs(spec: &ModelSpec, opts: &EmitOptions) -> String {
    format!(
        "//! Generated by aot-llm: standalone inference binary for `{}`.
#![allow(dead_code, unused_imports, clippy::all)]

mod model;
mod payload;
mod rt;

fn main() {{
    let runtime = {};
    std::process::exit(rt::app::main(runtime));
}}
",
        spec.name,
        runtime_literal(opts, "    ")
    )
}

/// `lib.rs` of a model library: the `Runtime` plus the C API of `aot_llm.h`
/// (see `rt/capi.rs`).
pub fn emit_lib_rs(spec: &ModelSpec, opts: &EmitOptions) -> String {
    format!(
        "//! Generated by aot-llm: model library for `{}` (C API in `aot_llm.h`).
#![allow(dead_code, unused_imports, clippy::all)]

mod model;
mod payload;
mod rt;

/// The compiled model as the runtime sees it.
pub fn runtime() -> rt::app::Runtime {{
    {}
}}

crate::export_c_api!(crate::rt::capi, crate::runtime(), {:?});
",
        spec.name,
        runtime_literal(opts, "    "),
        opts.symbol_prefix
    )
}

/// `aot_llm.h` for a library built with a symbol prefix: every `aot_`
/// identifier outside the common block is prefixed, and so is the include
/// guard. The common block (`enum aot_device`) is shared by all model
/// libraries and left alone.
pub fn c_header(prefix: &str) -> String {
    let h = crate::rt_sources::C_HEADER;
    if prefix.is_empty() {
        return h.to_string();
    }
    let (a, rest) = h.split_once("/* AOT_COMMON_BEGIN").unwrap_or((h, ""));
    let (common, b) = rest.split_once("/* AOT_COMMON_END */").unwrap_or((rest, ""));
    let rename = |text: &str| -> String {
        let bytes = text.as_bytes();
        let mut out = String::with_capacity(text.len() + 256);
        let mut i = 0;
        while i < bytes.len() {
            let boundary = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if boundary && text[i..].starts_with("aot_") {
                out.push_str(prefix);
            }
            let c = text[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
        out
    };
    let guard = format!("{}AOT_LLM_H", prefix.to_ascii_uppercase());
    format!("{}/* AOT_COMMON_BEGIN{}/* AOT_COMMON_END */{}", rename(a), common, rename(b)).replace("AOT_LLM_H", &guard)
}

pub fn emit_cargo_toml(opts: &EmitOptions) -> String {
    let target = match opts.lib {
        None => format!("[[bin]]\nname = \"{}\"\npath = \"src/main.rs\"\n", opts.crate_name),
        Some(kind) => format!("[lib]\nname = \"{}\"\npath = \"src/lib.rs\"\ncrate-type = [\"{}\"]\n", opts.crate_name, kind.crate_type()),
    };
    format!(
        "[package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         edition = \"2021\"\n\
         publish = false\n\n\
         {target}\n\
         [dependencies]\n\n\
         # Standalone project even when generated inside another cargo workspace.\n\
         [workspace]\n\n\
         [profile.release]\n\
         opt-level = 3\n\
         lto = \"fat\"\n\
         codegen-units = 1\n\
         panic = \"abort\"\n\
         strip = true\n\
         debug = false\n\
         incremental = false\n",
        name = opts.crate_name,
        target = target,
    )
}

pub fn emit_readme(spec: &ModelSpec, opts: &EmitOptions) -> String {
    format!(
        "# {name}\n\n\
         Standalone inference binary generated by aot-llm {ver} from `{src}`.\n\n\
         * `src/model.rs`   - static forward pass and tensor table (generated)\n\
         * `src/payload.rs` - weight / tokenizer embedding (generated)\n\
         * `src/rt/`        - runtime kernels (copied from aot-kernels)\n\
         * `tokenizer.bin`  - serialised vocabulary\n\
         {lib}\n\
         Rebuild with `cargo build --release`. {note}\n",
        name = opts.crate_name,
        ver = opts.compiler_version,
        src = spec.source,
        lib = if opts.lib.is_some() { "* `src/lib.rs` + `aot_llm.h` - C API of the model library\n" } else { "" },
        note = if opts.sidecar { "Weights are read from `<binary>.weights` at run time." } else { "The weights are read from the original GGUF file at build time (see `.incbin` in `src/payload.rs`), so keep it in place while rebuilding." }
    )
}

/// Cargo configuration for the generated project: `target-cpu=native` on
/// request, and self-contained static linking with `rust-lld` for musl
/// targets so Linux binaries can be produced from any host.
pub fn cargo_config(opts: &EmitOptions) -> String {
    let mut c = String::new();
    if opts.native {
        c.push_str("[build]\nrustflags = [\"-C\", \"target-cpu=native\"]\n\n");
    }
    if let Some(t) = opts.target.as_deref().filter(|t| t.contains("-musl")) {
        c.push_str(&format!("[target.{t}]\nlinker = \"rust-lld\"\nrustflags = [\"-C\", \"target-feature=+crt-static\", \"-C\", \"link-self-contained=yes\"]\n"));
    }
    c
}

/// Write the whole project into `dir` (created if needed).
pub fn write_project(dir: &Path, spec: &ModelSpec, tokenizer_blob: &[u8], opts: &EmitOptions) -> Result<()> {
    let src = dir.join("src");
    let rt = src.join("rt");
    fs::create_dir_all(&rt).with_context(|| format!("creating {}", rt.display()))?;
    fs::write(dir.join("Cargo.toml"), emit_cargo_toml(opts))?;
    fs::write(dir.join("README.md"), emit_readme(spec, opts))?;
    fs::write(dir.join("tokenizer.bin"), tokenizer_blob)?;
    // A project is either a binary or a library; remove the other entry
    // point so a re-emit with different options leaves no stale file.
    if opts.lib.is_some() {
        fs::write(src.join("lib.rs"), emit_lib_rs(spec, opts))?;
        fs::write(dir.join("aot_llm.h"), c_header(&opts.symbol_prefix))?;
        let _ = fs::remove_file(src.join("main.rs"));
    } else {
        fs::write(src.join("main.rs"), emit_main_rs(spec, opts))?;
        let _ = fs::remove_file(src.join("lib.rs"));
        let _ = fs::remove_file(dir.join("aot_llm.h"));
    }
    fs::write(src.join("model.rs"), emit_model_rs(spec, opts))?;
    fs::write(src.join("payload.rs"), emit_payload_rs(spec, opts))?;
    for (name, contents) in RT_SOURCES {
        fs::write(rt.join(name), contents)?;
    }
    let cargo_dir = dir.join(".cargo");
    let config = cargo_config(opts);
    if config.is_empty() {
        let _ = fs::remove_file(cargo_dir.join("config.toml"));
    } else {
        fs::create_dir_all(&cargo_dir)?;
        fs::write(cargo_dir.join("config.toml"), config)?;
    }
    // A stale lock file from an older toolchain is harmless but confusing.
    let _ = fs::remove_file(dir.join("Cargo.lock"));
    Ok(())
}
