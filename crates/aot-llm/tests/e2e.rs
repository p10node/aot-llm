//! End-to-end tests: build a tiny random GGUF for each supported
//! architecture (all nine weight types covered), compile it with the `aot-llm`
//! CLI, run the produced binary and compare its logits against an f32
//! reference forward pass.
//!
//! * `llama`: the plain block (adjacent-pair RoPE, no biases).
//! * `qwen2`: q / k / v biases and NEOX RoPE.
//! * `qwen3`: per-head q / k RMSNorm, NEOX RoPE, and a head size that makes
//!   `q_dim` larger than `dim` (as in Qwen3-0.6B).
//! * `phi3`: fused `attn_qkv` and `ffn_up` (gate + up) tensors, NEOX RoPE.
//! * `gemma3`: scaled embedding, q/k norms, post-attention / post-FFN norms,
//!   GeGLU, sliding-window layers with their own RoPE base and a global
//!   layer with linear RoPE scaling (6 layers, so one of them is global).

use aot_codegen::tokenizer::build_blob;
use aot_gguf::writer::GgufWriter;
use aot_gguf::{GgmlType, GgufFile, MetaValue};
use aot_kernels::quant::{dequant_row, quantize_q8_0, quantize_q8_k, Kind};
use aot_kernels::quantize::quantize_row;
use aot_kernels::tokenizer::Tokenizer;
use std::cell::RefCell;
use std::path::PathBuf;
use std::process::Command;

const DIM: usize = 256;
const HIDDEN: usize = 512;
const EPS: f32 = 1e-5;
/// RoPE base of the local (sliding-window) layers; llama.cpp hardcodes it for Gemma 3.
const LOCAL_ROPE_BASE: f32 = 10000.0;
/// A quantized activation this close (relative) to a rounding boundary is a
/// tie: the binary's float noise against the reference is ~2e-7.
const TIE: f32 = 1e-5;

/// Architecture variant under test.
#[derive(Clone, Copy)]
struct Cfg {
    arch: &'static str,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// NEOX RoPE layout (pairs half a head apart).
    neox: bool,
    /// q / k / v projection biases.
    bias: bool,
    /// Per-head RMSNorm of q and k before RoPE.
    qk_norm: bool,
    /// q/k/v written as one `attn_qkv` tensor and gate/up as one `ffn_up` (Phi-3).
    fused: bool,
    n_layer: usize,
    /// Gemma: embedding scaled by sqrt(dim), post-norms, GeGLU.
    gemma: bool,
    /// Sliding window of the local layers (every 6th layer is global); 0 = none.
    swa_window: usize,
    /// RoPE base and linear scale of the global layers.
    rope_base: f32,
    rope_scale: f32,
}

impl Cfg {
    fn local_layer(&self, li: usize) -> bool {
        self.swa_window > 0 && !(li + 1).is_multiple_of(6)
    }
}

impl Cfg {
    fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
}

const BASE: Cfg = Cfg { arch: "llama", n_head: 4, n_kv_head: 2, head_dim: 64, neox: false, bias: false, qk_norm: false, fused: false, n_layer: 2, gemma: false, swa_window: 0, rope_base: 10000.0, rope_scale: 1.0 };
const LLAMA: Cfg = BASE;
const QWEN2: Cfg = Cfg { arch: "qwen2", neox: true, bias: true, ..BASE };
const QWEN3: Cfg = Cfg { arch: "qwen3", head_dim: 128, neox: true, qk_norm: true, ..BASE };
const PHI3: Cfg = Cfg { arch: "phi3", neox: true, fused: true, ..BASE };
const GEMMA3: Cfg = Cfg { arch: "gemma3", neox: true, qk_norm: true, n_layer: 6, gemma: true, swa_window: 4, rope_base: 1_000_000.0, rope_scale: 2.0, ..BASE };

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self, a: f32) -> f32 {
        ((self.next() % 1_000_001) as f32 / 500_000.0 - 1.0) * a
    }
    fn vec(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(a)).collect()
    }
}

/// A dense f32 matrix (rows x cols) plus its quantized encoding.
struct Mat {
    rows: usize,
    cols: usize,
    f32: Vec<f32>,
    kind: Kind,
    bytes: Vec<u8>,
}

fn ggml(kind: Kind) -> GgmlType {
    match kind {
        Kind::F32 => GgmlType::F32,
        Kind::F16 => GgmlType::F16,
        Kind::BF16 => GgmlType::BF16,
        Kind::Q4_0 => GgmlType::Q4_0,
        Kind::Q5_0 => GgmlType::Q5_0,
        Kind::Q8_0 => GgmlType::Q8_0,
        Kind::Q4_K => GgmlType::Q4_K,
        Kind::Q5_K => GgmlType::Q5_K,
        Kind::Q6_K => GgmlType::Q6_K,
    }
}

fn mat(rng: &mut Rng, rows: usize, cols: usize, kind: Kind, amp: f32) -> Mat {
    let raw = rng.vec(rows * cols, amp);
    let mut bytes = Vec::new();
    let mut f32v = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let enc = quantize_row(kind, &raw[r * cols..(r + 1) * cols]);
        let mut dec = vec![0f32; cols];
        dequant_row(kind, &enc, &mut dec);
        f32v.extend_from_slice(&dec);
        bytes.extend_from_slice(&enc);
    }
    Mat { rows, cols, f32: f32v, kind, bytes }
}

/// A quantized activation: (Q8_K rather than Q8_0, index in the vector, f32 bits).
/// The runtime quantizes a vector once for all the matmuls that read it, so
/// the key names the value, not the matmul.
type TieKey = (bool, usize, u32);

/// How the reference feeds activations to a matmul.
struct Acts {
    /// Apply the CPU path's activation quantization (the GPU path multiplies
    /// f32 activations directly).
    quant: bool,
    /// Quantized activations seen within `TIE` of a rounding boundary.
    ties: RefCell<Vec<TieKey>>,
    /// Round this one to the other neighbouring level.
    flip: Option<TieKey>,
}

impl Acts {
    fn new(quant: bool, flip: Option<TieKey>) -> Self {
        Acts { quant, ties: RefCell::new(Vec::new()), flip }
    }
}

/// Quantize the activation vector exactly like the runtime does before a
/// matmul with weights of `kind`, then dequantize it back to f32. This makes
/// the reference bit-for-bit comparable, up to float summation order and the
/// ties that order can tip (see `check_quant_logits`).
fn act_quant(kind: Kind, x: &[f32], acts: &Acts) -> Vec<f32> {
    let qd: Vec<(i8, f32)> = match kind {
        Kind::Q4_0 | Kind::Q5_0 | Kind::Q8_0 => {
            let mut b = Vec::new();
            quantize_q8_0(x, &mut b);
            b.iter().flat_map(|blk| blk.qs.iter().map(move |&q| (q, blk.d))).collect()
        }
        Kind::Q4_K | Kind::Q5_K | Kind::Q6_K => {
            let mut b = Vec::new();
            quantize_q8_k(x, &mut b);
            b.iter().flat_map(|blk| blk.qs.iter().map(move |&q| (q, blk.d))).collect()
        }
        _ => return x.to_vec(),
    };
    let k_format = matches!(kind, Kind::Q4_K | Kind::Q5_K | Kind::Q6_K);
    x.iter()
        .zip(qd)
        .enumerate()
        .map(|(i, (&v, (q, d)))| {
            let key = (k_format, i, v.to_bits());
            let t = v / d;
            if d != 0.0 && (t - t.floor() - 0.5).abs() < TIE * t.abs().max(1.0) && !acts.ties.borrow().contains(&key) {
                acts.ties.borrow_mut().push(key);
            }
            let q = q as i32;
            let q = if acts.flip == Some(key) { (if t > q as f32 { q + 1 } else { q - 1 }).clamp(-127, 127) } else { q };
            q as f32 * d
        })
        .collect()
}

fn matvec(m: &Mat, x: &[f32], acts: &Acts) -> Vec<f32> {
    assert_eq!(x.len(), m.cols);
    let xq = if acts.quant { act_quant(m.kind, x, acts) } else { x.to_vec() };
    (0..m.rows).map(|r| (0..m.cols).map(|c| m.f32[r * m.cols + c] * xq[c]).sum()).collect()
}

fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + EPS).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * b).collect()
}

/// RoPE over `n_heads` heads of `head_dim`, adjacent pairs or NEOX pairs.
fn rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, neox: bool, base: f32, scale: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        for i in 0..half {
            let theta = (pos as f32 / scale) * base.powf(-(2.0 * i as f32) / head_dim as f32);
            let (s, c) = theta.sin_cos();
            let (i0, i1) = if neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let a = x[h * head_dim + i0];
            let b = x[h * head_dim + i1];
            x[h * head_dim + i0] = a * c - b * s;
            x[h * head_dim + i1] = a * s + b * c;
        }
    }
}

fn add(x: &mut [f32], b: &[f32]) {
    for (a, b) in x.iter_mut().zip(b) {
        *a += b;
    }
}

fn norm_heads(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let v = rmsnorm(&x[h * head_dim..(h + 1) * head_dim], w);
        x[h * head_dim..(h + 1) * head_dim].copy_from_slice(&v);
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    ffn_norm: Vec<f32>,
    gate: Mat,
    up: Mat,
    down: Mat,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    attn_post_norm: Option<Vec<f32>>,
    ffn_post_norm: Option<Vec<f32>>,
}

struct Model {
    cfg: Cfg,
    embd: Mat,
    output_norm: Vec<f32>,
    output: Mat,
    layers: Vec<Layer>,
}

/// Reference forward over a token sequence; returns logits of the last token.
fn reference_logits(m: &Model, tokens: &[u32], acts: &Acts) -> Vec<f32> {
    let cfg = &m.cfg;
    let hd = cfg.head_dim;
    let kv_dim = cfg.kv_dim();
    let mut k_cache = vec![vec![0f32; tokens.len() * kv_dim]; cfg.n_layer];
    let mut v_cache = vec![vec![0f32; tokens.len() * kv_dim]; cfg.n_layer];
    let mut logits = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let mut x: Vec<f32> = m.embd.f32[t as usize * DIM..(t as usize + 1) * DIM].to_vec();
        if cfg.gemma {
            let s = (DIM as f32).sqrt();
            for v in x.iter_mut() {
                *v *= s;
            }
        }
        for (li, l) in m.layers.iter().enumerate() {
            let local = cfg.local_layer(li);
            let (base, scale) = if local { (LOCAL_ROPE_BASE, 1.0) } else { (cfg.rope_base, cfg.rope_scale) };
            let t_start = if local && pos + 1 > cfg.swa_window { pos + 1 - cfg.swa_window } else { 0 };
            let xb = rmsnorm(&x, &l.attn_norm);
            let mut q = matvec(&l.wq, &xb, acts);
            let mut k = matvec(&l.wk, &xb, acts);
            let mut v = matvec(&l.wv, &xb, acts);
            if let (Some(bq), Some(bk), Some(bv)) = (&l.bq, &l.bk, &l.bv) {
                add(&mut q, bq);
                add(&mut k, bk);
                add(&mut v, bv);
            }
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                norm_heads(&mut q, qn, cfg.n_head, hd);
                norm_heads(&mut k, kn, cfg.n_kv_head, hd);
            }
            rope(&mut q, cfg.n_head, hd, pos, cfg.neox, base, scale);
            rope(&mut k, cfg.n_kv_head, hd, pos, cfg.neox, base, scale);
            k_cache[li][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            v_cache[li][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let mut att = vec![0f32; cfg.q_dim()];
            for h in 0..cfg.n_head {
                let kvh = h / (cfg.n_head / cfg.n_kv_head);
                let qh = &q[h * hd..(h + 1) * hd];
                let mut sc: Vec<f32> = (t_start..=pos)
                    .map(|tp| {
                        let kt = &k_cache[li][tp * kv_dim + kvh * hd..tp * kv_dim + (kvh + 1) * hd];
                        qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for (tp, s) in (t_start..).zip(sc.iter()) {
                    let vt = &v_cache[li][tp * kv_dim + kvh * hd..tp * kv_dim + (kvh + 1) * hd];
                    for i in 0..hd {
                        att[h * hd + i] += s / sum * vt[i];
                    }
                }
            }
            let o = matvec(&l.wo, &att, acts);
            let o = match &l.attn_post_norm {
                Some(w) => rmsnorm(&o, w),
                None => o,
            };
            add(&mut x, &o);
            let xb = rmsnorm(&x, &l.ffn_norm);
            let g = matvec(&l.gate, &xb, acts);
            let u = matvec(&l.up, &xb, acts);
            let hbuf: Vec<f32> =
                if cfg.gemma { g.iter().zip(&u).map(|(g, u)| 0.5 * g * (1.0 + (0.797_884_6f32 * (g + 0.044715 * g * g * g)).tanh()) * u).collect() } else { g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect() };
            let d = matvec(&l.down, &hbuf, acts);
            let d = match &l.ffn_post_norm {
                Some(w) => rmsnorm(&d, w),
                None => d,
            };
            add(&mut x, &d);
        }
        if pos + 1 == tokens.len() {
            let xb = rmsnorm(&x, &m.output_norm);
            logits = matvec(&m.output, &xb, acts);
        }
    }
    logits
}

fn build_gguf(m: &Model, vocab: &[String], scores: &[f32], types: &[i32]) -> Vec<u8> {
    let cfg = &m.cfg;
    let arch = cfg.arch;
    let mut w = GgufWriter::new();
    let s = |v: &str| MetaValue::String(v.to_string());
    w.kv("general.architecture", s(arch))
        .kv("general.name", s(&format!("aot-e2e-tiny-{arch}")))
        .kv("general.file_type", MetaValue::U32(15))
        .kv(&format!("{arch}.embedding_length"), MetaValue::U32(DIM as u32))
        .kv(&format!("{arch}.feed_forward_length"), MetaValue::U32(HIDDEN as u32))
        .kv(&format!("{arch}.block_count"), MetaValue::U32(cfg.n_layer as u32))
        .kv(&format!("{arch}.attention.head_count"), MetaValue::U32(cfg.n_head as u32))
        .kv(&format!("{arch}.attention.head_count_kv"), MetaValue::U32(cfg.n_kv_head as u32))
        .kv(&format!("{arch}.attention.layer_norm_rms_epsilon"), MetaValue::F32(EPS))
        .kv(&format!("{arch}.rope.freq_base"), MetaValue::F32(cfg.rope_base))
        .kv(&format!("{arch}.context_length"), MetaValue::U32(64))
        .kv("tokenizer.ggml.model", s("llama"))
        .kv("tokenizer.ggml.tokens", MetaValue::Array(vocab.iter().map(|t| s(t)).collect()))
        .kv("tokenizer.ggml.scores", MetaValue::Array(scores.iter().map(|&v| MetaValue::F32(v)).collect()))
        .kv("tokenizer.ggml.token_type", MetaValue::Array(types.iter().map(|&v| MetaValue::I32(v)).collect()))
        .kv("tokenizer.ggml.bos_token_id", MetaValue::U32(1))
        .kv("tokenizer.ggml.eos_token_id", MetaValue::U32(2));
    if cfg.head_dim * cfg.n_head != DIM {
        // Qwen3 states the head size explicitly (it is not dim / n_head).
        w.kv(&format!("{arch}.attention.key_length"), MetaValue::U32(cfg.head_dim as u32));
        w.kv(&format!("{arch}.attention.value_length"), MetaValue::U32(cfg.head_dim as u32));
    } else {
        w.kv(&format!("{arch}.rope.dimension_count"), MetaValue::U32(cfg.head_dim as u32));
    }
    if cfg.swa_window > 0 {
        w.kv(&format!("{arch}.attention.sliding_window"), MetaValue::U32(cfg.swa_window as u32));
    }
    if cfg.rope_scale != 1.0 {
        w.kv(&format!("{arch}.rope.scaling.type"), s("linear"));
        w.kv(&format!("{arch}.rope.scaling.factor"), MetaValue::F32(cfg.rope_scale));
    }
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    w.tensor("token_embd.weight", &[DIM as u64, vocab.len() as u64], ggml(m.embd.kind), m.embd.bytes.clone());
    w.tensor("output_norm.weight", &[DIM as u64], GgmlType::F32, f32b(&m.output_norm));
    w.tensor("output.weight", &[DIM as u64, vocab.len() as u64], ggml(m.output.kind), m.output.bytes.clone());
    for (i, l) in m.layers.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}.weight");
        w.tensor(&n("attn_norm"), &[DIM as u64], GgmlType::F32, f32b(&l.attn_norm));
        if cfg.fused {
            // Phi-3: q, k, v rows in one tensor; gate then up rows in one tensor.
            let mut qkv = l.wq.bytes.clone();
            qkv.extend_from_slice(&l.wk.bytes);
            qkv.extend_from_slice(&l.wv.bytes);
            w.tensor(&n("attn_qkv"), &[l.wq.cols as u64, (l.wq.rows + l.wk.rows + l.wv.rows) as u64], ggml(l.wq.kind), qkv);
        } else {
            w.tensor(&n("attn_q"), &[l.wq.cols as u64, l.wq.rows as u64], ggml(l.wq.kind), l.wq.bytes.clone());
            w.tensor(&n("attn_k"), &[l.wk.cols as u64, l.wk.rows as u64], ggml(l.wk.kind), l.wk.bytes.clone());
            w.tensor(&n("attn_v"), &[l.wv.cols as u64, l.wv.rows as u64], ggml(l.wv.kind), l.wv.bytes.clone());
        }
        w.tensor(&n("attn_output"), &[l.wo.cols as u64, l.wo.rows as u64], ggml(l.wo.kind), l.wo.bytes.clone());
        w.tensor(&n("ffn_norm"), &[DIM as u64], GgmlType::F32, f32b(&l.ffn_norm));
        if cfg.fused {
            let mut gu = l.gate.bytes.clone();
            gu.extend_from_slice(&l.up.bytes);
            w.tensor(&n("ffn_up"), &[l.gate.cols as u64, (l.gate.rows + l.up.rows) as u64], ggml(l.gate.kind), gu);
        } else {
            w.tensor(&n("ffn_gate"), &[l.gate.cols as u64, l.gate.rows as u64], ggml(l.gate.kind), l.gate.bytes.clone());
            w.tensor(&n("ffn_up"), &[l.up.cols as u64, l.up.rows as u64], ggml(l.up.kind), l.up.bytes.clone());
        }
        w.tensor(&n("ffn_down"), &[l.down.cols as u64, l.down.rows as u64], ggml(l.down.kind), l.down.bytes.clone());
        if let (Some(bq), Some(bk), Some(bv)) = (&l.bq, &l.bk, &l.bv) {
            w.tensor(&format!("blk.{i}.attn_q.bias"), &[bq.len() as u64], GgmlType::F32, f32b(bq));
            w.tensor(&format!("blk.{i}.attn_k.bias"), &[bk.len() as u64], GgmlType::F32, f32b(bk));
            w.tensor(&format!("blk.{i}.attn_v.bias"), &[bv.len() as u64], GgmlType::F32, f32b(bv));
        }
        if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
            w.tensor(&n("attn_q_norm"), &[qn.len() as u64], GgmlType::F32, f32b(qn));
            w.tensor(&n("attn_k_norm"), &[kn.len() as u64], GgmlType::F32, f32b(kn));
        }
        if let (Some(pa), Some(pf)) = (&l.attn_post_norm, &l.ffn_post_norm) {
            w.tensor(&n("post_attention_norm"), &[pa.len() as u64], GgmlType::F32, f32b(pa));
            w.tensor(&n("post_ffw_norm"), &[pf.len() as u64], GgmlType::F32, f32b(pf));
        }
    }
    w.finish()
}

fn build_model(cfg: Cfg, rng: &mut Rng, vocab_n: usize) -> Model {
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let norm = |rng: &mut Rng, n: usize| -> Vec<f32> { (0..n).map(|_| 1.0 + rng.uniform(0.5)).collect() };
    let opt = |rng: &mut Rng, on: bool, n: usize, amp: f32| -> Option<Vec<f32>> { on.then(|| rng.vec(n, amp)) };
    let optn = |rng: &mut Rng, on: bool, n: usize| -> Option<Vec<f32>> { on.then(|| norm(rng, n)) };
    // Layer 0 exercises the quantized kernels, layer 1 the float / Q8_0 / Q4_0 ones.
    let kinds = [[Kind::Q4_K, Kind::Q4_0, Kind::Q8_0, Kind::Q5_K, Kind::Q4_K, Kind::Q4_K, Kind::Q6_K], [Kind::F16, Kind::BF16, Kind::F32, Kind::Q4_0, Kind::Q8_0, Kind::Q8_0, Kind::Q5_0]];
    let layers = (0..cfg.n_layer)
        .map(|li| &kinds[li % 2])
        .map(|k| Layer {
            attn_norm: norm(rng, DIM),
            wq: mat(rng, q_dim, DIM, k[0], 0.2),
            // Fused tensors share one format across their parts.
            wk: mat(rng, kv_dim, DIM, if cfg.fused { k[0] } else { k[1] }, 0.2),
            wv: mat(rng, kv_dim, DIM, if cfg.fused { k[0] } else { k[2] }, 0.2),
            wo: mat(rng, DIM, q_dim, k[3], 0.2),
            ffn_norm: norm(rng, DIM),
            gate: mat(rng, HIDDEN, DIM, k[4], 0.2),
            up: mat(rng, HIDDEN, DIM, if cfg.fused { k[4] } else { k[5] }, 0.2),
            down: mat(rng, DIM, HIDDEN, k[6], 0.14),
            bq: opt(rng, cfg.bias, q_dim, 0.3),
            bk: opt(rng, cfg.bias, kv_dim, 0.3),
            bv: opt(rng, cfg.bias, kv_dim, 0.3),
            q_norm: optn(rng, cfg.qk_norm, cfg.head_dim),
            k_norm: optn(rng, cfg.qk_norm, cfg.head_dim),
            attn_post_norm: optn(rng, cfg.gemma, DIM),
            ffn_post_norm: optn(rng, cfg.gemma, DIM),
        })
        .collect();
    Model { cfg, embd: mat(rng, vocab_n, DIM, Kind::Q4_K, 1.0), output_norm: norm(rng, DIM), output: mat(rng, vocab_n, DIM, Kind::Q6_K, 0.2), layers }
}

fn vocab() -> (Vec<String>, Vec<f32>, Vec<i32>) {
    let mut vocab: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    let mut types = vec![2, 3, 3];
    let mut scores = vec![0f32, 0.0, 0.0];
    for b in 0..=255u8 {
        vocab.push(format!("<0x{b:02X}>"));
        types.push(6);
        scores.push(0.0);
    }
    for c in "abcdefghijklmnopqrstuvwxyz\u{2581}".chars() {
        vocab.push(c.to_string());
        types.push(1);
        scores.push(-100.0);
    }
    for word in ["\u{2581}hello", "\u{2581}world", "\u{2581}the", "\u{2581}capital", "\u{2581}of", "\u{2581}is"] {
        let chars: Vec<char> = word.chars().collect();
        for n in 2..=chars.len() {
            let piece: String = chars[..n].iter().collect();
            if !vocab.contains(&piece) {
                vocab.push(piece);
                types.push(1);
                scores.push(-(20.0 - n as f32));
            }
        }
    }
    (vocab, scores, types)
}

fn check_logits(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    let max_abs = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max_err = got.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    eprintln!("{label}: max |logit| {max_abs:.3}, max abs error {max_err:.4}, argmax {} vs {}", argmax(got), argmax(expected));
    // The reference applies the same activation quantization, so only float
    // summation-order noise remains.
    assert!(max_err <= tolerance(expected), "{label}: logits differ: max error {max_err} (range {max_abs})");
    assert_eq!(argmax(got), argmax(expected), "{label}: argmax differs");
}

fn tolerance(expected: &[f32]) -> f32 {
    2e-3 * expected.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-3
}

/// Logits of the quantized-activation reference, and how they change when
/// each of its near-tie activations rounds the other way.
struct QuantRef {
    logits: Vec<f32>,
    flips: Vec<Vec<f32>>,
}

fn quant_reference(m: &Model, tokens: &[u32]) -> QuantRef {
    let acts = Acts::new(true, None);
    let logits = reference_logits(m, tokens, &acts);
    let flips = acts.ties.into_inner().into_iter().map(|t| reference_logits(m, tokens, &Acts::new(true, Some(t))).iter().zip(&logits).map(|(a, b)| a - b).collect()).collect();
    QuantRef { logits, flips }
}

/// Compare CPU logits against the quantized-activation reference.
///
/// Quantizing an activation is discontinuous: one within float noise of a
/// rounding boundary can land on neighbouring levels in the binary and in the
/// reference, which sum in different orders, and in these tiny random models
/// a single such flip moves the logits by a few percent of their range
/// (x86-64 Qwen2: 0.089 of 6.4, a Q8_K value at 4.4999995 vs 4.5000005). A
/// deviation must therefore be explained by rounding at most two near-tie
/// activations the other way (the fewest that do); the remainder is held to
/// the usual tolerance.
fn check_quant_logits(label: &str, got: &[f32], r: &QuantRef) {
    let expected = |flipped: &[usize]| -> Vec<f32> { r.logits.iter().enumerate().map(|(v, l)| l + flipped.iter().map(|&i| r.flips[i][v]).sum::<f32>()).collect() };
    let fits = |e: &[f32]| {
        let tol = tolerance(e);
        got.iter().zip(e).all(|(a, b)| (a - b).abs() <= tol)
    };
    let n = r.flips.len();
    let flipped = std::iter::once(vec![]).chain((0..n).map(|i| vec![i])).chain((0..n).flat_map(|i| (i + 1..n).map(move |j| vec![i, j]))).find(|c| fits(&expected(c))).unwrap_or_default();
    check_logits(&format!("{label} ({} of {n} near-tie activations flipped)", flipped.len()), got, &expected(&flipped));
}

fn parse_logits(out: &std::process::Output) -> Vec<f32> {
    assert!(out.status.success(), "binary failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).split_whitespace().map(|v| v.parse().unwrap()).collect()
}

fn compile_run_and_match_reference(cfg: Cfg) {
    let mut rng = Rng(0x5EED);
    let (vocab, scores, types) = vocab();
    let model = build_model(cfg, &mut rng, vocab.len());
    let gguf = build_gguf(&model, &vocab, &scores, &types);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("e2e-{}", cfg.arch));
    std::fs::create_dir_all(&tmp).unwrap();
    let gguf_path = tmp.join("tiny.gguf");
    std::fs::write(&gguf_path, &gguf).unwrap();
    let out = tmp.join("tiny_bin");

    // Tokens the binary will see (same blob builder as the compiler).
    let parsed = GgufFile::parse(&gguf, gguf.len() as u64).unwrap();
    let (blob, _) = build_blob(&parsed).unwrap();
    let tok = Tokenizer::new(Box::leak(blob.into_boxed_slice())).unwrap();
    // The Gemma prompt repeats so that it is longer than the sliding window.
    let repeat: usize = std::env::var("E2E_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(if cfg.swa_window > 0 { 2 } else { 1 });
    let prompt = vec!["the capital of hello world is"; repeat].join(" ");
    let ids = tok.encode(&prompt, true, true);
    assert_eq!(ids.len(), 1 + 6 * repeat, "{ids:?}");
    let expected = quant_reference(&model, &ids);

    let status = Command::new(env!("CARGO_BIN_EXE_aot-llm")).args(["compile", "--model"]).arg(&gguf_path).arg("--output").arg(&out).arg("--quiet").status().expect("running aot-llm");
    assert!(status.success(), "aot-llm compile failed");

    // Both prompt paths (token by token and batched) must match the reference.
    for batch in ["1", "128"] {
        let run = Command::new(&out).args(["--prompt", &prompt, "--dump-logits", "--quiet", "--batch", batch]).output().expect("running compiled binary");
        check_quant_logits(&format!("{} cpu batch {batch}", cfg.arch), &parse_logits(&run), &expected);
    }

    // Generation runs and stops cleanly.
    let gen = Command::new(&out).args(["--prompt", &prompt, "-n", "8", "--tokens", "--quiet", "--threads", "2"]).output().unwrap();
    assert!(gen.status.success());
    let n = String::from_utf8_lossy(&gen.stdout).split_whitespace().count();
    assert!((1..=8).contains(&n), "generated {n} tokens");

    // Same model with the Metal backend: the GPU uses f32 activations, so it
    // is compared against the unquantized reference.
    #[cfg(target_os = "macos")]
    {
        let out_gpu = tmp.join("tiny_gpu");
        let status = Command::new(env!("CARGO_BIN_EXE_aot-llm")).args(["compile", "--model"]).arg(&gguf_path).arg("--output").arg(&out_gpu).args(["--quiet", "--gpu", "metal"]).status().expect("running aot-llm");
        assert!(status.success(), "aot-llm compile --gpu metal failed");
        let expected = reference_logits(&model, &ids, &Acts::new(false, None));
        for batch in ["1", "128"] {
            let run = Command::new(&out_gpu).args(["--prompt", &prompt, "--dump-logits", "--quiet", "--batch", batch, "--device", "gpu"]).output().expect("running compiled binary");
            let stderr = String::from_utf8_lossy(&run.stderr);
            if !run.status.success() && (stderr.contains("no Metal device") || stderr.contains("cannot load")) {
                eprintln!("skipping the GPU check: {}", stderr.trim());
                return;
            }
            check_logits(&format!("{} gpu batch {batch}", cfg.arch), &parse_logits(&run), &expected);
        }
        let gen = Command::new(&out_gpu).args(["--prompt", &prompt, "-n", "8", "--tokens", "--quiet", "--device", "gpu"]).output().unwrap();
        assert!(gen.status.success(), "{}", String::from_utf8_lossy(&gen.stderr));
        let n = String::from_utf8_lossy(&gen.stdout).split_whitespace().count();
        assert!((1..=8).contains(&n), "gpu generated {n} tokens");
    }
}

#[test]
fn llama_compile_run_and_match_reference() {
    compile_run_and_match_reference(LLAMA);
}

/// C program against the model library: tokenize, evaluate, read the logits
/// and run the convenience generator.
const C_CLIENT: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aot_llm.h"

static int count_cb(const char *piece, size_t len, uint32_t token, void *user) {
    (void)piece; (void)len; (void)token;
    (*(int *)user)++;
    return 0;
}

static int collect_cb(const char *piece, size_t len, uint32_t token, void *user) {
    (void)token;
    strncat((char *)user, piece, len);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    aot_model_params mp = aot_model_default_params();
    mp.n_threads = 2;
    mp.device = AOT_DEVICE_CPU;
    aot_model *m = aot_open(&mp);
    if (!m) { fprintf(stderr, "open: %s\n", aot_last_error()); return 1; }
    aot_model_info info;
    aot_model_info_get(m, &info);
    uint32_t ids[256];
    int32_t n = aot_tokenize(m, argv[1], 0, 1, 1, ids, 256);
    if (n <= 0 || n > 256) { fprintf(stderr, "tokenize: %s\n", aot_last_error()); return 1; }
    printf("TOKENS");
    for (int32_t i = 0; i < n; i++) printf(" %u", ids[i]);
    printf("\n");
    aot_ctx *c = aot_ctx_new(m, 64, 128);
    if (!c) { fprintf(stderr, "ctx: %s\n", aot_last_error()); return 1; }
    if (aot_eval(c, ids, (size_t)n, 1) != 0) { fprintf(stderr, "eval: %s\n", aot_last_error()); return 1; }
    if (aot_ctx_n_past(c) != (uint32_t)n) { fprintf(stderr, "n_past %u\n", aot_ctx_n_past(c)); return 1; }
    const float *logits = aot_logits(c);
    printf("LOGITS");
    for (uint32_t i = 0; i < info.n_vocab; i++) printf(" %.6f", logits[i]);
    printf("\n");
    /* Token by token must give the same logits. */
    aot_ctx_reset(c);
    for (int32_t i = 0; i < n; i++)
        if (aot_eval(c, &ids[i], 1, i + 1 == n) != 0) { fprintf(stderr, "eval1: %s\n", aot_last_error()); return 1; }
    printf("LOGITS1");
    for (uint32_t i = 0; i < info.n_vocab; i++) printf(" %.6f", aot_logits(c)[i]);
    printf("\n");
    /* Save the sequence, load it into a fresh context: same logits. */
    if (aot_ctx_save(c, argv[2]) != 0) { fprintf(stderr, "save: %s\n", aot_last_error()); return 1; }
    aot_ctx *c2 = aot_ctx_new(m, 64, 128);
    if (aot_ctx_load(c2, argv[2]) != 0) { fprintf(stderr, "load: %s\n", aot_last_error()); return 1; }
    if (aot_ctx_n_past(c2) != (uint32_t)n) { fprintf(stderr, "loaded n_past %u\n", aot_ctx_n_past(c2)); return 1; }
    printf("LOGITS2");
    for (uint32_t i = 0; i < info.n_vocab; i++) printf(" %.6f", aot_logits(c2)[i]);
    printf("\n");
    aot_ctx_free(c2);
    aot_ctx_reset(c);
    aot_generate_params gp = aot_generate_default_params();
    gp.max_tokens = 5;
    gp.add_bos = 1;
    int seen = 0;
    int32_t gen = aot_generate(c, argv[1], 0, &gp, count_cb, &seen);
    if (gen < 0) { fprintf(stderr, "generate: %s\n", aot_last_error()); return 1; }
    printf("GEN %d %d %u\n", gen, seen, aot_ctx_n_past(c));
    /* Constrained answer: exactly one of the choices. */
    aot_ctx_reset(c);
    const char *choices[2] = { "hello", "world" };
    gp.choices = choices; gp.n_choices = 2; gp.max_tokens = 8;
    char text[256]; text[0] = 0;
    int32_t gen2 = aot_generate(c, argv[1], 0, &gp, collect_cb, text);
    if (gen2 < 0) { fprintf(stderr, "generate choices: %s\n", aot_last_error()); return 1; }
    printf("CHOICE %s\n", text);
    char buf[64];
    int32_t len = aot_token_piece(m, ids[n - 1], buf, sizeof buf);
    printf("PIECE %.*s\n", len > 0 ? (int)len : 0, buf);
    aot_ctx_free(c);
    aot_close(m);
    return 0;
}
"#;

#[test]
fn lib_c_api_matches_reference() {
    let Ok(cc) = which_cc() else {
        eprintln!("skipping: no C compiler found");
        return;
    };
    let cfg = LLAMA;
    let mut rng = Rng(0x5EED);
    let (vocab, scores, types) = vocab();
    let model = build_model(cfg, &mut rng, vocab.len());
    let gguf = build_gguf(&model, &vocab, &scores, &types);
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("e2e-lib");
    std::fs::create_dir_all(&tmp).unwrap();
    let gguf_path = tmp.join("tiny.gguf");
    std::fs::write(&gguf_path, &gguf).unwrap();
    let out = tmp.join("libdir");

    let status = Command::new(env!("CARGO_BIN_EXE_aot-llm")).args(["compile", "--model"]).arg(&gguf_path).arg("--output").arg(&out).args(["--quiet", "--lib", "--name", "tinymodel"]).status().expect("running aot-llm");
    assert!(status.success(), "aot-llm compile --lib failed");
    let lib = out.join(if cfg!(target_os = "windows") { "tinymodel.lib" } else { "libtinymodel.a" });
    assert!(lib.exists(), "{} missing", lib.display());
    assert!(out.join("aot_llm.h").exists());

    let c_src = tmp.join("client.c");
    std::fs::write(&c_src, C_CLIENT).unwrap();
    let exe = tmp.join("client");
    let mut cmd = Command::new(&cc);
    cmd.arg("-O1").arg(&c_src).arg("-I").arg(&out).arg(&lib).arg("-o").arg(&exe);
    if cfg!(target_os = "linux") {
        cmd.args(["-lpthread", "-ldl", "-lm"]);
    }
    let cc_out = cmd.output().expect("running the C compiler");
    assert!(cc_out.status.success(), "cc failed: {}", String::from_utf8_lossy(&cc_out.stderr));

    let parsed = GgufFile::parse(&gguf, gguf.len() as u64).unwrap();
    let (blob, _) = build_blob(&parsed).unwrap();
    let tok = Tokenizer::new(Box::leak(blob.into_boxed_slice())).unwrap();
    let prompt = "the capital of hello world is";
    let ids = tok.encode(prompt, true, true);
    let expected = quant_reference(&model, &ids);

    let kv_path = tmp.join("client.kv");
    let run = Command::new(&exe).arg(prompt).arg(&kv_path).output().expect("running the C client");
    assert!(run.status.success(), "client failed: {}", String::from_utf8_lossy(&run.stderr));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let mut lines = stdout.lines();
    let toks: Vec<u32> = lines.next().unwrap().strip_prefix("TOKENS ").unwrap().split(' ').map(|v| v.parse().unwrap()).collect();
    assert_eq!(toks, ids, "C tokenizer differs");
    let got: Vec<f32> = lines.next().unwrap().strip_prefix("LOGITS ").unwrap().split(' ').map(|v| v.parse().unwrap()).collect();
    check_quant_logits("lib batched", &got, &expected);
    let got1: Vec<f32> = lines.next().unwrap().strip_prefix("LOGITS1 ").unwrap().split(' ').map(|v| v.parse().unwrap()).collect();
    check_quant_logits("lib token by token", &got1, &expected);
    let got2: Vec<f32> = lines.next().unwrap().strip_prefix("LOGITS2 ").unwrap().split(' ').map(|v| v.parse().unwrap()).collect();
    assert_eq!(got2, got, "logits after aot_ctx_save / aot_ctx_load differ");
    let gen = lines.next().unwrap();
    let parts: Vec<usize> = gen.strip_prefix("GEN ").unwrap().split(' ').map(|v| v.parse().unwrap()).collect();
    assert!((1..=5).contains(&parts[0]), "{gen}");
    // Byte tokens of the tiny vocabulary can form incomplete UTF-8 sequences,
    // which the generator buffers, so the callback may fire fewer times.
    assert!(parts[1] >= 1 && parts[1] <= parts[0], "callback count: {gen}");
    // After generation the context holds the prompt plus the generated tokens
    // that were fed back (all but the last one).
    assert_eq!(parts[2], ids.len() + parts[0] - 1, "{gen}");
    let choice = lines.next().unwrap();
    assert!(choice == "CHOICE hello" || choice == "CHOICE world", "{choice}");
    let piece = lines.next().unwrap();
    assert_eq!(piece, format!("PIECE {}", tok.decode(&ids[ids.len() - 1..], false)));
}

/// The CLI's output constraints and the saved-run continuation.
#[test]
fn cli_stop_choices_and_kv_continuation() {
    let cfg = LLAMA;
    let mut rng = Rng(0x5EED);
    let (vocab, scores, types) = vocab();
    let model = build_model(cfg, &mut rng, vocab.len());
    let gguf = build_gguf(&model, &vocab, &scores, &types);
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("e2e-cli");
    std::fs::create_dir_all(&tmp).unwrap();
    let gguf_path = tmp.join("tiny.gguf");
    std::fs::write(&gguf_path, &gguf).unwrap();
    let out = tmp.join("tiny_bin");
    let status = Command::new(env!("CARGO_BIN_EXE_aot-llm")).args(["compile", "--model"]).arg(&gguf_path).arg("--output").arg(&out).arg("--quiet").status().unwrap();
    assert!(status.success());
    let run = |args: &[&str]| -> String {
        let o = Command::new(&out).args(args).output().unwrap();
        assert!(o.status.success(), "{:?}: {}", args, String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).trim_end_matches('\n').to_string()
    };

    // --choices: the answer is one of the given strings, nothing else.
    for _ in 0..3 {
        let a = run(&["--prompt", "the capital of", "--choices", "hello,world", "-n", "8", "-q"]);
        assert!(a == "hello" || a == "world", "{a:?}");
    }
    let a = run(&["--prompt", "the capital of", "--choice", "hello world", "--choice", "is", "-n", "8", "-q", "--temp", "0.9"]);
    assert!(a == "hello world" || a == "is", "{a:?}");

    // --stop: the stop string never appears in the output.
    let a = run(&["--prompt", "the capital of", "--stop", "l", "-n", "24", "-q", "--temp", "1.0", "--top-k", "0", "--seed", "3"]);
    assert!(!a.contains('l'), "{a:?}");

    // --save-kv / --load-kv: continuing a saved prompt gives the logits of
    // the whole prompt evaluated at once.
    let kv = tmp.join("run.kv");
    let kv_s = kv.to_str().unwrap();
    run(&["--prompt", "the capital", "-n", "0", "-q", "--save-kv", kv_s]);
    assert!(kv.exists());
    let continued = run(&["--load-kv", kv_s, "--prompt", "of hello world is", "--dump-logits", "-q"]);
    let whole = run(&["--prompt", "the capital of hello world is", "--dump-logits", "-q"]);
    assert_eq!(continued, whole, "--load-kv continuation differs from the whole prompt");
}

fn which_cc() -> Result<PathBuf, ()> {
    for c in [std::env::var("CC").unwrap_or_default(), "cc".into(), "clang".into(), "gcc".into()] {
        if c.is_empty() {
            continue;
        }
        if Command::new(&c).arg("--version").output().is_ok_and(|o| o.status.success()) {
            return Ok(PathBuf::from(c));
        }
    }
    Err(())
}

#[test]
fn qwen2_compile_run_and_match_reference() {
    compile_run_and_match_reference(QWEN2);
}

#[test]
fn qwen3_compile_run_and_match_reference() {
    compile_run_and_match_reference(QWEN3);
}

#[test]
fn phi3_compile_run_and_match_reference() {
    compile_run_and_match_reference(PHI3);
}

#[test]
fn gemma3_compile_run_and_match_reference() {
    compile_run_and_match_reference(GEMMA3);
}
