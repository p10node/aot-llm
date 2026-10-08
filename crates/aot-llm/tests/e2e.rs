//! End-to-end test: build a tiny random Llama GGUF covering every supported
//! quantization type, compile it with the `aot-llm` CLI, run the produced
//! binary and compare its logits against an f32 reference forward pass.

use aot_codegen::tokenizer::build_blob;
use aot_gguf::writer::GgufWriter;
use aot_gguf::{GgmlType, GgufFile, MetaValue};
use aot_kernels::quant::{dequant_row, quantize_q8_0, quantize_q8_k, Kind};
use aot_kernels::quantize::quantize_row;
use aot_kernels::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::process::Command;

const DIM: usize = 256;
const HIDDEN: usize = 512;
const N_LAYER: usize = 2;
const N_HEAD: usize = 4;
const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 64;
const EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 10000.0;

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

/// Quantize the activation vector exactly like the runtime does before a
/// matmul with weights of `kind`, then dequantize it back to f32. This makes
/// the reference bit-for-bit comparable (up to float summation order).
fn act_quant(kind: Kind, x: &[f32]) -> Vec<f32> {
    match kind {
        Kind::Q4_0 | Kind::Q8_0 => {
            let mut b = Vec::new();
            quantize_q8_0(x, &mut b);
            b.iter().flat_map(|blk| blk.qs.iter().map(move |&q| q as f32 * blk.d)).collect()
        }
        Kind::Q4_K | Kind::Q5_K | Kind::Q6_K => {
            let mut b = Vec::new();
            quantize_q8_k(x, &mut b);
            b.iter().flat_map(|blk| blk.qs.iter().map(move |&q| q as f32 * blk.d)).collect()
        }
        _ => x.to_vec(),
    }
}

fn matvec(m: &Mat, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), m.cols);
    let xq = act_quant(m.kind, x);
    (0..m.rows).map(|r| (0..m.cols).map(|c| m.f32[r * m.cols + c] * xq[c]).sum()).collect()
}

fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + EPS).sqrt();
    x.iter().zip(w).map(|(a, b)| a * r * b).collect()
}

fn rope(x: &mut [f32], n_heads: usize, pos: usize) {
    for h in 0..n_heads {
        for i in 0..HEAD_DIM / 2 {
            let theta = pos as f32 * ROPE_BASE.powf(-(2.0 * i as f32) / HEAD_DIM as f32);
            let (s, c) = theta.sin_cos();
            let a = x[h * HEAD_DIM + 2 * i];
            let b = x[h * HEAD_DIM + 2 * i + 1];
            x[h * HEAD_DIM + 2 * i] = a * c - b * s;
            x[h * HEAD_DIM + 2 * i + 1] = a * s + b * c;
        }
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
}

struct Model {
    embd: Mat,
    output_norm: Vec<f32>,
    output: Mat,
    layers: Vec<Layer>,
}

/// Reference forward over a token sequence; returns logits of the last token.
fn reference_logits(m: &Model, tokens: &[u32]) -> Vec<f32> {
    let kv_dim = N_KV_HEAD * HEAD_DIM;
    let mut k_cache = vec![vec![0f32; tokens.len() * kv_dim]; N_LAYER];
    let mut v_cache = vec![vec![0f32; tokens.len() * kv_dim]; N_LAYER];
    let mut logits = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let mut x: Vec<f32> = m.embd.f32[t as usize * DIM..(t as usize + 1) * DIM].to_vec();
        for (li, l) in m.layers.iter().enumerate() {
            let xb = rmsnorm(&x, &l.attn_norm);
            let mut q = matvec(&l.wq, &xb);
            let mut k = matvec(&l.wk, &xb);
            let v = matvec(&l.wv, &xb);
            rope(&mut q, N_HEAD, pos);
            rope(&mut k, N_KV_HEAD, pos);
            k_cache[li][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            v_cache[li][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let mut att = vec![0f32; N_HEAD * HEAD_DIM];
            for h in 0..N_HEAD {
                let kvh = h / (N_HEAD / N_KV_HEAD);
                let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let mut sc: Vec<f32> = (0..=pos)
                    .map(|tp| {
                        let kt = &k_cache[li][tp * kv_dim + kvh * HEAD_DIM..tp * kv_dim + (kvh + 1) * HEAD_DIM];
                        qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() / (HEAD_DIM as f32).sqrt()
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for (tp, s) in sc.iter().enumerate() {
                    let vt = &v_cache[li][tp * kv_dim + kvh * HEAD_DIM..tp * kv_dim + (kvh + 1) * HEAD_DIM];
                    for i in 0..HEAD_DIM {
                        att[h * HEAD_DIM + i] += s / sum * vt[i];
                    }
                }
            }
            let o = matvec(&l.wo, &att);
            for i in 0..DIM {
                x[i] += o[i];
            }
            let xb = rmsnorm(&x, &l.ffn_norm);
            let g = matvec(&l.gate, &xb);
            let u = matvec(&l.up, &xb);
            let hbuf: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let d = matvec(&l.down, &hbuf);
            for i in 0..DIM {
                x[i] += d[i];
            }
        }
        if pos + 1 == tokens.len() {
            let xb = rmsnorm(&x, &m.output_norm);
            logits = matvec(&m.output, &xb);
        }
    }
    logits
}

fn build_gguf(m: &Model, vocab: &[String], scores: &[f32], types: &[i32]) -> Vec<u8> {
    let mut w = GgufWriter::new();
    let s = |v: &str| MetaValue::String(v.to_string());
    w.kv("general.architecture", s("llama"))
        .kv("general.name", s("aot-e2e-tiny"))
        .kv("general.file_type", MetaValue::U32(15))
        .kv("llama.embedding_length", MetaValue::U32(DIM as u32))
        .kv("llama.feed_forward_length", MetaValue::U32(HIDDEN as u32))
        .kv("llama.block_count", MetaValue::U32(N_LAYER as u32))
        .kv("llama.attention.head_count", MetaValue::U32(N_HEAD as u32))
        .kv("llama.attention.head_count_kv", MetaValue::U32(N_KV_HEAD as u32))
        .kv("llama.attention.layer_norm_rms_epsilon", MetaValue::F32(EPS))
        .kv("llama.rope.freq_base", MetaValue::F32(ROPE_BASE))
        .kv("llama.rope.dimension_count", MetaValue::U32(HEAD_DIM as u32))
        .kv("llama.context_length", MetaValue::U32(64))
        .kv("tokenizer.ggml.model", s("llama"))
        .kv("tokenizer.ggml.tokens", MetaValue::Array(vocab.iter().map(|t| s(t)).collect()))
        .kv("tokenizer.ggml.scores", MetaValue::Array(scores.iter().map(|&v| MetaValue::F32(v)).collect()))
        .kv("tokenizer.ggml.token_type", MetaValue::Array(types.iter().map(|&v| MetaValue::I32(v)).collect()))
        .kv("tokenizer.ggml.bos_token_id", MetaValue::U32(1))
        .kv("tokenizer.ggml.eos_token_id", MetaValue::U32(2));
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    w.tensor("token_embd.weight", &[DIM as u64, vocab.len() as u64], ggml(m.embd.kind), m.embd.bytes.clone());
    w.tensor("output_norm.weight", &[DIM as u64], GgmlType::F32, f32b(&m.output_norm));
    w.tensor("output.weight", &[DIM as u64, vocab.len() as u64], ggml(m.output.kind), m.output.bytes.clone());
    for (i, l) in m.layers.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}.weight");
        w.tensor(&n("attn_norm"), &[DIM as u64], GgmlType::F32, f32b(&l.attn_norm));
        w.tensor(&n("attn_q"), &[l.wq.cols as u64, l.wq.rows as u64], ggml(l.wq.kind), l.wq.bytes.clone());
        w.tensor(&n("attn_k"), &[l.wk.cols as u64, l.wk.rows as u64], ggml(l.wk.kind), l.wk.bytes.clone());
        w.tensor(&n("attn_v"), &[l.wv.cols as u64, l.wv.rows as u64], ggml(l.wv.kind), l.wv.bytes.clone());
        w.tensor(&n("attn_output"), &[l.wo.cols as u64, l.wo.rows as u64], ggml(l.wo.kind), l.wo.bytes.clone());
        w.tensor(&n("ffn_norm"), &[DIM as u64], GgmlType::F32, f32b(&l.ffn_norm));
        w.tensor(&n("ffn_gate"), &[l.gate.cols as u64, l.gate.rows as u64], ggml(l.gate.kind), l.gate.bytes.clone());
        w.tensor(&n("ffn_up"), &[l.up.cols as u64, l.up.rows as u64], ggml(l.up.kind), l.up.bytes.clone());
        w.tensor(&n("ffn_down"), &[l.down.cols as u64, l.down.rows as u64], ggml(l.down.kind), l.down.bytes.clone());
    }
    w.finish()
}

#[test]
fn compile_run_and_match_reference() {
    let mut rng = Rng(0x5EED);
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
    let vocab_n = vocab.len();
    let q_dim = N_HEAD * HEAD_DIM;
    let kv_dim = N_KV_HEAD * HEAD_DIM;
    let norm = |rng: &mut Rng| -> Vec<f32> { (0..DIM).map(|_| 1.0 + rng.uniform(0.5)).collect() };
    // Layer 0 exercises the quantized kernels, layer 1 the float / Q8_0 / Q4_0 ones.
    let layers = vec![
        Layer {
            attn_norm: norm(&mut rng),
            wq: mat(&mut rng, q_dim, DIM, Kind::Q4_K, 0.2),
            wk: mat(&mut rng, kv_dim, DIM, Kind::Q4_0, 0.2),
            wv: mat(&mut rng, kv_dim, DIM, Kind::Q8_0, 0.2),
            wo: mat(&mut rng, DIM, q_dim, Kind::Q5_K, 0.2),
            ffn_norm: norm(&mut rng),
            gate: mat(&mut rng, HIDDEN, DIM, Kind::Q4_K, 0.2),
            up: mat(&mut rng, HIDDEN, DIM, Kind::Q4_K, 0.2),
            down: mat(&mut rng, DIM, HIDDEN, Kind::Q6_K, 0.14),
        },
        Layer {
            attn_norm: norm(&mut rng),
            wq: mat(&mut rng, q_dim, DIM, Kind::F16, 0.2),
            wk: mat(&mut rng, kv_dim, DIM, Kind::BF16, 0.2),
            wv: mat(&mut rng, kv_dim, DIM, Kind::F32, 0.2),
            wo: mat(&mut rng, DIM, q_dim, Kind::Q4_0, 0.2),
            ffn_norm: norm(&mut rng),
            gate: mat(&mut rng, HIDDEN, DIM, Kind::Q8_0, 0.2),
            up: mat(&mut rng, HIDDEN, DIM, Kind::Q8_0, 0.2),
            down: mat(&mut rng, DIM, HIDDEN, Kind::Q4_0, 0.14),
        },
    ];
    let model = Model {
        embd: mat(&mut rng, vocab_n, DIM, Kind::Q4_K, 1.0),
        output_norm: norm(&mut rng),
        output: mat(&mut rng, vocab_n, DIM, Kind::Q6_K, 0.2),
        layers,
    };
    let gguf = build_gguf(&model, &vocab, &scores, &types);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("e2e");
    std::fs::create_dir_all(&tmp).unwrap();
    let gguf_path = tmp.join("tiny.gguf");
    std::fs::write(&gguf_path, &gguf).unwrap();
    let out = tmp.join("tiny_bin");

    // Tokens the binary will see (same blob builder as the compiler).
    let parsed = GgufFile::parse(&gguf, gguf.len() as u64).unwrap();
    let (blob, _) = build_blob(&parsed).unwrap();
    let tok = Tokenizer::new(Box::leak(blob.into_boxed_slice())).unwrap();
    let prompt = "the capital of hello world is";
    let ids = tok.encode(prompt, true, true);
    assert_eq!(ids.len(), 7, "{ids:?}");
    let expected = reference_logits(&model, &ids);

    let status = Command::new(env!("CARGO_BIN_EXE_aot-llm"))
        .args(["compile", "--model"])
        .arg(&gguf_path)
        .arg("--output")
        .arg(&out)
        .arg("--quiet")
        .status()
        .expect("running aot-llm");
    assert!(status.success(), "aot-llm compile failed");

    let run = Command::new(&out).args(["--prompt", prompt, "--dump-logits", "--quiet"]).output().expect("running compiled binary");
    assert!(run.status.success(), "binary failed: {}", String::from_utf8_lossy(&run.stderr));
    let got: Vec<f32> = String::from_utf8_lossy(&run.stdout).split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert_eq!(got.len(), expected.len());

    let max_abs = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max_err = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    eprintln!("max |logit| {max_abs:.3}, max abs error {max_err:.4}, argmax {} vs {}", argmax(&got), argmax(&expected));
    // The reference applies the same activation quantization, so only float
    // summation-order noise remains.
    assert!(max_err <= 2e-3 * max_abs + 1e-3, "logits differ: max error {max_err} (range {max_abs})");
    assert_eq!(argmax(&got), argmax(&expected));

    // Generation runs and stops cleanly.
    let gen = Command::new(&out).args(["--prompt", prompt, "-n", "8", "--tokens", "--quiet", "--threads", "2"]).output().unwrap();
    assert!(gen.status.success());
    let n = String::from_utf8_lossy(&gen.stdout).split_whitespace().count();
    assert!((1..=8).contains(&n), "generated {n} tokens");
}
