//! Metal kernels against the CPU reference implementations: every weight
//! format of the matrix product (single vector and batched, partial lane
//! iterations, residual accumulation), RMSNorm, RoPE, attention for both
//! head sizes and the SwiGLU gate. Skipped (with a message) when no Metal
//! device is available.
#![cfg(target_os = "macos")]

use aot_kernels::metal::{BufId, Gpu, MvOpts};
use aot_kernels::ops;
use aot_kernels::pool::Pool;
use aot_kernels::quant::{dequant_row, Kind};
use aot_kernels::quantize::quantize_row;
use aot_kernels::state::{Buf, Dims, State, Tensor};

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
    fn fill(&mut self, v: &mut [f32], a: f32) {
        for x in v {
            *x = self.uniform(a);
        }
    }
}

fn dims(head_dim: usize) -> Dims {
    Dims { dim: 2560, hidden: 2560, n_layer: 2, n_head: 4, n_kv_head: 2, head_dim, rot_dim: head_dim, vocab: 2560, n_ctx_train: 64, eps: 1e-5, rope_base: 10000.0, rope_scale: 1.0 }
}

/// Leak a page-aligned copy of `bytes` as the weights blob.
fn weights_blob(bytes: &[u8]) -> &'static [u8] {
    let mut b: Buf<u8> = Buf::zeroed(bytes.len().max(1));
    b[..bytes.len()].copy_from_slice(bytes);
    let s: &'static [u8] = Box::leak(Box::new(b));
    &s[..bytes.len()]
}

fn gpu_or_skip(weights: &'static [u8], state: &State, d: &Dims) -> Option<Gpu> {
    match Gpu::new(weights, state, d) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn max_err(got: &[f32], want: &[f64]) -> (f64, f64) {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    let err = got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
    (err, scale)
}

#[test]
fn matmul_every_kind_matches_dequantized_reference() {
    let mut rng = Rng(0xC0FFEE);
    let kinds = [Kind::F32, Kind::F16, Kind::BF16, Kind::Q4_0, Kind::Q8_0, Kind::Q4_K, Kind::Q5_K, Kind::Q6_K];
    // (rows, cols): rows not a multiple of 16 (partial simdgroups) and column
    // counts that end inside a lane iteration for every block size.
    // Row counts also cover the 1 / 2 / 4 rows-per-simdgroup kernels.
    for (rows, cols) in [(100usize, 768usize), (37, 2560), (1500, 256), (4100, 256)] {
        let mut blob = Vec::new();
        let mut tensors = Vec::new();
        let mut dense: Vec<Vec<f32>> = Vec::new();
        for &kind in &kinds {
            while blob.len() % 32 != 0 {
                blob.push(0);
            }
            let off = blob.len();
            let mut f = Vec::with_capacity(rows * cols);
            for _ in 0..rows {
                let mut row = vec![0f32; cols];
                rng.fill(&mut row, 1.0);
                let enc = quantize_row(kind, &row);
                let mut dec = vec![0f32; cols];
                dequant_row(kind, &enc, &mut dec);
                f.extend_from_slice(&dec);
                blob.extend_from_slice(&enc);
            }
            tensors.push(Tensor { name: "w", kind, off, len: blob.len() - off, rows, cols });
            dense.push(f);
        }
        let w = weights_blob(&blob);
        let d = dims(64);
        let m = 11;
        let state = State::new(&d, 64, None, 1, 32);
        let Some(mut gpu) = gpu_or_skip(w, &state, &d) else { return };
        let mut x = vec![0f32; m * cols];
        rng.fill(&mut x, 1.0);
        // Rows of `cols` floats, packed; the GPU reads them with x_stride = cols.
        let mut state = state;
        state.bx[..m * cols].copy_from_slice(&x);
        for (ti, t) in tensors.iter().enumerate() {
            let f = &dense[ti];
            let xs = &x;
            let reference: Vec<f64> = (0..m).flat_map(|i| (0..rows).map(move |r| (0..cols).map(|c| f[r * cols + c] as f64 * xs[i * cols + c] as f64).sum::<f64>())).collect();
            for &mm in &[1usize, 2, m] {
                gpu.begin();
                gpu.matmul(BufId::Bxb, 0, t, BufId::Bx, 0, mm, false);
                gpu.finish();
                let got = &state.bxb[..mm * rows];
                let (err, scale) = max_err(got, &reference[..mm * rows]);
                eprintln!("{:>5} {rows}x{cols} m={mm}: max abs err {err:.2e} (max |ref| {scale:.2})", t.kind.name());
                assert!(err <= 2e-5 * scale + 1e-5, "{} {rows}x{cols} m={mm}: error {err} vs scale {scale}", t.kind.name());
            }
            // Residual accumulation: y += W x on top of the previous result.
            gpu.begin();
            gpu.matmul(BufId::Bxb, 0, t, BufId::Bx, 0, m, true);
            gpu.finish();
            let twice: Vec<f64> = reference.iter().map(|v| 2.0 * v).collect();
            let (err, scale) = max_err(&state.bxb[..m * rows], &twice);
            assert!(err <= 4e-5 * scale + 1e-5, "{} accumulate: error {err} vs scale {scale}", t.kind.name());
        }
    }
}

#[test]
fn rmsnorm_rope_silu_match_cpu_ops() {
    let mut rng = Rng(0xBEEF);
    let d = dims(64);
    let n = d.dim;
    let m = 3;
    let mut wv = vec![0f32; n];
    rng.fill(&mut wv, 1.0);
    let mut blob: Vec<u8> = wv.iter().flat_map(|v| v.to_le_bytes()).collect();
    while !blob.len().is_multiple_of(32) {
        blob.push(0);
    }
    let w = weights_blob(&blob);
    let norm_w = Tensor { name: "norm", kind: Kind::F32, off: 0, len: n * 4, rows: 1, cols: n };
    let mut state = State::new(&d, 64, None, 1, 16);
    let Some(mut gpu) = gpu_or_skip(w, &state, &d) else { return };

    // RMSNorm over m rows (reading from an element offset).
    rng.fill(&mut state.bx[..(m + 1) * n], 3.0);
    let mut want = vec![0f32; m * n];
    ops::rmsnorm_rows(&mut want, &state.bx[n..(m + 1) * n], &wv, d.eps, m);
    gpu.begin();
    gpu.rmsnorm(BufId::Bxb, BufId::Bx, n, &norm_w, n, m, d.eps);
    gpu.finish();
    let want64: Vec<f64> = want.iter().map(|&v| v as f64).collect();
    let (err, scale) = max_err(&state.bxb[..m * n], &want64);
    eprintln!("rmsnorm: max abs err {err:.2e} (max {scale:.2})");
    assert!(err <= 1e-5 * scale + 1e-6);

    // RoPE on q rows and on k rows living in the KV cache.
    let qd = d.q_dim();
    let kv = d.kv_dim();
    let pos0 = 5;
    ops::rope_cos_sin_rows(&mut state.brope_cs, &state.inv_freq, pos0, m, d.rope_scale);
    rng.fill(&mut state.bq[..m * qd], 1.0);
    let k_off = 7 * kv;
    rng.fill(&mut state.k_cache[k_off..k_off + m * kv], 1.0);
    let mut q_ref = state.bq[..m * qd].to_vec();
    let mut k_ref = state.k_cache[k_off..k_off + m * kv].to_vec();
    ops::rope_rows(&mut q_ref, m, d.n_head, d.head_dim, d.rot_dim, &state.brope_cs);
    ops::rope_rows(&mut k_ref, m, d.n_kv_head, d.head_dim, d.rot_dim, &state.brope_cs);
    gpu.begin();
    gpu.rope(BufId::Bq, BufId::KCache, k_off, m, BufId::BropeCs, &d);
    gpu.finish();
    let (eq, sq) = max_err(&state.bq[..m * qd], &q_ref.iter().map(|&v| v as f64).collect::<Vec<_>>());
    let (ek, sk) = max_err(&state.k_cache[k_off..k_off + m * kv], &k_ref.iter().map(|&v| v as f64).collect::<Vec<_>>());
    eprintln!("rope: q err {eq:.2e} k err {ek:.2e}");
    assert!(eq <= 1e-5 * sq && ek <= 1e-5 * sk);

    // SwiGLU gate.
    let nh = 1000;
    rng.fill(&mut state.bhb[..nh], 4.0);
    rng.fill(&mut state.bhb2[..nh], 4.0);
    let mut h_ref = state.bhb[..nh].to_vec();
    ops::silu_mul(&mut h_ref, &state.bhb2[..nh]);
    gpu.begin();
    gpu.silu_mul(BufId::Bhb, BufId::Bhb2, nh);
    gpu.finish();
    let (es, ss) = max_err(&state.bhb[..nh], &h_ref.iter().map(|&v| v as f64).collect::<Vec<_>>());
    eprintln!("silu_mul: err {es:.2e}");
    assert!(es <= 1e-5 * ss + 1e-6);
}

/// The fused decode kernels (RoPE in the projection epilogue, gate / up /
/// SwiGLU in one dispatch) against the separate operators.
#[test]
fn fused_decode_kernels_match_separate_ops() {
    let mut rng = Rng(0xF00D);
    let d = Dims { dim: 512, hidden: 1024, n_layer: 1, n_head: 4, n_kv_head: 2, head_dim: 64, rot_dim: 64, vocab: 16, n_ctx_train: 64, eps: 1e-5, rope_base: 10000.0, rope_scale: 1.0 };
    for kind in [Kind::Q4_K, Kind::Q6_K, Kind::Q4_0, Kind::F16] {
        let mut blob = Vec::new();
        let mut mk = |rng: &mut Rng, rows: usize, cols: usize| -> Tensor {
            while blob.len() % 32 != 0 {
                blob.push(0);
            }
            let off = blob.len();
            for _ in 0..rows {
                let mut row = vec![0f32; cols];
                rng.fill(&mut row, 0.5);
                blob.extend_from_slice(&quantize_row(kind, &row));
            }
            Tensor { name: "w", kind, off, len: blob.len() - off, rows, cols }
        };
        let wq = mk(&mut rng, d.q_dim(), d.dim);
        let wk = mk(&mut rng, d.kv_dim(), d.dim);
        let gate = mk(&mut rng, d.hidden, d.dim);
        let up = mk(&mut rng, d.hidden, d.dim);
        let w = weights_blob(&blob);
        let mut state = State::new(&d, 64, None, 1, 1);
        let Some(mut gpu) = gpu_or_skip(w, &state, &d) else { return };
        rng.fill(&mut state.xb, 1.0);
        let pos = 17;
        ops::rope_cos_sin(&mut state.rope_cs, &state.inv_freq, pos as f32);
        let kv_at = 3 * d.kv_dim();

        // Separate: projections, then rope on q and on the cache rows.
        gpu.begin();
        gpu.matmul(BufId::Q, 0, &wq, BufId::Xb, 0, 1, false);
        gpu.matmul(BufId::KCache, kv_at, &wk, BufId::Xb, 0, 1, false);
        gpu.rope(BufId::Q, BufId::KCache, kv_at, 1, BufId::RopeCs, &d);
        gpu.matmul(BufId::Hb, 0, &gate, BufId::Xb, 0, 1, false);
        gpu.matmul(BufId::Hb2, 0, &up, BufId::Xb, 0, 1, false);
        gpu.silu_mul(BufId::Hb, BufId::Hb2, d.hidden);
        gpu.finish();
        let q_ref: Vec<f64> = state.q.iter().map(|&v| v as f64).collect();
        let k_ref: Vec<f64> = state.k_cache[kv_at..kv_at + d.kv_dim()].iter().map(|&v| v as f64).collect();
        let h_ref: Vec<f64> = state.hb.iter().map(|&v| v as f64).collect();
        state.q.fill(0.0);
        state.hb.fill(0.0);
        state.k_cache[kv_at..kv_at + d.kv_dim()].fill(0.0);

        gpu.begin();
        gpu.matmul_rope(BufId::Q, 0, &wq, BufId::Xb, 0, BufId::RopeCs, &d);
        gpu.matmul_rope(BufId::KCache, kv_at, &wk, BufId::Xb, 0, BufId::RopeCs, &d);
        gpu.gate_up_silu(BufId::Hb, &gate, &up, BufId::Xb);
        gpu.finish();
        let (eq, sq) = max_err(&state.q, &q_ref);
        let (ek, sk) = max_err(&state.k_cache[kv_at..kv_at + d.kv_dim()], &k_ref);
        let (eh, sh) = max_err(&state.hb, &h_ref);
        eprintln!("{}: fused rope q err {eq:.2e} k err {ek:.2e}; fused gate err {eh:.2e} (max {sh:.2})", kind.name());
        assert!(eq <= 1e-5 * sq + 1e-6 && ek <= 1e-5 * sk + 1e-6, "{}: fused rope differs", kind.name());
        assert!(eh <= 1e-5 * sh + 1e-6, "{}: fused gate differs", kind.name());

    }
}

#[test]
fn attention_matches_cpu_for_both_head_sizes() {
    for head_dim in [64usize, 128] {
        let mut rng = Rng(0xA77 + head_dim as u64);
        let d = Dims { n_head: 4, n_kv_head: 2, head_dim, rot_dim: head_dim, ..dims(head_dim) };
        let ctx = 100;
        let mut state = State::new(&d, ctx, None, 2, 16);
        let w = weights_blob(&[0u8; 64]);
        let Some(mut gpu) = gpu_or_skip(w, &state, &d) else { return };
        let kv = d.kv_dim();
        let qd = d.q_dim();
        let layer = 1;
        let base = layer * ctx * kv;
        rng.fill(&mut state.k_cache[base..base + ctx * kv], 2.0);
        rng.fill(&mut state.v_cache[base..base + ctx * kv], 2.0);
        let pool = Pool::new(2);
        // Single query at several positions (crosses the 32-position block
        // boundaries and uses several simdgroups), then a batch of rows.
        for (pos0, m) in [(0usize, 1usize), (1, 1), (31, 1), (32, 1), (70, 1), (99, 1), (3, 9), (60, 13)] {
            rng.fill(&mut state.bq[..m * qd], 3.0);
            let mut want = vec![0f32; m * qd];
            let mut scratch = vec![0f32; 2 * ctx];
            ops::attention_rows(&pool, &mut want, &state.bq[..m * qd], &state.k_cache, &state.v_cache, &mut scratch, &d, ctx, layer, pos0, m);
            gpu.begin();
            gpu.attention(BufId::Bxb, BufId::Bq, layer, pos0, m, &d, ctx);
            gpu.finish();
            let (err, scale) = max_err(&state.bxb[..m * qd], &want.iter().map(|&v| v as f64).collect::<Vec<_>>());
            eprintln!("attention hd={head_dim} pos0={pos0} m={m}: err {err:.2e} (max {scale:.2})");
            assert!(err <= 2e-5 * scale + 1e-5, "hd={head_dim} pos0={pos0} m={m}: error {err} vs {scale}");
        }
    }
}

/// Per-operator GPU timings on TinyLlama-sized tensors (random bytes, one
/// distinct weight set per layer so caches behave as with a real model).
/// `cargo test -p aot-kernels --test metal --release -- --ignored --nocapture bench_ops`
#[test]
#[ignore]
fn bench_ops() {
    use std::time::Instant;
    let d = Dims { dim: 2048, hidden: 5632, n_layer: 22, n_head: 32, n_kv_head: 4, head_dim: 64, rot_dim: 64, vocab: 32000, n_ctx_train: 2048, eps: 1e-5, rope_base: 10000.0, rope_scale: 1.0 };
    let ctx = 256;
    let mut rng = Rng(0x5EED);
    // Layout per layer: wq, wk (Q4_K), wv (Q6_K), wo, gate, up (Q4_K), down (Q6_K); plus the Q6_K output head.
    let shapes: Vec<(&str, Kind, usize, usize)> = vec![
        ("wq", Kind::Q4_K, d.q_dim(), d.dim),
        ("wk", Kind::Q4_K, d.kv_dim(), d.dim),
        ("wv", Kind::Q6_K, d.kv_dim(), d.dim),
        ("wo", Kind::Q4_K, d.dim, d.q_dim()),
        ("gate", Kind::Q4_K, d.hidden, d.dim),
        ("up", Kind::Q4_K, d.hidden, d.dim),
        ("down", Kind::Q6_K, d.dim, d.hidden),
    ];
    let mut off = 0usize;
    let mut layers: Vec<Vec<Tensor>> = Vec::new();
    for _ in 0..d.n_layer {
        let mut l = Vec::new();
        for &(name, kind, rows, cols) in &shapes {
            let len = kind.row_bytes(cols) * rows;
            l.push(Tensor { name, kind, off, len, rows, cols });
            off += len.div_ceil(32) * 32;
        }
        layers.push(l);
    }
    let output = Tensor { name: "output", kind: Kind::Q6_K, off, len: Kind::Q6_K.row_bytes(d.dim) * d.vocab, rows: d.vocab, cols: d.dim };
    off += output.len;
    let norm = Tensor { name: "norm", kind: Kind::F32, off, len: d.dim * 4, rows: 1, cols: d.dim };
    off += norm.len;
    let mut blob: Buf<u8> = Buf::zeroed(off);
    for b in blob.iter_mut() {
        *b = (rng.next() & 0x7F) as u8; // small nibbles/scales: keeps values finite
    }
    let w: &'static [u8] = Box::leak(Box::new(blob));
    eprintln!("weights: {:.0} MB", off as f64 / 1e6);
    let mut state = State::new(&d, ctx, None, 1, 64);
    let Some(mut gpu) = gpu_or_skip(w, &state, &d) else { return };
    rng.fill(&mut state.x, 1.0);
    rng.fill(&mut state.bx, 1.0);
    let pos = 100;
    ops::rope_cos_sin(&mut state.rope_cs, &state.inv_freq, pos as f32);

    // Warm-up pass (raises the GPU clock, wires the pages), then the best of
    // three timed passes of `reps` dispatches in one command buffer.
    let time = |gpu: &mut Gpu, label: &str, reps: usize, bytes: usize, f: &dyn Fn(&mut Gpu)| {
        gpu.begin();
        for _ in 0..reps {
            f(gpu);
        }
        gpu.finish();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            gpu.begin();
            for _ in 0..reps {
                f(gpu);
            }
            gpu.finish();
            best = best.min(t.elapsed().as_secs_f64() / reps as f64);
        }
        eprintln!("{label:<40} {:8.1} us{}", best * 1e6, if bytes > 0 { format!("  {:6.1} GB/s", bytes as f64 / best / 1e9) } else { String::new() });
    };
    let l0 = &layers[0];
    // Rows per simdgroup, interleaved over three rounds so clock drift and
    // other GPU users affect every variant alike.
    for round in 0..3 {
        for &nr in &[1usize, 2, 4, 8] {
            gpu.set_mv_rows(Some(nr));
            for t in l0.iter().chain(std::iter::once(&output)) {
                let (src, dst) = if t.cols == d.hidden { (BufId::Hb, BufId::X) } else { (BufId::X, BufId::Hb) };
                time(&mut gpu, &format!("nr={nr} r{round} mv {:<6} {:>5}x{:<5} {}", t.name, t.rows, t.cols, t.kind.name()), 40, t.len, &|g| g.matmul(dst, 0, t, src, 0, 1, false));
            }
        }
    }
    gpu.set_mv_rows(None);
    for t in l0.iter().chain(std::iter::once(&output)) {
        let (src, dst) = if t.cols == d.hidden { (BufId::Hb, BufId::X) } else { (BufId::X, BufId::Hb) };
        time(&mut gpu, &format!("mv {:<6} {:>5}x{:<5} {}", t.name, t.rows, t.cols, t.kind.name()), 50, t.len, &|g| g.matmul(dst, 0, t, src, 0, 1, false));
    }
    for m in [8usize, 32, 64] {
        for t in [&l0[0], &l0[4], &l0[6]] {
            let (src, dst) = if t.cols == d.hidden { (BufId::Bhb, BufId::Bx) } else { (BufId::Bx, BufId::Bhb) };
            let flops = 2.0 * m as f64 * t.rows as f64 * t.cols as f64;
            time(&mut gpu, &format!("mm m={m:<2} {:<6} {:>5}x{:<5} {} ({:.1} GFLOP)", t.name, t.rows, t.cols, t.kind.name(), flops / 1e9), 20, t.len, &|g| g.matmul(dst, 0, t, src, 0, m, false));
        }
    }
    time(&mut gpu, "mv_rope wq", 50, l0[0].len, &|g| g.matmul_rope(BufId::Q, 0, &l0[0], BufId::Xb, 0, BufId::RopeCs, &d));
    time(&mut gpu, "mv_rope wk", 50, l0[1].len, &|g| g.matmul_rope(BufId::KCache, 0, &l0[1], BufId::Xb, 0, BufId::RopeCs, &d));
    time(&mut gpu, "gate_up_silu", 50, l0[4].len * 2, &|g| g.gate_up_silu(BufId::Hb, &l0[4], &l0[5], BufId::Xb));
    time(&mut gpu, "rmsnorm 2048", 50, 0, &|g| g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps));
    time(&mut gpu, "rope", 50, 0, &|g| g.rope(BufId::Q, BufId::KCache, 0, 1, BufId::RopeCs, &d));
    time(&mut gpu, "silu_mul 5632", 50, 0, &|g| g.silu_mul(BufId::Hb, BufId::Hb2, d.hidden));
    for p in [10usize, 100, 250] {
        time(&mut gpu, &format!("attention pos={p}"), 50, 0, &|g| g.attention(BufId::Xb, BufId::Q, 0, p, 1, &d, ctx));
    }
    time(&mut gpu, "attention m=64 pos0=100", 10, 0, &|g| g.attention(BufId::Bxb, BufId::Bq, 0, 100, 64, &d, ctx));

    // A whole decode step with the real operator sequence, and ablations.
    let token = |g: &mut Gpu, matmuls: bool, small: bool, attn: bool| {
        let kv = d.kv_dim();
        for (i, l) in layers.iter().enumerate() {
            let kv_at = (i * ctx + pos) * kv;
            if small {
                g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps);
            }
            if matmuls {
                g.matmul(BufId::Q, 0, &l[0], BufId::Xb, 0, 1, false);
                g.matmul(BufId::KCache, kv_at, &l[1], BufId::Xb, 0, 1, false);
                g.matmul(BufId::VCache, kv_at, &l[2], BufId::Xb, 0, 1, false);
            }
            if small {
                g.rope(BufId::Q, BufId::KCache, kv_at, 1, BufId::RopeCs, &d);
            }
            if attn {
                g.attention(BufId::Xb, BufId::Q, i, pos, 1, &d, ctx);
            }
            if matmuls {
                g.matmul(BufId::X, 0, &l[3], BufId::Xb, 0, 1, true);
            }
            if small {
                g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps);
            }
            if matmuls {
                g.matmul(BufId::Hb, 0, &l[4], BufId::Xb, 0, 1, false);
                g.matmul(BufId::Hb2, 0, &l[5], BufId::Xb, 0, 1, false);
            }
            if small {
                g.silu_mul(BufId::Hb, BufId::Hb2, d.hidden);
            }
            if matmuls {
                g.matmul(BufId::X, 0, &l[6], BufId::Hb, 0, 1, true);
            }
        }
        if small {
            g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps);
        }
        if matmuls {
            g.matmul(BufId::Logits, 0, &output, BufId::Xb, 0, 1, false);
        }
    };
    time(&mut gpu, "decode step (12/layer): everything", 10, off, &|g| token(g, true, true, true));
    time(&mut gpu, "decode step (12/layer): matmuls only", 10, off, &|g| token(g, true, false, false));
    time(&mut gpu, "decode step (12/layer): matmuls + small", 10, off, &|g| token(g, true, true, false));
    time(&mut gpu, "decode step (12/layer): small + attention", 10, 0, &|g| token(g, false, true, true));
    // The fused sequence the generated forward_gpu uses (8 dispatches per layer).
    let fused = |g: &mut Gpu| {
        let kv = d.kv_dim();
        for (i, l) in layers.iter().enumerate() {
            let kv_at = (i * ctx + pos) * kv;
            g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps);
            g.mv(BufId::Q, 0, &l[0], BufId::Xb, 0, Some(&d), MvOpts { rope: Some(BufId::RopeCs), ..MvOpts::default() });
            g.mv(BufId::KCache, kv_at, &l[1], BufId::Xb, 0, Some(&d), MvOpts { rope: Some(BufId::RopeCs), ..MvOpts::default() });
            g.mv(BufId::VCache, kv_at, &l[2], BufId::Xb, 0, None, MvOpts::default());
            g.attention(BufId::Xb, BufId::Q, i, pos, 1, &d, ctx);
            g.mv(BufId::X, 0, &l[3], BufId::Xb, 0, None, MvOpts { accumulate: true, ..MvOpts::default() });
            g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps);
            g.gate_up_silu(BufId::Hb, &l[4], &l[5], BufId::Xb);
            g.mv(BufId::X, 0, &l[6], BufId::Hb, 0, None, MvOpts { accumulate: true, ..MvOpts::default() });
        }
        g.rmsnorm(BufId::Xb, BufId::X, 0, &norm, d.dim, 1, d.eps);
        g.mv(BufId::Logits, 0, &output, BufId::Xb, 0, None, MvOpts::default());
    };
    time(&mut gpu, "decode step (8/layer, fused)", 10, off, &fused);
}
