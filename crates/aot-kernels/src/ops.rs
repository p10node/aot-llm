//! Non-matmul operators of the Transformer block.

use super::pool::{split, Pool, SendPtr};
use super::state::Dims;

/// `out = x / rms(x) * w`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    debug_assert!(out.len() >= n && w.len() >= n);
    let mut ss = 0f32;
    for &v in x {
        ss += v * v;
    }
    let r = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * r * w[i];
    }
}

/// Fill `cs` with interleaved `(cos, sin)` of `pos * inv_freq[i]` for every
/// rotated dimension pair.
///
/// Never inlined on purpose: the single-token and batched paths must produce
/// bit-identical tables, and LLVM lowers `sin`/`cos` differently depending on
/// the inlining context (separate calls vs. a fused `sincos`), which costs an
/// ulp and then diverges over many layers.
#[inline(never)]
pub fn rope_cos_sin(cs: &mut [f32], inv_freq: &[f32], pos: f32) {
    for (i, &f) in inv_freq.iter().enumerate() {
        let (s, c) = (pos * f).sin_cos();
        cs[2 * i] = c;
        cs[2 * i + 1] = s;
    }
}

/// Apply rotary position embedding in place to `n_heads` heads of
/// `head_dim`, rotating adjacent pairs (`x[2i], x[2i+1]`) of the first
/// `rot_dim` dimensions. This is the "normal" RoPE layout used by Llama in
/// GGUF files (the converter permutes Q/K so adjacent pairs are rotated).
pub fn rope(x: &mut [f32], n_heads: usize, head_dim: usize, rot_dim: usize, cs: &[f32]) {
    for h in 0..n_heads {
        let v = &mut x[h * head_dim..h * head_dim + rot_dim];
        for i in 0..rot_dim / 2 {
            let c = cs[2 * i];
            let s = cs[2 * i + 1];
            let x0 = v[2 * i];
            let x1 = v[2 * i + 1];
            v[2 * i] = x0 * c - x1 * s;
            v[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/// RoPE in the "NEOX" layout used by Qwen2 / Qwen3 (and GPT-NeoX): the
/// rotated pairs are `(x[i], x[i + rot_dim / 2])` instead of adjacent
/// elements. `cs` is the same interleaved (cos, sin) table as for [`rope`].
pub fn rope_neox(x: &mut [f32], n_heads: usize, head_dim: usize, rot_dim: usize, cs: &[f32]) {
    let half = rot_dim / 2;
    for h in 0..n_heads {
        let v = &mut x[h * head_dim..h * head_dim + rot_dim];
        for i in 0..half {
            let c = cs[2 * i];
            let s = cs[2 * i + 1];
            let x0 = v[i];
            let x1 = v[i + half];
            v[i] = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }
}

/// RMSNorm applied in place to each of the `n_heads` heads of `head_dim`
/// elements with one shared weight vector of `head_dim` (Qwen3's q / k norm).
pub fn rmsnorm_heads(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    debug_assert!(w.len() >= head_dim);
    for h in 0..n_heads {
        let v = &mut x[h * head_dim..(h + 1) * head_dim];
        let mut ss = 0f32;
        for &a in v.iter() {
            ss += a * a;
        }
        let r = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            v[i] = v[i] * r * w[i];
        }
    }
}

/// `x[i] += b[i]` for `m` rows of `b.len()` elements (projection biases).
pub fn add_bias_rows(x: &mut [f32], b: &[f32], m: usize) {
    let n = b.len();
    for i in 0..m {
        add_inplace(&mut x[i * n..(i + 1) * n], b);
    }
}

/// In-place softmax.
pub fn softmax(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        max = max.max(v);
    }
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `h = silu(h) * up` (SwiGLU gating).
pub fn silu_mul(h: &mut [f32], up: &[f32]) {
    for (a, &b) in h.iter_mut().zip(up) {
        let v = *a;
        *a = v / (1.0 + (-v).exp()) * b;
    }
}

/// `x += y`.
pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    for (a, &b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Copy the current key/value vectors into the cache at `(layer, pos)`.
#[allow(clippy::too_many_arguments)]
pub fn store_kv(k_cache: &mut [f32], v_cache: &mut [f32], k: &[f32], v: &[f32], d: &Dims, ctx: usize, layer: usize, pos: usize) {
    let kv = d.kv_dim();
    let at = (layer * ctx + pos) * kv;
    k_cache[at..at + kv].copy_from_slice(&k[..kv]);
    v_cache[at..at + kv].copy_from_slice(&v[..kv]);
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Causal multi-head attention for the token at `pos` over the cache of
/// `layer`. Heads are distributed across the pool; each head reads the
/// query `q`, scores all cached keys, softmaxes and mixes cached values
/// into `out`.
#[allow(clippy::too_many_arguments)]
pub fn attention(pool: &Pool, out: &mut [f32], q: &[f32], k_cache: &[f32], v_cache: &[f32], scores: &mut [f32], d: &Dims, ctx: usize, layer: usize, pos: usize) {
    let hd = d.head_dim;
    let kv = d.kv_dim();
    let gqa = d.gqa();
    let scale = 1.0 / (hd as f32).sqrt();
    let base = layer * ctx * kv;
    let n = pos + 1;
    let o = SendPtr(out.as_mut_ptr());
    let sc = SendPtr(scores.as_mut_ptr());
    pool.run(&|tid, nth| {
        for h in split(d.n_head, tid, nth) {
            let kvh = h / gqa;
            let qh = &q[h * hd..(h + 1) * hd];
            // SAFETY: each head owns its own score row and output slice.
            let s = unsafe { sc.slice_mut(h * ctx..h * ctx + n) };
            for t in 0..n {
                let at = base + t * kv + kvh * hd;
                s[t] = dot(qh, &k_cache[at..at + hd]) * scale;
            }
            softmax(s);
            let oh = unsafe { o.slice_mut(h * hd..(h + 1) * hd) };
            oh.fill(0.0);
            for t in 0..n {
                let a = s[t];
                let at = base + t * kv + kvh * hd;
                let vt = &v_cache[at..at + hd];
                for i in 0..hd {
                    oh[i] += a * vt[i];
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -1.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0] && v[0] > v[3]);
    }

    #[test]
    fn rmsnorm_unit_weights() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0f32; 2];
        rmsnorm(&mut out, &x, &w, 0.0);
        let rms = ((9.0 + 16.0) / 2.0f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-6);
        assert!((out[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_pairs() {
        let inv = [1.0f32, 0.5];
        let mut cs = [0f32; 4];
        rope_cos_sin(&mut cs, &inv, std::f32::consts::FRAC_PI_2);
        let mut x = [1.0f32, 0.0, 1.0, 0.0];
        rope(&mut x, 1, 4, 4, &cs);
        // First pair rotated by pi/2, second by pi/4.
        assert!(x[0].abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
        let r = std::f32::consts::FRAC_1_SQRT_2;
        assert!((x[2] - r).abs() < 1e-6 && (x[3] - r).abs() < 1e-6);
    }

    #[test]
    fn rope_neox_rotates_half_offset_pairs() {
        let inv = [1.0f32, 0.5];
        let mut cs = [0f32; 4];
        rope_cos_sin(&mut cs, &inv, std::f32::consts::FRAC_PI_2);
        // head_dim 4: pairs (0, 2) rotated by pi/2 and (1, 3) by pi/4.
        let mut x = [1.0f32, 1.0, 0.0, 0.0];
        rope_neox(&mut x, 1, 4, 4, &cs);
        assert!(x[0].abs() < 1e-6 && (x[2] - 1.0).abs() < 1e-6);
        let r = std::f32::consts::FRAC_1_SQRT_2;
        assert!((x[1] - r).abs() < 1e-6 && (x[3] - r).abs() < 1e-6);
        // Equivalent to the adjacent-pair layout after a permutation of the head.
        let mut cs8 = [0f32; 8];
        let inv8 = [1.0f32, 0.7, 0.4, 0.1];
        rope_cos_sin(&mut cs8, &inv8, 3.0);
        let src: Vec<f32> = (0..8).map(|i| i as f32 * 0.37 - 1.0).collect();
        let mut neox = src.clone();
        rope_neox(&mut neox, 1, 8, 8, &cs8);
        let mut perm: Vec<f32> = (0..4).flat_map(|i| [src[i], src[i + 4]]).collect();
        rope(&mut perm, 1, 8, 8, &cs8);
        for i in 0..4 {
            assert!((neox[i] - perm[2 * i]).abs() < 1e-6 && (neox[i + 4] - perm[2 * i + 1]).abs() < 1e-6);
        }
    }

    #[test]
    fn rmsnorm_heads_normalises_each_head() {
        let w = [2.0f32, 1.0];
        let mut x = [3.0f32, 4.0, 0.5, 0.5];
        rmsnorm_heads(&mut x, &w, 2, 2, 0.0);
        let rms0 = ((9.0 + 16.0) / 2.0f32).sqrt();
        let rms1 = 0.5f32;
        assert!((x[0] - 2.0 * 3.0 / rms0).abs() < 1e-5 && (x[1] - 4.0 / rms0).abs() < 1e-5);
        assert!((x[2] - 2.0 * 0.5 / rms1).abs() < 1e-5 && (x[3] - 0.5 / rms1).abs() < 1e-5);
        let mut rows = [3.0f32, 4.0, 0.5, 0.5, 3.0, 4.0, 0.5, 0.5];
        rmsnorm_heads_rows(&mut rows, &w, 2, 2, 2, 0.0);
        assert_eq!(&rows[..4], &x);
        assert_eq!(&rows[4..], &x);
    }

    #[test]
    fn attention_single_token_returns_value() {
        let d = Dims { dim: 8, hidden: 8, n_layer: 1, n_head: 2, n_kv_head: 1, head_dim: 4, rot_dim: 4, vocab: 1, n_ctx_train: 4, eps: 1e-5, rope_base: 10000.0, rope_scale: 1.0 };
        let ctx = 4;
        let pool = Pool::new(2);
        let q = vec![1.0f32; 8];
        let mut kc = vec![0f32; ctx * 4];
        let mut vc = vec![0f32; ctx * 4];
        store_kv(&mut kc, &mut vc, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], &d, ctx, 0, 0);
        let mut scores = vec![0f32; 2 * ctx];
        let mut out = vec![0f32; 8];
        attention(&pool, &mut out, &q, &kc, &vc, &mut scores, &d, ctx, 0, 0);
        assert_eq!(&out[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&out[4..], &[5.0, 6.0, 7.0, 8.0]);
    }
}

// ---------------------------------------------------------------------------
// Row-batched variants used for prompt processing
// ---------------------------------------------------------------------------

/// [`rmsnorm`] applied to each of the `m` rows of `x` (row length `w.len()`).
pub fn rmsnorm_rows(out: &mut [f32], x: &[f32], w: &[f32], eps: f32, m: usize) {
    let n = w.len();
    for i in 0..m {
        rmsnorm(&mut out[i * n..(i + 1) * n], &x[i * n..(i + 1) * n], w, eps);
    }
}

/// Interleaved (cos, sin) tables for positions `pos0 .. pos0 + m`, one row
/// of `2 * inv_freq.len()` per position.
pub fn rope_cos_sin_rows(cs: &mut [f32], inv_freq: &[f32], pos0: usize, m: usize, scale: f32) {
    let rd = 2 * inv_freq.len();
    for i in 0..m {
        rope_cos_sin(&mut cs[i * rd..(i + 1) * rd], inv_freq, (pos0 + i) as f32 / scale);
    }
}

/// [`rope`] applied to each of the `m` rows of `x` with its own position table.
pub fn rope_rows(x: &mut [f32], m: usize, n_heads: usize, head_dim: usize, rot_dim: usize, cs: &[f32]) {
    let n = n_heads * head_dim;
    for i in 0..m {
        rope(&mut x[i * n..(i + 1) * n], n_heads, head_dim, rot_dim, &cs[i * rot_dim..(i + 1) * rot_dim]);
    }
}

/// [`rope_neox`] applied to each of the `m` rows of `x` with its own position table.
pub fn rope_neox_rows(x: &mut [f32], m: usize, n_heads: usize, head_dim: usize, rot_dim: usize, cs: &[f32]) {
    let n = n_heads * head_dim;
    for i in 0..m {
        rope_neox(&mut x[i * n..(i + 1) * n], n_heads, head_dim, rot_dim, &cs[i * rot_dim..(i + 1) * rot_dim]);
    }
}

/// [`rmsnorm_heads`] over `m` rows of `n_heads * head_dim` elements.
pub fn rmsnorm_heads_rows(x: &mut [f32], w: &[f32], m: usize, n_heads: usize, head_dim: usize, eps: f32) {
    rmsnorm_heads(&mut x[..m * n_heads * head_dim], w, m * n_heads, head_dim, eps);
}

// ---------------------------------------------------------------------------
// Parallel element-wise operators for prompt batches. The arithmetic per
// element is exactly that of the serial functions above (same order), so the
// results stay bit-identical to the token-by-token path; only the work is
// spread over the pool, which otherwise sits idle between the matrix products
// while the main thread normalises, gates and quantizes tens of thousands of
// values per layer.
// ---------------------------------------------------------------------------

/// Elements per parallel chunk: enough work to amortise the hand-off.
const PAR_CHUNK: usize = 4096;

/// [`rmsnorm_rows`] with the rows distributed over the pool.
pub fn rmsnorm_rows_par(pool: &Pool, out: &mut [f32], x: &[f32], w: &[f32], eps: f32, m: usize) {
    let n = w.len();
    if m < 2 || pool.n_threads() == 1 {
        return rmsnorm_rows(out, x, w, eps, m);
    }
    let o = SendPtr(out.as_mut_ptr());
    pool.run_chunks(m, 1, &|range| {
        for i in range {
            // SAFETY: each row is written by exactly one chunk.
            let dst = unsafe { o.slice_mut(i * n..(i + 1) * n) };
            rmsnorm(dst, &x[i * n..(i + 1) * n], w, eps);
        }
    });
}

/// [`silu_mul`] over the pool.
pub fn silu_mul_par(pool: &Pool, h: &mut [f32], up: &[f32]) {
    let n = h.len().min(up.len());
    if n < 2 * PAR_CHUNK || pool.n_threads() == 1 {
        return silu_mul(&mut h[..n], &up[..n]);
    }
    let o = SendPtr(h.as_mut_ptr());
    pool.run_chunks(n, PAR_CHUNK, &|range| {
        // SAFETY: disjoint element ranges.
        let dst = unsafe { o.slice_mut(range.clone()) };
        silu_mul(dst, &up[range]);
    });
}

/// [`add_inplace`] over the pool.
pub fn add_inplace_par(pool: &Pool, x: &mut [f32], y: &[f32]) {
    let n = x.len().min(y.len());
    if n < 2 * PAR_CHUNK || pool.n_threads() == 1 {
        return add_inplace(&mut x[..n], &y[..n]);
    }
    let o = SendPtr(x.as_mut_ptr());
    pool.run_chunks(n, PAR_CHUNK, &|range| {
        // SAFETY: disjoint element ranges.
        let dst = unsafe { o.slice_mut(range.clone()) };
        add_inplace(dst, &y[range]);
    });
}

/// [`super::quant::quantize_q8_k`] with the blocks distributed over the pool.
pub fn quantize_q8_k_par(pool: &Pool, x: &[f32], out: &mut Vec<super::quant::BlockQ8_K>) {
    use super::quant::{quantize_q8_k, BlockQ8_K, QK_K};
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;
    if nb < 16 || pool.n_threads() == 1 {
        return quantize_q8_k(x, out);
    }
    out.resize(nb, BlockQ8_K::default());
    let o = SendPtr(out.as_mut_ptr());
    pool.run_chunks(nb, 4, &|range| {
        // SAFETY: disjoint block ranges; each block depends only on its 256 inputs.
        let dst = unsafe { o.slice_mut(range.clone()) };
        let mut tmp = Vec::with_capacity(range.len());
        quantize_q8_k(&x[range.start * QK_K..range.end * QK_K], &mut tmp);
        dst.copy_from_slice(&tmp);
    });
}

/// [`super::quant::quantize_q8_0`] with the blocks distributed over the pool.
pub fn quantize_q8_0_par(pool: &Pool, x: &[f32], out: &mut Vec<super::quant::BlockQ8_0>) {
    use super::quant::{quantize_q8_0, BlockQ8_0, QK8_0};
    debug_assert_eq!(x.len() % QK8_0, 0);
    let nb = x.len() / QK8_0;
    if nb < 128 || pool.n_threads() == 1 {
        return quantize_q8_0(x, out);
    }
    out.resize(nb, BlockQ8_0::default());
    let o = SendPtr(out.as_mut_ptr());
    pool.run_chunks(nb, 32, &|range| {
        // SAFETY: disjoint block ranges.
        let dst = unsafe { o.slice_mut(range.clone()) };
        let mut tmp = Vec::with_capacity(range.len());
        quantize_q8_0(&x[range.start * QK8_0..range.end * QK8_0], &mut tmp);
        dst.copy_from_slice(&tmp);
    });
}

/// Store `m` consecutive key/value rows starting at `pos0`.
#[allow(clippy::too_many_arguments)]
pub fn store_kv_rows(k_cache: &mut [f32], v_cache: &mut [f32], k: &[f32], v: &[f32], d: &Dims, ctx: usize, layer: usize, pos0: usize, m: usize) {
    let kv = d.kv_dim();
    let at = (layer * ctx + pos0) * kv;
    k_cache[at..at + m * kv].copy_from_slice(&k[..m * kv]);
    v_cache[at..at + m * kv].copy_from_slice(&v[..m * kv]);
}

/// Causal attention for `m` query rows at positions `pos0 .. pos0 + m`
/// (their keys/values must already be in the cache). Work items are
/// (row, head) pairs; `scratch` needs `n_threads * ctx` floats.
#[allow(clippy::too_many_arguments)]
pub fn attention_rows(pool: &Pool, out: &mut [f32], q: &[f32], k_cache: &[f32], v_cache: &[f32], scratch: &mut [f32], d: &Dims, ctx: usize, layer: usize, pos0: usize, m: usize) {
    let hd = d.head_dim;
    let kv = d.kv_dim();
    let qd = d.q_dim();
    let gqa = d.gqa();
    let scale = 1.0 / (hd as f32).sqrt();
    let base = layer * ctx * kv;
    assert!(scratch.len() >= pool.n_threads() * ctx, "attention scratch too small");
    let o = SendPtr(out.as_mut_ptr());
    let sc = SendPtr(scratch.as_mut_ptr());
    pool.run(&|tid, nth| {
        // SAFETY: one scratch row per thread.
        let s_all = unsafe { sc.slice_mut(tid * ctx..(tid + 1) * ctx) };
        for task in split(m * d.n_head, tid, nth) {
            let i = task / d.n_head;
            let h = task % d.n_head;
            let kvh = h / gqa;
            let n = pos0 + i + 1;
            let qh = &q[i * qd + h * hd..i * qd + (h + 1) * hd];
            let s = &mut s_all[..n];
            for t in 0..n {
                let at = base + t * kv + kvh * hd;
                s[t] = dot(qh, &k_cache[at..at + hd]) * scale;
            }
            softmax(s);
            // SAFETY: each (row, head) output slice is owned by exactly one task.
            let oh = unsafe { o.slice_mut(i * qd + h * hd..i * qd + (h + 1) * hd) };
            oh.fill(0.0);
            for t in 0..n {
                let a = s[t];
                let at = base + t * kv + kvh * hd;
                let vt = &v_cache[at..at + hd];
                for j in 0..hd {
                    oh[j] += a * vt[j];
                }
            }
        }
    });
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn parallel_elementwise_matches_serial_bitwise() {
        let pool = Pool::new(4);
        let mut seed = 11u64;
        let (m, n) = (9usize, 2048usize);
        let x: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 3.0).collect();
        let w: Vec<f32> = (0..n).map(|_| 1.0 + lcg(&mut seed) * 0.3).collect();
        let mut a = vec![0f32; m * n];
        let mut b = vec![0f32; m * n];
        rmsnorm_rows(&mut a, &x, &w, 1e-5, m);
        rmsnorm_rows_par(&pool, &mut b, &x, &w, 1e-5, m);
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
        let up: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 2.0).collect();
        let mut h1 = x.clone();
        let mut h2 = x.clone();
        silu_mul(&mut h1, &up);
        silu_mul_par(&pool, &mut h2, &up);
        assert!(h1.iter().zip(&h2).all(|(p, q)| p.to_bits() == q.to_bits()));
        add_inplace(&mut h1, &up);
        add_inplace_par(&pool, &mut h2, &up);
        assert!(h1.iter().zip(&h2).all(|(p, q)| p.to_bits() == q.to_bits()));
        let (mut q1, mut q2) = (Vec::new(), Vec::new());
        super::super::quant::quantize_q8_k(&x, &mut q1);
        quantize_q8_k_par(&pool, &x, &mut q2);
        assert!(q1.iter().zip(&q2).all(|(p, q)| p.d.to_bits() == q.d.to_bits() && p.qs == q.qs && p.bsums == q.bsums));
        let (mut r1, mut r2) = (Vec::new(), Vec::new());
        super::super::quant::quantize_q8_0(&x, &mut r1);
        quantize_q8_0_par(&pool, &x, &mut r2);
        assert!(r1.iter().zip(&r2).all(|(p, q)| p.d.to_bits() == q.d.to_bits() && p.qs == q.qs));
    }

    #[test]
    fn attention_rows_matches_single_bitwise() {
        let d = Dims { dim: 2048, hidden: 1, n_layer: 1, n_head: 32, n_kv_head: 4, head_dim: 64, rot_dim: 64, vocab: 1, n_ctx_train: 1, eps: 1e-5, rope_base: 1.0, rope_scale: 1.0 };
        let ctx = 128;
        let kv = d.kv_dim();
        let pool = Pool::new(3);
        let mut seed = 7u64;
        let kc: Vec<f32> = (0..ctx * kv).map(|_| lcg(&mut seed)).collect();
        let vc: Vec<f32> = (0..ctx * kv).map(|_| lcg(&mut seed)).collect();
        for pos in [0usize, 1, 7, 51, 52, 53, 100] {
            let q: Vec<f32> = (0..d.q_dim()).map(|_| lcg(&mut seed) * 4.0).collect();
            let mut scratch = vec![0f32; 32 * ctx];
            let mut single = vec![0f32; d.q_dim()];
            attention(&pool, &mut single, &q, &kc, &vc, &mut scratch, &d, ctx, 0, pos);
            let mut batched = vec![0f32; d.q_dim()];
            attention_rows(&pool, &mut batched, &q, &kc, &vc, &mut scratch, &d, ctx, 0, pos, 1);
            let diff = single.iter().zip(&batched).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(diff, 0, "pos {pos}: {diff} elements differ");
        }
    }
}
