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
pub fn attention(
    pool: &Pool,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    scores: &mut [f32],
    d: &Dims,
    ctx: usize,
    layer: usize,
    pos: usize,
) {
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
    fn attention_single_token_returns_value() {
        let d = Dims {
            dim: 8,
            hidden: 8,
            n_layer: 1,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 4,
            rot_dim: 4,
            vocab: 1,
            n_ctx_train: 4,
            eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
        };
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
