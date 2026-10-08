//! Reference weight encoders producing ggml-compatible blocks.
//!
//! These are *valid* but not *optimal* quantizers (ggml searches for better
//! scales). They exist so tests and synthetic models can exercise every
//! kernel without depending on external tooling. They are not used on the
//! inference path.

use super::quant::*;

#[inline]
fn nearest(x: f32) -> i32 {
    x.round() as i32
}

/// Encode a row (multiple of 32 values) as Q4_0.
pub fn quantize_row_q4_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK8_0, 0);
    let mut out = Vec::with_capacity(x.len() / QK8_0 * Q4_0_SIZE);
    for blk in x.as_chunks::<QK8_0>().0 {
        // ggml picks the value with the largest magnitude (signed) and maps it
        // to -8, which uses the full asymmetric range of the nibble.
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let lo = (nearest(blk[j] * id) + 8).clamp(0, 15) as u8;
            let hi = (nearest(blk[j + 16] * id) + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Encode a row (multiple of 32 values) as Q8_0.
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK8_0, 0);
    let mut out = Vec::with_capacity(x.len() / QK8_0 * Q8_0_SIZE);
    for blk in x.as_chunks::<QK8_0>().0 {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for &v in blk {
            out.push(nearest(v * id).clamp(-127, 127) as i8 as u8);
        }
    }
    out
}

/// Compute per-sub-block (scale, min) pairs such that `x ~= scale*q - min`
/// with `q` in `0..=qmax`.
fn sub_block_scale_min(sub: &[f32], qmax: f32) -> (f32, f32) {
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &v in sub {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let min = (-lo).max(0.0); // ggml keeps minimums non-negative
    let scale = (hi + min) / qmax;
    (scale.max(0.0), min)
}

/// Pack 8 (scale, min) pairs (each in 0..=63) into the 12-byte K-quant array.
fn pack_scales_k4(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            s[j] = ls[j];
            s[j + 4] = lm[j];
        } else {
            s[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
            s[j - 4] |= (ls[j] >> 4) << 6;
            s[j] |= (lm[j] >> 4) << 6;
        }
    }
    s
}

/// Shared body of the Q4_K / Q5_K encoders. Returns per-block
/// (d, dmin, scales[12], quantized values 0..=qmax).
fn encode_k4_block(blk: &[f32], qmax: f32) -> (f32, f32, [u8; 12], [u8; QK_K]) {
    let mut scales = [0f32; 8];
    let mut mins = [0f32; 8];
    for j in 0..8 {
        let (s, m) = sub_block_scale_min(&blk[32 * j..32 * j + 32], qmax);
        scales[j] = s;
        mins[j] = m;
    }
    let max_scale = scales.iter().cloned().fold(0f32, f32::max);
    let max_min = mins.iter().cloned().fold(0f32, f32::max);
    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let inv_m = if dmin != 0.0 { 1.0 / dmin } else { 0.0 };
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        ls[j] = nearest(scales[j] * inv_d).clamp(0, 63) as u8;
        lm[j] = nearest(mins[j] * inv_m).clamp(0, 63) as u8;
    }
    // Re-read the rounded scales through f16 so the encoder and decoder agree.
    let d = f16_to_f32(f32_to_f16(d));
    let dmin = f16_to_f32(f32_to_f16(dmin));
    let mut q = [0u8; QK_K];
    for j in 0..8 {
        let sc = d * ls[j] as f32;
        let m = dmin * lm[j] as f32;
        let inv = if sc != 0.0 { 1.0 / sc } else { 0.0 };
        for l in 0..32 {
            q[32 * j + l] = nearest((blk[32 * j + l] + m) * inv).clamp(0, qmax as i32) as u8;
        }
    }
    (d, dmin, pack_scales_k4(&ls, &lm), q)
}

/// Encode a row (multiple of 256 values) as Q4_K.
pub fn quantize_row_q4_k(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let mut out = Vec::with_capacity(x.len() / QK_K * Q4_K_SIZE);
    for blk in x.as_chunks::<QK_K>().0 {
        let (d, dmin, scales, q) = encode_k4_block(blk, 15.0);
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        for j in 0..4 {
            for l in 0..32 {
                out.push(q[64 * j + l] | (q[64 * j + 32 + l] << 4));
            }
        }
    }
    out
}

/// Encode a row (multiple of 256 values) as Q5_K.
pub fn quantize_row_q5_k(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let mut out = Vec::with_capacity(x.len() / QK_K * Q5_K_SIZE);
    for blk in x.as_chunks::<QK_K>().0 {
        let (d, dmin, scales, q) = encode_k4_block(blk, 31.0);
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        for j in 0..4 {
            let u1 = 1u8 << (2 * j);
            let u2 = 2u8 << (2 * j);
            for l in 0..32 {
                let a = q[64 * j + l];
                let b = q[64 * j + 32 + l];
                qs[32 * j + l] = (a & 0xF) | ((b & 0xF) << 4);
                if a & 16 != 0 {
                    qh[l] |= u1;
                }
                if b & 16 != 0 {
                    qh[l] |= u2;
                }
            }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

/// Encode a row (multiple of 256 values) as Q6_K.
pub fn quantize_row_q6_k(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK_K, 0);
    let mut out = Vec::with_capacity(x.len() / QK_K * Q6_K_SIZE);
    for blk in x.as_chunks::<QK_K>().0 {
        // One symmetric scale per 16 values, q in -32..=31.
        let mut scales = [0f32; 16];
        for j in 0..16 {
            let sub = &blk[16 * j..16 * j + 16];
            let mut amax = 0f32;
            let mut max = 0f32;
            for &v in sub {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            scales[j] = max / -32.0;
        }
        let mut max_scale = 0f32;
        let mut max_abs = 0f32;
        for &s in &scales {
            if s.abs() > max_abs {
                max_abs = s.abs();
                max_scale = s;
            }
        }
        let iscale = if max_scale != 0.0 { -128.0 / max_scale } else { 0.0 };
        let d = f16_to_f32(f32_to_f16(1.0 / iscale));
        let mut ls = [0i8; 16];
        for j in 0..16 {
            ls[j] = nearest(iscale * scales[j]).clamp(-128, 127) as i8;
        }
        let mut q = [0u8; QK_K];
        for j in 0..16 {
            let sc = d * ls[j] as f32;
            let inv = if sc != 0.0 { 1.0 / sc } else { 0.0 };
            for l in 0..16 {
                let v = nearest(blk[16 * j + l] * inv).clamp(-32, 31);
                q[16 * j + l] = (v + 32) as u8;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for n in 0..2 {
            for l in 0..32 {
                let q1 = q[128 * n + l];
                let q2 = q[128 * n + l + 32];
                let q3 = q[128 * n + l + 64];
                let q4 = q[128 * n + l + 96];
                ql[64 * n + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[64 * n + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
                qh[32 * n + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&ls.map(|v| v as u8));
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

pub fn quantize_row_f16(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect()
}

pub fn quantize_row_bf16(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|&v| f32_to_bf16(v).to_le_bytes()).collect()
}

pub fn quantize_row_f32(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

/// Encode a row with the given kind.
pub fn quantize_row(kind: Kind, x: &[f32]) -> Vec<u8> {
    match kind {
        Kind::F32 => quantize_row_f32(x),
        Kind::F16 => quantize_row_f16(x),
        Kind::BF16 => quantize_row_bf16(x),
        Kind::Q4_0 => quantize_row_q4_0(x),
        Kind::Q8_0 => quantize_row_q8_0(x),
        Kind::Q4_K => quantize_row_q4_k(x),
        Kind::Q5_K => quantize_row_q5_k(x),
        Kind::Q6_K => quantize_row_q6_k(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn check(kind: Kind, tol: f32) {
        let x = data(512, kind as u32 + 1);
        let enc = quantize_row(kind, &x);
        assert_eq!(enc.len(), kind.row_bytes(512));
        let mut dec = vec![0f32; 512];
        dequant_row(kind, &enc, &mut dec);
        let err: f32 = x.iter().zip(&dec).map(|(a, b)| (a - b).abs()).sum::<f32>() / 512.0;
        assert!(err < tol, "{kind:?}: mean abs error {err}");
    }

    #[test]
    fn roundtrips() {
        check(Kind::F32, 1e-7);
        check(Kind::F16, 1e-3);
        check(Kind::BF16, 1e-2);
        check(Kind::Q8_0, 1e-2);
        check(Kind::Q4_0, 0.1);
        check(Kind::Q4_K, 0.08);
        check(Kind::Q5_K, 0.04);
        check(Kind::Q6_K, 0.02);
    }
}
