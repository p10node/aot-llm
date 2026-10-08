//! x86_64 AVX2 + FMA dot-product kernels.
//!
//! Integer dot products use the `vpmaddubsw` (u8 x i8 -> i16 pairs) +
//! `vpmaddwd` (i16 -> i32) sequence. For signed x signed products (Q4_0,
//! Q8_0) the weight sign is moved onto the activation with `vpsignb`, which
//! is the standard trick from ggml. Saturation bounds are commented at each
//! use. All loads are unaligned.

use super::super::quant::*;
use std::arch::x86_64::*;

#[inline]
fn f16_at(p: *const u8) -> f32 {
    // SAFETY: caller passes a pointer into a block with >= 2 readable bytes.
    f16_to_f32(u16::from_le_bytes(unsafe { [*p, *p.add(1)] }))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
    _mm_cvtsi128_si32(s)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_f32(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
    _mm_cvtss_f32(s)
}

/// Sum of 32 signed i8 products as 8 x i32. `|a| <= 127`, `|b| <= 127`:
/// each i16 pair sum is at most 2 * 127 * 127 = 32258 < 32767, no saturation.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_i8_pairs(a: __m256i, b: __m256i) -> __m256i {
    let ax = _mm256_sign_epi8(a, a); // |a|
    let sy = _mm256_sign_epi8(b, a); // b * sign(a)
    let dot = _mm256_maddubs_epi16(ax, sy);
    _mm256_madd_epi16(dot, _mm256_set1_epi16(1))
}

/// Expand 16 bytes of nibbles into 32 bytes: low nibbles in the low 128-bit
/// lane, high nibbles in the high lane (matches the Q4_0 element order).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn nibbles_32(p: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(p as *const __m128i);
    let both = _mm256_set_m128i(_mm_srli_epi16::<4>(tmp), tmp);
    _mm256_and_si256(both, _mm256_set1_epi8(0x0F))
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_q8_0(w: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    debug_assert!(w.len() >= nb * Q4_0_SIZE);
    let mut acc = _mm256_setzero_ps();
    let off = _mm256_set1_epi8(8);
    let wp = w.as_ptr();
    for (i, y) in x.iter().enumerate() {
        let b = wp.add(i * Q4_0_SIZE);
        let d = _mm256_set1_ps(f16_at(b) * y.d);
        let bx = _mm256_sub_epi8(nibbles_32(b.add(2)), off); // -8..7
        let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
        let p = _mm256_cvtepi32_ps(mul_sum_i8_pairs(bx, by));
        acc = _mm256_fmadd_ps(d, p, acc);
    }
    hsum_f32(acc)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_0_q8_0(w: &[u8], x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    debug_assert!(w.len() >= nb * Q8_0_SIZE);
    let mut acc = _mm256_setzero_ps();
    let wp = w.as_ptr();
    for (i, y) in x.iter().enumerate() {
        let b = wp.add(i * Q8_0_SIZE);
        let d = _mm256_set1_ps(f16_at(b) * y.d);
        let bx = _mm256_loadu_si256(b.add(2) as *const __m256i);
        let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
        let p = _mm256_cvtepi32_ps(mul_sum_i8_pairs(bx, by));
        acc = _mm256_fmadd_ps(d, p, acc);
    }
    hsum_f32(acc)
}

/// Decode the eight 6-bit (scale, min) pairs and the minimum correction.
#[inline]
unsafe fn k4_scales(scales: *const u8, y: &BlockQ8_K) -> ([i16; 8], i32) {
    let s = std::slice::from_raw_parts(scales, 12);
    let mut sc = [0i16; 8];
    let mut summ = 0i32;
    for j in 0..8 {
        let (a, m) = scale_min_k4(j, s);
        sc[j] = a as i16;
        summ += m as i32 * (y.bsums[2 * j] as i32 + y.bsums[2 * j + 1] as i32);
    }
    (sc, summ)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
    debug_assert!(w.len() >= x.len() * Q4_K_SIZE);
    let m4 = _mm256_set1_epi8(0x0F);
    let mut sumf = 0f32;
    for (i, y) in x.iter().enumerate() {
        let b = w.as_ptr().add(i * Q4_K_SIZE);
        let d = f16_at(b) * y.d;
        let dmin = f16_at(b.add(2)) * y.d;
        let (sc, summ) = k4_scales(b.add(4), y);
        let qs = b.add(16);
        let yp = y.qs.as_ptr();
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let q = _mm256_loadu_si256(qs.add(32 * j) as *const __m256i);
            let lo = _mm256_and_si256(q, m4); // elements 64j .. 64j+32
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4); // 64j+32 .. 64j+64
            let y0 = _mm256_loadu_si256(yp.add(64 * j) as *const __m256i);
            let y1 = _mm256_loadu_si256(yp.add(64 * j + 32) as *const __m256i);
            // u8 (0..15) x i8: pair sums <= 2 * 15 * 127 = 3810, no saturation.
            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), _mm256_set1_epi16(sc[2 * j]));
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), _mm256_set1_epi16(sc[2 * j + 1]));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
        }
        sumf += d * hsum_i32(sumi) as f32 - dmin * summ as f32;
    }
    sumf
}

/// Move bit `bit` of every byte in `v` to bit 4 (the 5th-bit position of a
/// Q5_K nibble). 16-bit lane shifts are fine because the mask keeps only
/// bit 4, which always originates from the same byte for shifts < 8.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn qh_bit_to_16(v: __m256i, bit: usize) -> __m256i {
    let m = _mm256_set1_epi8(0x10);
    let s = match bit {
        0 => _mm256_slli_epi16::<4>(v),
        1 => _mm256_slli_epi16::<3>(v),
        2 => _mm256_slli_epi16::<2>(v),
        3 => _mm256_slli_epi16::<1>(v),
        4 => v,
        5 => _mm256_srli_epi16::<1>(v),
        6 => _mm256_srli_epi16::<2>(v),
        _ => _mm256_srli_epi16::<3>(v),
    };
    _mm256_and_si256(s, m)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q5_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
    debug_assert!(w.len() >= x.len() * Q5_K_SIZE);
    let m4 = _mm256_set1_epi8(0x0F);
    let mut sumf = 0f32;
    for (i, y) in x.iter().enumerate() {
        let b = w.as_ptr().add(i * Q5_K_SIZE);
        let d = f16_at(b) * y.d;
        let dmin = f16_at(b.add(2)) * y.d;
        let (sc, summ) = k4_scales(b.add(4), y);
        let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
        let qs = b.add(48);
        let yp = y.qs.as_ptr();
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let q = _mm256_loadu_si256(qs.add(32 * j) as *const __m256i);
            let lo = _mm256_or_si256(_mm256_and_si256(q, m4), qh_bit_to_16(qh, 2 * j));
            let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(q), m4), qh_bit_to_16(qh, 2 * j + 1));
            let y0 = _mm256_loadu_si256(yp.add(64 * j) as *const __m256i);
            let y1 = _mm256_loadu_si256(yp.add(64 * j + 32) as *const __m256i);
            // u8 (0..31) x i8: pair sums <= 2 * 31 * 127 = 7874, no saturation.
            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), _mm256_set1_epi16(sc[2 * j]));
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), _mm256_set1_epi16(sc[2 * j + 1]));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
        }
        sumf += d * hsum_i32(sumi) as f32 - dmin * summ as f32;
    }
    sumf
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
    debug_assert!(w.len() >= x.len() * Q6_K_SIZE);
    let m4 = _mm256_set1_epi8(0x0F);
    let m30 = _mm256_set1_epi8(0x30);
    let mut sumf = 0f32;
    for (i, y) in x.iter().enumerate() {
        let b = w.as_ptr().add(i * Q6_K_SIZE);
        let d = f16_at(b.add(208)) * y.d;
        let scales = std::slice::from_raw_parts(b.add(192) as *const i8, 16);
        // Values are used unsigned (0..63); the -32 offset is applied through
        // the activation block sums: sum((q-32)*s*y) = sum(q*s*y) - 32*sum(s*bsum).
        let mut bias = 0i32;
        for k in 0..16 {
            bias += scales[k] as i32 * y.bsums[k] as i32;
        }
        let mut sumi = _mm256_setzero_si256();
        for n in 0..2 {
            let ql = b.add(64 * n);
            let qh = b.add(128 + 32 * n);
            let sc = &scales[8 * n..8 * n + 8];
            let yp = y.qs.as_ptr().add(128 * n);
            let ql0 = _mm256_loadu_si256(ql as *const __m256i);
            let ql32 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            let qhv = _mm256_loadu_si256(qh as *const __m256i);
            let q1 = _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_and_si256(_mm256_slli_epi16::<4>(qhv), m30));
            let q2 = _mm256_or_si256(_mm256_and_si256(ql32, m4), _mm256_and_si256(_mm256_slli_epi16::<2>(qhv), m30));
            let q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(ql0), m4), _mm256_and_si256(qhv, m30));
            let q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(ql32), m4), _mm256_and_si256(_mm256_srli_epi16::<2>(qhv), m30));
            let qv = [q1, q2, q3, q4];
            for (k, q) in qv.iter().enumerate() {
                let yv = _mm256_loadu_si256(yp.add(32 * k) as *const __m256i);
                // u8 (0..63) x i8: pair sums <= 2 * 63 * 127 = 16002, no saturation.
                let p16 = _mm256_maddubs_epi16(*q, yv);
                // i16 lanes 0..7 hold elements 0..16 (scale sc[2k]), lanes 8..15
                // elements 16..32 (scale sc[2k+1]).
                let scv = _mm256_set_m128i(_mm_set1_epi16(sc[2 * k + 1] as i16), _mm_set1_epi16(sc[2 * k] as i16));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, scv));
            }
        }
        sumf += d * (hsum_i32(sumi) - 32 * bias) as f32;
    }
    sumf
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f16_f32(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(w.len() >= 2 * n);
    let mut acc = _mm256_setzero_ps();
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(wp.add(2 * i) as *const __m128i);
        acc = _mm256_fmadd_ps(_mm256_cvtph_ps(h), _mm256_loadu_ps(xp.add(i)), acc);
        i += 8;
    }
    let mut sum = hsum_f32(acc);
    while i < n {
        sum += f16_at(wp.add(2 * i)) * x[i];
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_bf16_f32(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(w.len() >= 2 * n);
    let mut acc = _mm256_setzero_ps();
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(wp.add(2 * i) as *const __m128i);
        let f = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)));
        acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(xp.add(i)), acc);
        i += 8;
    }
    let mut sum = hsum_f32(acc);
    while i < n {
        sum += bf16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]])) * x[i];
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_f32(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(w.len() >= 4 * n);
    let mut acc = _mm256_setzero_ps();
    let wp = w.as_ptr() as *const f32;
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(i)), _mm256_loadu_ps(xp.add(i)), acc);
        i += 8;
    }
    let mut sum = hsum_f32(acc);
    while i < n {
        sum += read_f32(w, 4 * i) * x[i];
        i += 1;
    }
    sum
}

macro_rules! safe_wrappers {
    ($($name:ident = $path:path : $xt:ty;)*) => {
        $( fn $name(w: &[u8], x: &[$xt]) -> f32 {
            // SAFETY: only reachable through a kernel table whose selection
            // checked the required CPU features at runtime.
            unsafe { $path(w, x) }
        } )*
    };
}

safe_wrappers! {
    s_q4_0 = dot_q4_0_q8_0 : BlockQ8_0;
    s_q8_0 = dot_q8_0_q8_0 : BlockQ8_0;
    s_q4_k = dot_q4_k_q8_k : BlockQ8_K;
    s_q5_k = dot_q5_k_q8_k : BlockQ8_K;
    s_q6_k = dot_q6_k_q8_k : BlockQ8_K;
    s_f16 = dot_f16_f32 : f32;
    s_bf16 = dot_bf16_f32 : f32;
    s_f32 = dot_f32_f32 : f32;
}

/// AVX2 + FMA kernels (f16 weights fall back to the scalar converter).
pub const KERNELS: super::Kernels = super::Kernels {
    name: "avx2",
    q4_0_q8_0: s_q4_0,
    q8_0_q8_0: s_q8_0,
    q4_k_q8_k: s_q4_k,
    q5_k_q8_k: s_q5_k,
    q6_k_q8_k: s_q6_k,
    f16_f32: super::scalar::dot_f16_f32,
    bf16_f32: s_bf16,
    f32_f32: s_f32,
};

/// AVX2 + FMA + F16C kernels.
pub const KERNELS_F16C: super::Kernels = super::Kernels {
    name: "avx2+f16c",
    q4_0_q8_0: s_q4_0,
    q8_0_q8_0: s_q8_0,
    q4_k_q8_k: s_q4_k,
    q5_k_q8_k: s_q5_k,
    q6_k_q8_k: s_q6_k,
    f16_f32: s_f16,
    bf16_f32: s_bf16,
    f32_f32: s_f32,
};
