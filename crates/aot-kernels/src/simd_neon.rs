//! ARM64 NEON dot-product kernels.
//!
//! Two variants are generated from one macro body:
//! * `dotprod`: uses `sdot` (`vdotq_s32`), 4 x (i8 * i8) per lane per op;
//! * `plain`:   base NEON (`smull` + pairwise add) for CPUs without the
//!   dot-product extension.
//!
//! All loads are unaligned (`vld1q_*` has no alignment requirement) so the
//! kernels work directly on memory-mapped GGUF data.

use super::super::quant::*;
use std::arch::aarch64::*;

#[inline]
fn f16_at(p: *const u8) -> f32 {
    // SAFETY: caller passes a pointer into a block with >= 2 readable bytes.
    f16_to_f32(u16::from_le_bytes(unsafe { [*p, *p.add(1)] }))
}

/// Signed 8-bit dot product without the `dotprod` extension.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn mull_dot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let hi = vmull_high_s8(a, b);
    vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)))
}

/// Signed 8-bit dot product with the `dotprod` extension (`sdot`).
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    vdotq_s32(acc, a, b)
}

/// Convert 4 packed half floats to f32 (`fcvtl` is part of base NEON).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn f16x4_to_f32(p: *const u8) -> float32x4_t {
    let h: uint16x4_t = vld1_u16(p as *const u16);
    let r: float32x4_t;
    std::arch::asm!("fcvtl {o:v}.4s, {i:v}.4h", i = in(vreg) h, o = out(vreg) r, options(pure, nomem, nostack));
    r
}

macro_rules! neon_kernels {
    ($modname:ident, $feat:literal, $vdot:ident) => {
        pub mod $modname {
            use super::*;

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn vdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
                $vdot(acc, a, b)
            }

            /// One Q4_0 block (18 bytes) against one Q8_0 block.
            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn q4_0_block(b: *const u8, y: &BlockQ8_0, m4: uint8x16_t, s8: int8x16_t) -> (int32x4_t, f32) {
                let d = f16_at(b) * y.d;
                let q = vld1q_u8(b.add(2));
                let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4)), s8);
                let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), s8);
                let yp = y.qs.as_ptr();
                let p = vdot(vdot(vdupq_n_s32(0), lo, vld1q_s8(yp)), hi, vld1q_s8(yp.add(16)));
                (p, d)
            }

            #[target_feature(enable = $feat)]
            pub unsafe fn dot_q4_0_q8_0(w: &[u8], x: &[BlockQ8_0]) -> f32 {
                let nb = x.len();
                debug_assert!(w.len() >= nb * Q4_0_SIZE);
                let m4 = vdupq_n_u8(0x0F);
                let s8 = vdupq_n_s8(8);
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let wp = w.as_ptr();
                let mut i = 0;
                while i + 2 <= nb {
                    let (p0, d0) = q4_0_block(wp.add(i * Q4_0_SIZE), &x[i], m4, s8);
                    let (p1, d1) = q4_0_block(wp.add((i + 1) * Q4_0_SIZE), &x[i + 1], m4, s8);
                    acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), d0);
                    acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), d1);
                    i += 2;
                }
                if i < nb {
                    let (p0, d0) = q4_0_block(wp.add(i * Q4_0_SIZE), &x[i], m4, s8);
                    acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), d0);
                }
                vaddvq_f32(vaddq_f32(acc0, acc1))
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn q8_0_block(b: *const u8, y: &BlockQ8_0) -> (int32x4_t, f32) {
                let d = f16_at(b) * y.d;
                let q0 = vld1q_s8(b.add(2) as *const i8);
                let q1 = vld1q_s8(b.add(18) as *const i8);
                let yp = y.qs.as_ptr();
                let p = vdot(vdot(vdupq_n_s32(0), q0, vld1q_s8(yp)), q1, vld1q_s8(yp.add(16)));
                (p, d)
            }

            #[target_feature(enable = $feat)]
            pub unsafe fn dot_q8_0_q8_0(w: &[u8], x: &[BlockQ8_0]) -> f32 {
                let nb = x.len();
                debug_assert!(w.len() >= nb * Q8_0_SIZE);
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let wp = w.as_ptr();
                let mut i = 0;
                while i + 2 <= nb {
                    let (p0, d0) = q8_0_block(wp.add(i * Q8_0_SIZE), &x[i]);
                    let (p1, d1) = q8_0_block(wp.add((i + 1) * Q8_0_SIZE), &x[i + 1]);
                    acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), d0);
                    acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), d1);
                    i += 2;
                }
                if i < nb {
                    let (p0, d0) = q8_0_block(wp.add(i * Q8_0_SIZE), &x[i]);
                    acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), d0);
                }
                vaddvq_f32(vaddq_f32(acc0, acc1))
            }

            /// Decode the eight 6-bit (scale, min) pairs of a K-quant block and
            /// accumulate `sum(min_j * bsum_j)` for the minimum correction.
            #[inline]
            unsafe fn k4_scales(scales: *const u8, y: &BlockQ8_K) -> ([i32; 8], i32) {
                let s = std::slice::from_raw_parts(scales, 12);
                let mut sc = [0i32; 8];
                let mut summ = 0i32;
                for j in 0..8 {
                    let (a, m) = scale_min_k4(j, s);
                    sc[j] = a as i32;
                    summ += m as i32 * (y.bsums[2 * j] as i32 + y.bsums[2 * j + 1] as i32);
                }
                (sc, summ)
            }

            #[target_feature(enable = $feat)]
            pub unsafe fn dot_q4_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
                debug_assert!(w.len() >= x.len() * Q4_K_SIZE);
                let m4 = vdupq_n_u8(0x0F);
                let zero = vdupq_n_s32(0);
                let mut sumf = 0f32;
                for (i, y) in x.iter().enumerate() {
                    let b = w.as_ptr().add(i * Q4_K_SIZE);
                    let d = f16_at(b) * y.d;
                    let dmin = f16_at(b.add(2)) * y.d;
                    let (sc, summ) = k4_scales(b.add(4), y);
                    let qs = b.add(16);
                    let yp = y.qs.as_ptr();
                    let mut sumi = zero;
                    for j in 0..4 {
                        let q0 = vld1q_u8(qs.add(32 * j));
                        let q1 = vld1q_u8(qs.add(32 * j + 16));
                        let l0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
                        let l1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
                        let h0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                        let h1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
                        let yj = yp.add(64 * j);
                        let p_lo = vdot(vdot(zero, l0, vld1q_s8(yj)), l1, vld1q_s8(yj.add(16)));
                        let p_hi = vdot(vdot(zero, h0, vld1q_s8(yj.add(32))), h1, vld1q_s8(yj.add(48)));
                        sumi = vmlaq_n_s32(sumi, p_lo, sc[2 * j]);
                        sumi = vmlaq_n_s32(sumi, p_hi, sc[2 * j + 1]);
                    }
                    sumf += d * vaddvq_s32(sumi) as f32 - dmin * summ as f32;
                }
                sumf
            }

            #[target_feature(enable = $feat)]
            pub unsafe fn dot_q5_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
                debug_assert!(w.len() >= x.len() * Q5_K_SIZE);
                let m4 = vdupq_n_u8(0x0F);
                let m16 = vdupq_n_u8(16);
                let zero = vdupq_n_s32(0);
                let mut sumf = 0f32;
                for (i, y) in x.iter().enumerate() {
                    let b = w.as_ptr().add(i * Q5_K_SIZE);
                    let d = f16_at(b) * y.d;
                    let dmin = f16_at(b.add(2)) * y.d;
                    let (sc, summ) = k4_scales(b.add(4), y);
                    let qh0 = vld1q_u8(b.add(16));
                    let qh1 = vld1q_u8(b.add(32));
                    let qs = b.add(48);
                    let yp = y.qs.as_ptr();
                    let mut sumi = zero;
                    for j in 0..4 {
                        let q0 = vld1q_u8(qs.add(32 * j));
                        let q1 = vld1q_u8(qs.add(32 * j + 16));
                        let u1 = vdupq_n_u8(1 << (2 * j));
                        let u2 = vdupq_n_u8(2 << (2 * j));
                        // vtstq yields 0xFF where the high bit is set; AND with 16 adds it in.
                        let l0 = vaddq_u8(vandq_u8(q0, m4), vandq_u8(vtstq_u8(qh0, u1), m16));
                        let l1 = vaddq_u8(vandq_u8(q1, m4), vandq_u8(vtstq_u8(qh1, u1), m16));
                        let h0 = vaddq_u8(vshrq_n_u8::<4>(q0), vandq_u8(vtstq_u8(qh0, u2), m16));
                        let h1 = vaddq_u8(vshrq_n_u8::<4>(q1), vandq_u8(vtstq_u8(qh1, u2), m16));
                        let yj = yp.add(64 * j);
                        let p_lo = vdot(
                            vdot(zero, vreinterpretq_s8_u8(l0), vld1q_s8(yj)),
                            vreinterpretq_s8_u8(l1),
                            vld1q_s8(yj.add(16)),
                        );
                        let p_hi = vdot(
                            vdot(zero, vreinterpretq_s8_u8(h0), vld1q_s8(yj.add(32))),
                            vreinterpretq_s8_u8(h1),
                            vld1q_s8(yj.add(48)),
                        );
                        sumi = vmlaq_n_s32(sumi, p_lo, sc[2 * j]);
                        sumi = vmlaq_n_s32(sumi, p_hi, sc[2 * j + 1]);
                    }
                    sumf += d * vaddvq_s32(sumi) as f32 - dmin * summ as f32;
                }
                sumf
            }

            #[target_feature(enable = $feat)]
            pub unsafe fn dot_q6_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
                debug_assert!(w.len() >= x.len() * Q6_K_SIZE);
                let m4 = vdupq_n_u8(0x0F);
                let m3 = vdupq_n_u8(0x03);
                let s32 = vdupq_n_s8(32);
                let zero = vdupq_n_s32(0);
                let mut sumf = 0f32;
                for (i, y) in x.iter().enumerate() {
                    let b = w.as_ptr().add(i * Q6_K_SIZE);
                    let d = f16_at(b.add(208)) * y.d;
                    let mut sumi = zero;
                    for n in 0..2 {
                        let ql = b.add(64 * n);
                        let qh = b.add(128 + 32 * n);
                        let sc = b.add(192 + 8 * n) as *const i8;
                        let yp = y.qs.as_ptr().add(128 * n);
                        for half in 0..2 {
                            let l = 16 * half;
                            let ql0 = vld1q_u8(ql.add(l));
                            let ql32 = vld1q_u8(ql.add(32 + l));
                            let qhv = vld1q_u8(qh.add(l));
                            let q1 = vorrq_u8(vandq_u8(ql0, m4), vshlq_n_u8::<4>(vandq_u8(qhv, m3)));
                            let q2 = vorrq_u8(vandq_u8(ql32, m4), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qhv), m3)));
                            let q3 = vorrq_u8(vshrq_n_u8::<4>(ql0), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qhv), m3)));
                            let q4 = vorrq_u8(vshrq_n_u8::<4>(ql32), vshlq_n_u8::<4>(vshrq_n_u8::<6>(qhv)));
                            let q1 = vsubq_s8(vreinterpretq_s8_u8(q1), s32);
                            let q2 = vsubq_s8(vreinterpretq_s8_u8(q2), s32);
                            let q3 = vsubq_s8(vreinterpretq_s8_u8(q3), s32);
                            let q4 = vsubq_s8(vreinterpretq_s8_u8(q4), s32);
                            sumi = vmlaq_n_s32(sumi, vdot(zero, q1, vld1q_s8(yp.add(l))), *sc.add(half) as i32);
                            sumi = vmlaq_n_s32(sumi, vdot(zero, q2, vld1q_s8(yp.add(32 + l))), *sc.add(half + 2) as i32);
                            sumi = vmlaq_n_s32(sumi, vdot(zero, q3, vld1q_s8(yp.add(64 + l))), *sc.add(half + 4) as i32);
                            sumi = vmlaq_n_s32(sumi, vdot(zero, q4, vld1q_s8(yp.add(96 + l))), *sc.add(half + 6) as i32);
                        }
                    }
                    sumf += d * vaddvq_s32(sumi) as f32;
                }
                sumf
            }
        }
    };
}

neon_kernels!(dotprod, "neon,dotprod", sdot);
neon_kernels!(plain, "neon", mull_dot);

/// f16 weights against f32 input.
#[target_feature(enable = "neon")]
pub unsafe fn dot_f16_f32(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(w.len() >= 2 * n);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        acc0 = vfmaq_f32(acc0, f16x4_to_f32(wp.add(2 * i)), vld1q_f32(xp.add(i)));
        acc1 = vfmaq_f32(acc1, f16x4_to_f32(wp.add(2 * i + 8)), vld1q_f32(xp.add(i + 4)));
        i += 8;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        sum += f16_at(wp.add(2 * i)) * x[i];
        i += 1;
    }
    sum
}

/// bf16 weights against f32 input (bf16 -> f32 is a 16-bit left shift).
#[target_feature(enable = "neon")]
pub unsafe fn dot_bf16_f32(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(w.len() >= 2 * n);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let wp = w.as_ptr() as *const u16;
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        let a = vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(wp.add(i))));
        let b = vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(wp.add(i + 4))));
        acc0 = vfmaq_f32(acc0, a, vld1q_f32(xp.add(i)));
        acc1 = vfmaq_f32(acc1, b, vld1q_f32(xp.add(i + 4)));
        i += 8;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        sum += bf16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]])) * x[i];
        i += 1;
    }
    sum
}

/// f32 weights against f32 input.
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_f32(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(w.len() >= 4 * n);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let wp = w.as_ptr() as *const f32;
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 8 <= n {
        acc0 = vfmaq_f32(acc0, vld1q_f32(wp.add(i)), vld1q_f32(xp.add(i)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(wp.add(i + 4)), vld1q_f32(xp.add(i + 4)));
        i += 8;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
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
    dp_q4_0 = dotprod::dot_q4_0_q8_0 : BlockQ8_0;
    dp_q8_0 = dotprod::dot_q8_0_q8_0 : BlockQ8_0;
    dp_q4_k = dotprod::dot_q4_k_q8_k : BlockQ8_K;
    dp_q5_k = dotprod::dot_q5_k_q8_k : BlockQ8_K;
    dp_q6_k = dotprod::dot_q6_k_q8_k : BlockQ8_K;
    pl_q4_0 = plain::dot_q4_0_q8_0 : BlockQ8_0;
    pl_q8_0 = plain::dot_q8_0_q8_0 : BlockQ8_0;
    pl_q4_k = plain::dot_q4_k_q8_k : BlockQ8_K;
    pl_q5_k = plain::dot_q5_k_q8_k : BlockQ8_K;
    pl_q6_k = plain::dot_q6_k_q8_k : BlockQ8_K;
    f16 = dot_f16_f32 : f32;
    bf16 = dot_bf16_f32 : f32;
    f32w = dot_f32_f32 : f32;
}

pub const DOTPROD_KERNELS: super::Kernels = super::Kernels {
    name: "neon+dotprod",
    q4_0_q8_0: dp_q4_0,
    q8_0_q8_0: dp_q8_0,
    q4_k_q8_k: dp_q4_k,
    q5_k_q8_k: dp_q5_k,
    q6_k_q8_k: dp_q6_k,
    f16_f32: f16,
    bf16_f32: bf16,
    f32_f32: f32w,
};

pub const PLAIN_KERNELS: super::Kernels = super::Kernels {
    name: "neon",
    q4_0_q8_0: pl_q4_0,
    q8_0_q8_0: pl_q8_0,
    q4_k_q8_k: pl_q4_k,
    q5_k_q8_k: pl_q5_k,
    q6_k_q8_k: pl_q6_k,
    f16_f32: f16,
    bf16_f32: bf16,
    f32_f32: f32w,
};
