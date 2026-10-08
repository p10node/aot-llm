//! Quantized matrix-vector products.
//!
//! Every weight matrix is stored row-major in its quantized form; the input
//! vector is pre-quantized once per matmul into [`BlockQ8_0`] (for Q4_0/Q8_0
//! weights) or [`BlockQ8_K`] (for K-quants), and each output element is one
//! integer dot product scaled back to `f32`.
//!
//! Three implementations exist for every dot product:
//! * `scalar`  - portable reference, also used by tests;
//! * `neon`    - ARM64 NEON, with a variant that uses the `dotprod`
//!   extension (`sdot`) when the CPU supports it (`simd_neon.rs`);
//! * `avx2`    - x86_64 AVX2 + FMA (`simd_avx2.rs`).
//!
//! The best available implementation is selected once at runtime
//! ([`kernels`]), so the same binary runs on any CPU of its architecture.

use super::pool::{split, Pool, SendPtr};
use super::quant::*;
use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
#[path = "simd_neon.rs"]
pub mod neon;

#[cfg(target_arch = "x86_64")]
#[path = "simd_avx2.rs"]
pub mod avx2;

/// Portable reference implementations.
pub mod scalar {
    use super::super::quant::*;

    pub fn dot_q4_0_q8_0(w: &[u8], x: &[BlockQ8_0]) -> f32 {
        let mut sum = 0f32;
        for (i, xb) in x.iter().enumerate() {
            let b = &w[i * Q4_0_SIZE..(i + 1) * Q4_0_SIZE];
            let d = read_f16(b, 0);
            let mut s = 0i32;
            for j in 0..16 {
                let q = b[2 + j];
                s += ((q & 0x0F) as i32 - 8) * xb.qs[j] as i32;
                s += ((q >> 4) as i32 - 8) * xb.qs[j + 16] as i32;
            }
            sum += s as f32 * d * xb.d;
        }
        sum
    }

    pub fn dot_q8_0_q8_0(w: &[u8], x: &[BlockQ8_0]) -> f32 {
        let mut sum = 0f32;
        for (i, xb) in x.iter().enumerate() {
            let b = &w[i * Q8_0_SIZE..(i + 1) * Q8_0_SIZE];
            let d = read_f16(b, 0);
            let mut s = 0i32;
            for j in 0..QK8_0 {
                s += (b[2 + j] as i8) as i32 * xb.qs[j] as i32;
            }
            sum += s as f32 * d * xb.d;
        }
        sum
    }

    pub fn dot_q4_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
        let mut sum = 0f32;
        for (i, xb) in x.iter().enumerate() {
            let b = &w[i * Q4_K_SIZE..(i + 1) * Q4_K_SIZE];
            let d = read_f16(b, 0) * xb.d;
            let dmin = read_f16(b, 2) * xb.d;
            let scales = &b[4..16];
            let qs = &b[16..144];
            let mut sumi = 0i32;
            let mut summ = 0i32;
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, scales);
                summ += m as i32 * (xb.bsums[2 * j] as i32 + xb.bsums[2 * j + 1] as i32);
                let c = j / 2;
                let mut s = 0i32;
                for l in 0..32 {
                    let q = qs[32 * c + l];
                    let v = if j % 2 == 0 { q & 0x0F } else { q >> 4 };
                    s += v as i32 * xb.qs[32 * j + l] as i32;
                }
                sumi += sc as i32 * s;
            }
            sum += d * sumi as f32 - dmin * summ as f32;
        }
        sum
    }

    pub fn dot_q5_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
        let mut sum = 0f32;
        for (i, xb) in x.iter().enumerate() {
            let b = &w[i * Q5_K_SIZE..(i + 1) * Q5_K_SIZE];
            let d = read_f16(b, 0) * xb.d;
            let dmin = read_f16(b, 2) * xb.d;
            let scales = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            let mut sumi = 0i32;
            let mut summ = 0i32;
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, scales);
                summ += m as i32 * (xb.bsums[2 * j] as i32 + xb.bsums[2 * j + 1] as i32);
                let c = j / 2;
                let hbit = 1u8 << j;
                let mut s = 0i32;
                for l in 0..32 {
                    let q = qs[32 * c + l];
                    let mut v = if j % 2 == 0 { q & 0x0F } else { q >> 4 };
                    if qh[l] & hbit != 0 {
                        v += 16;
                    }
                    s += v as i32 * xb.qs[32 * j + l] as i32;
                }
                sumi += sc as i32 * s;
            }
            sum += d * sumi as f32 - dmin * summ as f32;
        }
        sum
    }

    pub fn dot_q6_k_q8_k(w: &[u8], x: &[BlockQ8_K]) -> f32 {
        let mut sum = 0f32;
        for (i, xb) in x.iter().enumerate() {
            let b = &w[i * Q6_K_SIZE..(i + 1) * Q6_K_SIZE];
            let d = read_f16(b, 208) * xb.d;
            let mut sumi = 0i32;
            for n in 0..2 {
                let ql = &b[64 * n..64 * n + 64];
                let qh = &b[128 + 32 * n..128 + 32 * n + 32];
                let sc = &b[192 + 8 * n..192 + 8 * n + 8];
                let y = &xb.qs[128 * n..128 * n + 128];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    sumi += (sc[is] as i8) as i32 * q1 * y[l] as i32;
                    sumi += (sc[is + 2] as i8) as i32 * q2 * y[l + 32] as i32;
                    sumi += (sc[is + 4] as i8) as i32 * q3 * y[l + 64] as i32;
                    sumi += (sc[is + 6] as i8) as i32 * q4 * y[l + 96] as i32;
                }
            }
            sum += d * sumi as f32;
        }
        sum
    }

    pub fn dot_f16_f32(w: &[u8], x: &[f32]) -> f32 {
        let mut sum = 0f32;
        for (i, &xi) in x.iter().enumerate() {
            sum += read_f16(w, 2 * i) * xi;
        }
        sum
    }

    pub fn dot_bf16_f32(w: &[u8], x: &[f32]) -> f32 {
        let mut sum = 0f32;
        for (i, &xi) in x.iter().enumerate() {
            sum += bf16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]])) * xi;
        }
        sum
    }

    pub fn dot_f32_f32(w: &[u8], x: &[f32]) -> f32 {
        let mut sum = 0f32;
        for (i, &xi) in x.iter().enumerate() {
            sum += read_f32(w, 4 * i) * xi;
        }
        sum
    }
}

/// Function table of the dot products selected for this CPU.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub name: &'static str,
    pub q4_0_q8_0: fn(&[u8], &[BlockQ8_0]) -> f32,
    pub q8_0_q8_0: fn(&[u8], &[BlockQ8_0]) -> f32,
    pub q4_k_q8_k: fn(&[u8], &[BlockQ8_K]) -> f32,
    pub q5_k_q8_k: fn(&[u8], &[BlockQ8_K]) -> f32,
    pub q6_k_q8_k: fn(&[u8], &[BlockQ8_K]) -> f32,
    pub f16_f32: fn(&[u8], &[f32]) -> f32,
    pub bf16_f32: fn(&[u8], &[f32]) -> f32,
    pub f32_f32: fn(&[u8], &[f32]) -> f32,
}

pub const SCALAR_KERNELS: Kernels = Kernels {
    name: "scalar",
    q4_0_q8_0: scalar::dot_q4_0_q8_0,
    q8_0_q8_0: scalar::dot_q8_0_q8_0,
    q4_k_q8_k: scalar::dot_q4_k_q8_k,
    q5_k_q8_k: scalar::dot_q5_k_q8_k,
    q6_k_q8_k: scalar::dot_q6_k_q8_k,
    f16_f32: scalar::dot_f16_f32,
    bf16_f32: scalar::dot_bf16_f32,
    f32_f32: scalar::dot_f32_f32,
};

/// Detect CPU features and pick the fastest kernel set.
pub fn detect_kernels() -> Kernels {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return neon::DOTPROD_KERNELS;
        }
        return neon::PLAIN_KERNELS;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return if std::is_x86_feature_detected!("f16c") {
                avx2::KERNELS_F16C
            } else {
                avx2::KERNELS
            };
        }
    }
    #[allow(unreachable_code)]
    SCALAR_KERNELS
}

/// The kernel set for this process (detected on first use).
pub fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(detect_kernels)
}

/// Force a kernel set (e.g. `--kernels scalar` for debugging). Must be called
/// before the first matmul to take effect.
pub fn set_kernels(k: Kernels) -> bool {
    OVERRIDE.set(k).is_ok()
}

static OVERRIDE: OnceLock<Kernels> = OnceLock::new();

/// Select kernels by name: `auto`, `scalar`, `neon`, `neon-dotprod`, `avx2`.
pub fn kernels_by_name(name: &str) -> Option<Kernels> {
    match name {
        "auto" => Some(detect_kernels()),
        "scalar" => Some(SCALAR_KERNELS),
        #[cfg(target_arch = "aarch64")]
        "neon" => Some(neon::PLAIN_KERNELS),
        #[cfg(target_arch = "aarch64")]
        "neon-dotprod" => Some(neon::DOTPROD_KERNELS),
        #[cfg(target_arch = "x86_64")]
        "avx2" => Some(avx2::KERNELS),
        _ => None,
    }
}

/// The kernel set in use (an override if one was set, else the detected one).
#[inline]
pub fn active_kernels() -> &'static Kernels {
    OVERRIDE.get().unwrap_or_else(kernels)
}

#[inline]
fn active() -> &'static Kernels {
    active_kernels()
}

/// Parallel loop over the rows of a quantized matrix.
#[inline]
fn matvec_rows<T: Sync>(
    pool: &Pool,
    out: &mut [f32],
    w: &[u8],
    x: &[T],
    rows: usize,
    row_bytes: usize,
    dot: fn(&[u8], &[T]) -> f32,
) {
    assert!(out.len() >= rows, "output too small: {} < {rows}", out.len());
    assert!(w.len() >= rows * row_bytes, "weight slice too small: {} < {}", w.len(), rows * row_bytes);
    let o = SendPtr(out.as_mut_ptr());
    pool.run(&|tid, nth| {
        for r in split(rows, tid, nth) {
            let v = dot(&w[r * row_bytes..(r + 1) * row_bytes], x);
            // SAFETY: rows are partitioned disjointly across threads.
            unsafe { o.write(r, v) };
        }
    });
}

/// `out[r] = sum_c W[r][c] * x[c]` for Q4_0 weights, `x` pre-quantized to Q8_0.
pub fn matvec_q4_0(pool: &Pool, out: &mut [f32], w: &[u8], x: &[BlockQ8_0], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols / QK8_0);
    matvec_rows(pool, out, w, x, rows, Kind::Q4_0.row_bytes(cols), active().q4_0_q8_0);
}

pub fn matvec_q8_0(pool: &Pool, out: &mut [f32], w: &[u8], x: &[BlockQ8_0], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols / QK8_0);
    matvec_rows(pool, out, w, x, rows, Kind::Q8_0.row_bytes(cols), active().q8_0_q8_0);
}

pub fn matvec_q4_k(pool: &Pool, out: &mut [f32], w: &[u8], x: &[BlockQ8_K], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols / QK_K);
    matvec_rows(pool, out, w, x, rows, Kind::Q4_K.row_bytes(cols), active().q4_k_q8_k);
}

pub fn matvec_q5_k(pool: &Pool, out: &mut [f32], w: &[u8], x: &[BlockQ8_K], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols / QK_K);
    matvec_rows(pool, out, w, x, rows, Kind::Q5_K.row_bytes(cols), active().q5_k_q8_k);
}

pub fn matvec_q6_k(pool: &Pool, out: &mut [f32], w: &[u8], x: &[BlockQ8_K], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols / QK_K);
    matvec_rows(pool, out, w, x, rows, Kind::Q6_K.row_bytes(cols), active().q6_k_q8_k);
}

pub fn matvec_f16(pool: &Pool, out: &mut [f32], w: &[u8], x: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols);
    matvec_rows(pool, out, w, x, rows, Kind::F16.row_bytes(cols), active().f16_f32);
}

pub fn matvec_bf16(pool: &Pool, out: &mut [f32], w: &[u8], x: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols);
    matvec_rows(pool, out, w, x, rows, Kind::BF16.row_bytes(cols), active().bf16_f32);
}

pub fn matvec_f32(pool: &Pool, out: &mut [f32], w: &[u8], x: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), cols);
    matvec_rows(pool, out, w, x, rows, Kind::F32.row_bytes(cols), active().f32_f32);
}

/// Two matrix-vector products that share the same input, issued as one
/// parallel region (used for the gate/up projections of the feed-forward
/// block). Both matrices must have the same kind and shape.
#[allow(clippy::too_many_arguments)]
pub fn matvec2_q8_k_input(
    pool: &Pool,
    out_a: &mut [f32],
    out_b: &mut [f32],
    w_a: &[u8],
    w_b: &[u8],
    x: &[BlockQ8_K],
    rows: usize,
    row_bytes: usize,
    dot: fn(&[u8], &[BlockQ8_K]) -> f32,
) {
    assert!(out_a.len() >= rows && out_b.len() >= rows);
    assert!(w_a.len() >= rows * row_bytes && w_b.len() >= rows * row_bytes);
    let oa = SendPtr(out_a.as_mut_ptr());
    let ob = SendPtr(out_b.as_mut_ptr());
    pool.run(&|tid, nth| {
        for r in split(rows, tid, nth) {
            // SAFETY: disjoint row ranges per thread.
            unsafe {
                oa.write(r, dot(&w_a[r * row_bytes..(r + 1) * row_bytes], x));
                ob.write(r, dot(&w_b[r * row_bytes..(r + 1) * row_bytes], x));
            }
        }
    });
}

/// Same as [`matvec2_q8_k_input`] for Q8_0-quantized inputs.
#[allow(clippy::too_many_arguments)]
pub fn matvec2_q8_0_input(
    pool: &Pool,
    out_a: &mut [f32],
    out_b: &mut [f32],
    w_a: &[u8],
    w_b: &[u8],
    x: &[BlockQ8_0],
    rows: usize,
    row_bytes: usize,
    dot: fn(&[u8], &[BlockQ8_0]) -> f32,
) {
    assert!(out_a.len() >= rows && out_b.len() >= rows);
    assert!(w_a.len() >= rows * row_bytes && w_b.len() >= rows * row_bytes);
    let oa = SendPtr(out_a.as_mut_ptr());
    let ob = SendPtr(out_b.as_mut_ptr());
    pool.run(&|tid, nth| {
        for r in split(rows, tid, nth) {
            // SAFETY: disjoint row ranges per thread.
            unsafe {
                oa.write(r, dot(&w_a[r * row_bytes..(r + 1) * row_bytes], x));
                ob.write(r, dot(&w_b[r * row_bytes..(r + 1) * row_bytes], x));
            }
        }
    });
}

/// Dot-product function for `kind` with a Q8_K input.
pub fn dot_fn_q8_k(kind: Kind) -> fn(&[u8], &[BlockQ8_K]) -> f32 {
    let k = active();
    match kind {
        Kind::Q4_K => k.q4_k_q8_k,
        Kind::Q5_K => k.q5_k_q8_k,
        Kind::Q6_K => k.q6_k_q8_k,
        _ => panic!("{} is not a K-quant", kind.name()),
    }
}

/// Dot-product function for `kind` with a Q8_0 input.
pub fn dot_fn_q8_0(kind: Kind) -> fn(&[u8], &[BlockQ8_0]) -> f32 {
    let k = active();
    match kind {
        Kind::Q4_0 => k.q4_0_q8_0,
        Kind::Q8_0 => k.q8_0_q8_0,
        _ => panic!("{} is not a Q8_0-input quant", kind.name()),
    }
}
