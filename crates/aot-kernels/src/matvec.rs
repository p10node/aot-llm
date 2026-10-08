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
    /// Optional 1x4 micro-kernels (one row, four activations) used by the
    /// batched products; `None` falls back to four single dots.
    pub q4_0_q8_0_x4: Option<fn(&[u8], [&[BlockQ8_0]; 4]) -> [f32; 4]>,
    pub q8_0_q8_0_x4: Option<fn(&[u8], [&[BlockQ8_0]; 4]) -> [f32; 4]>,
    pub q4_k_q8_k_x4: Option<fn(&[u8], [&[BlockQ8_K]; 4]) -> [f32; 4]>,
    pub q6_k_q8_k_x4: Option<fn(&[u8], [&[BlockQ8_K]; 4]) -> [f32; 4]>,
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
    q4_0_q8_0_x4: None,
    q8_0_q8_0_x4: None,
    q4_k_q8_k_x4: None,
    q6_k_q8_k_x4: None,
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
    pool.run_chunks(rows, row_chunk(rows, pool.n_threads()), &|range| {
        for r in range {
            let v = dot(&w[r * row_bytes..(r + 1) * row_bytes], x);
            // SAFETY: chunks are disjoint and each row is written once.
            unsafe { o.write(r, v) };
        }
    });
}

/// Chunk size for dynamically scheduled row loops: about eight chunks per
/// thread, never fewer than 8 rows.
#[inline]
fn row_chunk(rows: usize, nth: usize) -> usize {
    (rows / (nth * 8)).max(8)
}

/// Three products sharing one input and one kernel (the q/k/v projections),
/// issued as a single parallel region over the concatenated row space.
#[allow(clippy::too_many_arguments)]
fn matvec3_rows<T: Sync>(
    pool: &Pool,
    outs: [&mut [f32]; 3],
    ws: [&[u8]; 3],
    x: &[T],
    rows: [usize; 3],
    row_bytes: usize,
    dot: fn(&[u8], &[T]) -> f32,
) {
    for i in 0..3 {
        assert!(outs[i].len() >= rows[i] && ws[i].len() >= rows[i] * row_bytes, "q/k/v buffers too small");
    }
    let o = [SendPtr(outs[0].as_mut_ptr()), SendPtr(outs[1].as_mut_ptr()), SendPtr(outs[2].as_mut_ptr())];
    let total = rows[0] + rows[1] + rows[2];
    pool.run_chunks(total, row_chunk(total, pool.n_threads()), &|range| {
        for r in range {
            let (t, lr) = if r < rows[0] {
                (0, r)
            } else if r < rows[0] + rows[1] {
                (1, r - rows[0])
            } else {
                (2, r - rows[0] - rows[1])
            };
            let v = dot(&ws[t][lr * row_bytes..(lr + 1) * row_bytes], x);
            // SAFETY: each (tensor, row) is written exactly once.
            unsafe { o[t].write(lr, v) };
        }
    });
}

/// Fused q/k/v projections for K-quant weights of one kind.
#[allow(clippy::too_many_arguments)]
pub fn matvec3_q8_k_input(pool: &Pool, outs: [&mut [f32]; 3], ws: [&[u8]; 3], x: &[BlockQ8_K], rows: [usize; 3], row_bytes: usize, dot: fn(&[u8], &[BlockQ8_K]) -> f32) {
    matvec3_rows(pool, outs, ws, x, rows, row_bytes, dot);
}

/// Fused q/k/v projections for Q4_0 / Q8_0 weights of one kind.
#[allow(clippy::too_many_arguments)]
pub fn matvec3_q8_0_input(pool: &Pool, outs: [&mut [f32]; 3], ws: [&[u8]; 3], x: &[BlockQ8_0], rows: [usize; 3], row_bytes: usize, dot: fn(&[u8], &[BlockQ8_0]) -> f32) {
    matvec3_rows(pool, outs, ws, x, rows, row_bytes, dot);
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
    pool.run_chunks(rows, row_chunk(rows, pool.n_threads()), &|range| {
        for r in range {
            // SAFETY: disjoint chunks; each row written once per output.
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
    pool.run_chunks(rows, row_chunk(rows, pool.n_threads()), &|range| {
        for r in range {
            // SAFETY: disjoint chunks; each row written once per output.
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

// ---------------------------------------------------------------------------
// Batched products (prompt processing)
// ---------------------------------------------------------------------------

/// Rows per L1-resident weight block in the batched products.
const ROW_BLOCK: usize = 8;

/// Batched product: `out[i * rows + r] = sum_c W[r][c] * X[i][c]` for the `m`
/// input vectors stored back to back in `xs` (`nb` blocks each).
///
/// Rows are processed in blocks of [`ROW_BLOCK`] so the block stays in L1
/// while the activations stream past it, four at a time through the 1x4
/// micro-kernel when one exists. Every weight row is thus read from memory
/// once per batch, which turns prompt processing from memory-bound into
/// compute-bound.
#[allow(clippy::too_many_arguments)]
fn matmul_rows<T: Sync>(
    pool: &Pool,
    out: &mut [f32],
    w: &[u8],
    xs: &[T],
    m: usize,
    nb: usize,
    rows: usize,
    row_bytes: usize,
    dot: fn(&[u8], &[T]) -> f32,
    dot4: Option<fn(&[u8], [&[T]; 4]) -> [f32; 4]>,
) {
    assert!(out.len() >= m * rows, "output too small: {} < {}", out.len(), m * rows);
    assert!(w.len() >= rows * row_bytes, "weight slice too small");
    assert!(xs.len() >= m * nb, "input slice too small: {} < {}", xs.len(), m * nb);
    let o = SendPtr(out.as_mut_ptr());
    pool.run(&|tid, nth| {
        let range = split(rows, tid, nth);
        let mut r0 = range.start;
        while r0 < range.end {
            let r1 = (r0 + ROW_BLOCK).min(range.end);
            let mut i = 0;
            if let Some(d4) = dot4 {
                while i + 4 <= m {
                    let x4 = [&xs[i * nb..(i + 1) * nb], &xs[(i + 1) * nb..(i + 2) * nb], &xs[(i + 2) * nb..(i + 3) * nb], &xs[(i + 3) * nb..(i + 4) * nb]];
                    for r in r0..r1 {
                        let v = d4(&w[r * row_bytes..(r + 1) * row_bytes], x4);
                        // SAFETY: rows are partitioned disjointly across threads
                        // and each (i, r) cell is written exactly once.
                        unsafe {
                            o.write(i * rows + r, v[0]);
                            o.write((i + 1) * rows + r, v[1]);
                            o.write((i + 2) * rows + r, v[2]);
                            o.write((i + 3) * rows + r, v[3]);
                        }
                    }
                    i += 4;
                }
            }
            while i < m {
                let x = &xs[i * nb..(i + 1) * nb];
                for r in r0..r1 {
                    let v = dot(&w[r * row_bytes..(r + 1) * row_bytes], x);
                    // SAFETY: as above.
                    unsafe { o.write(i * rows + r, v) };
                }
                i += 1;
            }
            r0 = r1;
        }
    });
}

macro_rules! matmul_fns {
    ($($name:ident, $kind:expr, $field:ident, $x4:expr, $xt:ty, $bs:expr;)*) => {
        $(
            /// Batched version of the matching `matvec_*` (see [`matmul_rows`]).
            pub fn $name(pool: &Pool, out: &mut [f32], w: &[u8], xs: &[$xt], m: usize, rows: usize, cols: usize) {
                let k = active();
                matmul_rows(pool, out, w, xs, m, cols / $bs, rows, $kind.row_bytes(cols), k.$field, $x4(k));
            }
        )*
    };
}

matmul_fns! {
    matmul_q4_0, Kind::Q4_0, q4_0_q8_0, |k: &Kernels| k.q4_0_q8_0_x4, BlockQ8_0, QK8_0;
    matmul_q8_0, Kind::Q8_0, q8_0_q8_0, |k: &Kernels| k.q8_0_q8_0_x4, BlockQ8_0, QK8_0;
    matmul_q4_k, Kind::Q4_K, q4_k_q8_k, |k: &Kernels| k.q4_k_q8_k_x4, BlockQ8_K, QK_K;
    matmul_q5_k, Kind::Q5_K, q5_k_q8_k, |_k: &Kernels| None, BlockQ8_K, QK_K;
    matmul_q6_k, Kind::Q6_K, q6_k_q8_k, |k: &Kernels| k.q6_k_q8_k_x4, BlockQ8_K, QK_K;
    matmul_f16, Kind::F16, f16_f32, |_k: &Kernels| None, f32, 1;
    matmul_bf16, Kind::BF16, bf16_f32, |_k: &Kernels| None, f32, 1;
    matmul_f32, Kind::F32, f32_f32, |_k: &Kernels| None, f32, 1;
}

/// 1x4 micro-kernel for `kind` with a Q8_K input, if available.
pub fn dot4_fn_q8_k(kind: Kind) -> Option<fn(&[u8], [&[BlockQ8_K]; 4]) -> [f32; 4]> {
    let k = active();
    match kind {
        Kind::Q4_K => k.q4_k_q8_k_x4,
        Kind::Q6_K => k.q6_k_q8_k_x4,
        _ => None,
    }
}

/// 1x4 micro-kernel for `kind` with a Q8_0 input, if available.
pub fn dot4_fn_q8_0(kind: Kind) -> Option<fn(&[u8], [&[BlockQ8_0]; 4]) -> [f32; 4]> {
    let k = active();
    match kind {
        Kind::Q4_0 => k.q4_0_q8_0_x4,
        Kind::Q8_0 => k.q8_0_q8_0_x4,
        _ => None,
    }
}

/// Batched gate/up pair sharing one input (see [`matvec2_q8_k_input`]).
#[allow(clippy::too_many_arguments)]
pub fn matmul2_q8_k_input(
    pool: &Pool,
    out_a: &mut [f32],
    out_b: &mut [f32],
    w_a: &[u8],
    w_b: &[u8],
    xs: &[BlockQ8_K],
    m: usize,
    rows: usize,
    row_bytes: usize,
    dot: fn(&[u8], &[BlockQ8_K]) -> f32,
    dot4: Option<fn(&[u8], [&[BlockQ8_K]; 4]) -> [f32; 4]>,
) {
    let nb = xs.len() / m.max(1);
    matmul_rows(pool, out_a, w_a, xs, m, nb, rows, row_bytes, dot, dot4);
    matmul_rows(pool, out_b, w_b, xs, m, nb, rows, row_bytes, dot, dot4);
}

/// Batched gate/up pair for Q8_0-quantized inputs.
#[allow(clippy::too_many_arguments)]
pub fn matmul2_q8_0_input(
    pool: &Pool,
    out_a: &mut [f32],
    out_b: &mut [f32],
    w_a: &[u8],
    w_b: &[u8],
    xs: &[BlockQ8_0],
    m: usize,
    rows: usize,
    row_bytes: usize,
    dot: fn(&[u8], &[BlockQ8_0]) -> f32,
    dot4: Option<fn(&[u8], [&[BlockQ8_0]; 4]) -> [f32; 4]>,
) {
    let nb = xs.len() / m.max(1);
    matmul_rows(pool, out_a, w_a, xs, m, nb, rows, row_bytes, dot, dot4);
    matmul_rows(pool, out_b, w_b, xs, m, nb, rows, row_bytes, dot, dot4);
}
