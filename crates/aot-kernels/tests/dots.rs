//! Cross-checks every dot-product implementation against the scalar
//! reference and against a float reference computed from dequantized
//! weights.

use aot_kernels::matvec::{self, Kernels, SCALAR_KERNELS};
use aot_kernels::pool::Pool;
use aot_kernels::quant::*;
use aot_kernels::quantize::quantize_row;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn all_kernel_sets() -> Vec<Kernels> {
    let mut v = vec![SCALAR_KERNELS];
    #[cfg(target_arch = "aarch64")]
    {
        v.push(matvec::neon::PLAIN_KERNELS);
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            v.push(matvec::neon::DOTPROD_KERNELS);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            v.push(matvec::avx2::KERNELS);
            if std::is_x86_feature_detected!("f16c") {
                v.push(matvec::avx2::KERNELS_F16C);
            }
        }
    }
    v
}

fn float_ref(kind: Kind, w: &[u8], x: &[f32]) -> f32 {
    let mut d = vec![0f32; x.len()];
    dequant_row(kind, w, &mut d);
    d.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// Relative tolerance: activation quantization to int8 introduces ~0.5%
/// error, so compare SIMD against scalar tightly and against float loosely.
fn check_kind(kind: Kind, cols: usize, seed: u64) {
    let mut rng = Rng(seed);
    let w: Vec<f32> = (0..cols).map(|_| rng.f()).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.f() * 3.0).collect();
    let wq = quantize_row(kind, &w);
    let fref = float_ref(kind, &wq, &x);
    let scale = x.iter().map(|v| v.abs()).sum::<f32>() / cols as f32 * cols as f32 * 0.02 + 1e-3;

    let mut q8_0 = Vec::new();
    let mut q8_k = Vec::new();
    if kind.block_size() == QK8_0 {
        quantize_q8_0(&x, &mut q8_0);
    } else if kind.block_size() == QK_K {
        quantize_q8_k(&x, &mut q8_k);
    }

    let mut results = Vec::new();
    for k in all_kernel_sets() {
        let v = match kind {
            Kind::Q4_0 => (k.q4_0_q8_0)(&wq, &q8_0),
            Kind::Q8_0 => (k.q8_0_q8_0)(&wq, &q8_0),
            Kind::Q4_K => (k.q4_k_q8_k)(&wq, &q8_k),
            Kind::Q5_K => (k.q5_k_q8_k)(&wq, &q8_k),
            Kind::Q6_K => (k.q6_k_q8_k)(&wq, &q8_k),
            Kind::F16 => (k.f16_f32)(&wq, &x),
            Kind::BF16 => (k.bf16_f32)(&wq, &x),
            Kind::F32 => (k.f32_f32)(&wq, &x),
        };
        assert!(
            (v - fref).abs() < scale,
            "{kind:?} cols={cols} kernels={}: {v} vs float ref {fref} (tol {scale})",
            k.name
        );
        results.push((k.name, v));
    }
    let base = results[0].1;
    for (name, v) in &results[1..] {
        // Integer accumulation order differs but results must agree to fp32 rounding.
        assert!(
            (v - base).abs() <= base.abs() * 1e-4 + 1e-3,
            "{kind:?} cols={cols}: {name} = {v} differs from scalar = {base}"
        );
    }
}

#[test]
fn q4_0() {
    for (cols, seed) in [(32, 1), (64, 2), (96, 3), (2048, 4), (5632, 5)] {
        check_kind(Kind::Q4_0, cols, seed);
    }
}

#[test]
fn q8_0() {
    for (cols, seed) in [(32, 11), (64, 12), (96, 13), (2048, 14)] {
        check_kind(Kind::Q8_0, cols, seed);
    }
}

#[test]
fn q4_k() {
    for (cols, seed) in [(256, 21), (512, 22), (2048, 23), (5632, 24)] {
        check_kind(Kind::Q4_K, cols, seed);
    }
}

#[test]
fn q5_k() {
    for (cols, seed) in [(256, 31), (512, 32), (2048, 33)] {
        check_kind(Kind::Q5_K, cols, seed);
    }
}

#[test]
fn q6_k() {
    for (cols, seed) in [(256, 41), (512, 42), (2048, 43), (8192, 44)] {
        check_kind(Kind::Q6_K, cols, seed);
    }
}

#[test]
fn floats() {
    for (cols, seed) in [(8, 51), (12, 52), (64, 53), (2048, 54), (2053, 55)] {
        check_kind(Kind::F16, cols, seed);
        check_kind(Kind::BF16, cols, seed);
        check_kind(Kind::F32, cols, seed);
    }
}

#[test]
fn matvec_parallel_matches_serial() {
    let rows = 37;
    let cols = 512;
    let mut rng = Rng(99);
    let w: Vec<Vec<f32>> = (0..rows).map(|_| (0..cols).map(|_| rng.f()).collect()).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.f()).collect();
    let wq: Vec<u8> = w.iter().flat_map(|r| quantize_row(Kind::Q4_K, r)).collect();
    let mut xq = Vec::new();
    quantize_q8_k(&x, &mut xq);
    let pool1 = Pool::new(1);
    let pool4 = Pool::new(4);
    let mut out1 = vec![0f32; rows];
    let mut out4 = vec![0f32; rows];
    matvec::matvec_q4_k(&pool1, &mut out1, &wq, &xq, rows, cols);
    matvec::matvec_q4_k(&pool4, &mut out4, &wq, &xq, rows, cols);
    for r in 0..rows {
        assert_eq!(out1[r], out4[r]);
        let fref = float_ref(Kind::Q4_K, &wq[r * Kind::Q4_K.row_bytes(cols)..][..Kind::Q4_K.row_bytes(cols)], &x);
        assert!((out1[r] - fref).abs() < 0.5, "row {r}: {} vs {fref}", out1[r]);
    }
}
