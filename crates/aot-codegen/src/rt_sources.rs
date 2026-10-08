//! The runtime sources, embedded at build time so the compiler binary is
//! self-contained. They are written verbatim into `src/rt/` of every
//! generated project.

/// `(file name inside src/rt/, contents)`.
pub const RT_SOURCES: &[(&str, &str)] = &[
    ("mod.rs", include_str!("../../aot-kernels/src/lib.rs")),
    ("app.rs", include_str!("../../aot-kernels/src/app.rs")),
    ("chat.rs", include_str!("../../aot-kernels/src/chat.rs")),
    ("matvec.rs", include_str!("../../aot-kernels/src/matvec.rs")),
    ("simd_neon.rs", include_str!("../../aot-kernels/src/simd_neon.rs")),
    ("simd_avx2.rs", include_str!("../../aot-kernels/src/simd_avx2.rs")),
    ("ops.rs", include_str!("../../aot-kernels/src/ops.rs")),
    ("pool.rs", include_str!("../../aot-kernels/src/pool.rs")),
    ("quant.rs", include_str!("../../aot-kernels/src/quant.rs")),
    ("quantize.rs", include_str!("../../aot-kernels/src/quantize.rs")),
    ("sampler.rs", include_str!("../../aot-kernels/src/sampler.rs")),
    ("state.rs", include_str!("../../aot-kernels/src/state.rs")),
    ("tokenizer.rs", include_str!("../../aot-kernels/src/tokenizer.rs")),
    ("weights.rs", include_str!("../../aot-kernels/src/weights.rs")),
];
