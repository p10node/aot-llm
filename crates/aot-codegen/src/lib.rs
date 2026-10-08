//! Code generator: turns a parsed GGUF model into a standalone Rust project
//! and drives its compilation.
//!
//! Pipeline:
//! 1. [`model::ModelSpec::from_gguf`] extracts hyper-parameters and the
//!    tensor table (offsets, shapes, quantization types) and validates that
//!    the graph is a Llama-family transformer the runtime can execute.
//! 2. [`tokenizer::build_blob`] serialises the vocabulary into the runtime's
//!    zero-copy tokenizer format.
//! 3. [`emit::write_project`] writes `Cargo.toml`, the generated
//!    `model.rs` / `payload.rs` / `main.rs` and the runtime sources.
//! 4. [`build::cargo_build`] compiles it with `opt-level=3`, fat LTO, a
//!    single codegen unit, `panic=abort` and stripped symbols.

pub mod build;
pub mod emit;
pub mod model;
pub mod rt_sources;
pub mod tokenizer;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
