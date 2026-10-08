//! Error type for the GGUF parser.

use thiserror::Error;

/// Errors produced while parsing a GGUF file.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic: expected \"GGUF\", found {0:#010x}")]
    BadMagic(u32),

    #[error("unsupported GGUF version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),

    #[error("file truncated while reading {what} at offset {offset}")]
    Truncated { what: &'static str, offset: usize },

    #[error("invalid UTF-8 in string at offset {0}")]
    BadString(usize),

    #[error("unknown metadata value type {0}")]
    BadValueType(u32),

    #[error("unknown tensor type id {0} for tensor \"{1}\"")]
    BadTensorType(u32, String),

    #[error("tensor \"{0}\" has {1} dimensions (max 4)")]
    BadDims(String, u32),

    #[error("tensor \"{name}\" data range [{start}, {end}) exceeds file size {size}")]
    TensorOutOfBounds { name: String, start: u64, end: u64, size: u64 },

    #[error("metadata key \"{0}\" not found")]
    MissingKey(String),

    #[error("metadata key \"{0}\" has unexpected type (expected {1})")]
    WrongType(String, &'static str),

    #[error("tensor \"{0}\" not found")]
    MissingTensor(String),

    #[error("invalid alignment {0} (must be a power of two)")]
    BadAlignment(u64),
}

pub type Result<T> = std::result::Result<T, GgufError>;
