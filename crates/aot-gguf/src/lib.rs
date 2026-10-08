//! Minimal, dependency-light GGUF parser.
//!
//! The parser reads the GGUF header, the key/value metadata section and the
//! tensor descriptor table from a byte slice. It never touches tensor data:
//! callers receive byte offsets (relative to the start of the data section)
//! and resolve them against a memory-mapped file with [`Gguf::tensor_bytes`].
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

mod error;
mod types;

pub use error::{GgufError, Result};
pub use types::{file_type_name, GgmlType};

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// The `GGUF` magic, little-endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Default tensor data alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// Lossy conversion of any integer/float scalar to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            MetaValue::U8(v) => v as u64,
            MetaValue::I8(v) => v as u64,
            MetaValue::U16(v) => v as u64,
            MetaValue::I16(v) => v as u64,
            MetaValue::U32(v) => v as u64,
            MetaValue::I32(v) => v as u64,
            MetaValue::U64(v) => v,
            MetaValue::I64(v) => v as u64,
            MetaValue::Bool(v) => v as u64,
            _ => return None,
        })
    }

    /// Lossy conversion of any numeric scalar to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            MetaValue::F32(v) => v as f64,
            MetaValue::F64(v) => v,
            _ => self.as_u64()? as f64,
        })
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            MetaValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Short human readable type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            MetaValue::U8(_) => "u8",
            MetaValue::I8(_) => "i8",
            MetaValue::U16(_) => "u16",
            MetaValue::I16(_) => "i16",
            MetaValue::U32(_) => "u32",
            MetaValue::I32(_) => "i32",
            MetaValue::F32(_) => "f32",
            MetaValue::Bool(_) => "bool",
            MetaValue::String(_) => "string",
            MetaValue::Array(_) => "array",
            MetaValue::U64(_) => "u64",
            MetaValue::I64(_) => "i64",
            MetaValue::F64(_) => "f64",
        }
    }
}

/// Descriptor of one tensor stored in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order: `dims[0]` is the innermost (contiguous) one.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Size in bytes of the tensor payload.
    pub fn n_bytes(&self) -> u64 {
        self.ty.row_bytes(self.n_elements() as usize) as u64
    }

    /// Innermost dimension (number of columns for a matrix).
    pub fn cols(&self) -> u64 {
        self.dims.first().copied().unwrap_or(1)
    }

    /// Product of all dimensions except the innermost (number of rows).
    pub fn rows(&self) -> u64 {
        self.dims.iter().skip(1).product()
    }
}

/// Parsed GGUF header, metadata and tensor table.
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    /// Metadata in file order.
    pub metadata: Vec<(String, MetaValue)>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset of the (aligned) tensor data section.
    pub data_offset: u64,
    key_index: HashMap<String, usize>,
    tensor_index: HashMap<String, usize>,
}

impl GgufFile {
    /// Parse the header, metadata and tensor descriptors from `bytes`.
    ///
    /// `bytes` may be the whole file or just a prefix long enough to cover
    /// the descriptor table; tensor bounds are validated against `file_len`.
    pub fn parse(bytes: &[u8], file_len: u64) -> Result<Self> {
        let mut r = Reader { buf: bytes, pos: 0 };
        let magic = r.u32("magic")?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32("version")?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = r.u64("tensor count")?;
        let n_kv = r.u64("metadata count")?;

        let mut metadata = Vec::with_capacity(n_kv.min(1 << 16) as usize);
        let mut key_index = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string("metadata key")?;
            let vt = r.u32("metadata value type")?;
            let value = r.value(vt)?;
            key_index.insert(key.clone(), metadata.len());
            metadata.push((key, value));
        }

        let alignment = match key_index.get("general.alignment") {
            Some(&i) => metadata[i].1.as_u64().unwrap_or(DEFAULT_ALIGNMENT),
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::BadAlignment(alignment));
        }

        let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16) as usize);
        let mut tensor_index = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.string("tensor name")?;
            let n_dims = r.u32("tensor n_dims")?;
            if n_dims > 4 {
                return Err(GgufError::BadDims(name, n_dims));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64("tensor dim")?);
            }
            let ty_raw = r.u32("tensor type")?;
            let ty = GgmlType::from_u32(ty_raw)
                .ok_or_else(|| GgufError::BadTensorType(ty_raw, name.clone()))?;
            let offset = r.u64("tensor offset")?;
            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo { name, dims, ty, offset });
        }

        let data_offset = (r.pos as u64).div_ceil(alignment) * alignment;

        for t in &tensors {
            let start = data_offset + t.offset;
            let end = start + t.n_bytes();
            if end > file_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    start,
                    end,
                    size: file_len,
                });
            }
        }

        Ok(GgufFile { version, alignment, metadata, tensors, data_offset, key_index, tensor_index })
    }

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.key_index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.require(key)?.as_str().ok_or_else(|| GgufError::WrongType(key.into(), "string"))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.require(key)?.as_u64().ok_or_else(|| GgufError::WrongType(key.into(), "integer"))
    }

    pub fn get_f64(&self, key: &str) -> Result<f64> {
        self.require(key)?.as_f64().ok_or_else(|| GgufError::WrongType(key.into(), "number"))
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        self.require(key)?.as_bool().ok_or_else(|| GgufError::WrongType(key.into(), "bool"))
    }

    pub fn get_array(&self, key: &str) -> Result<&[MetaValue]> {
        self.require(key)?.as_array().ok_or_else(|| GgufError::WrongType(key.into(), "array"))
    }

    fn require(&self, key: &str) -> Result<&MetaValue> {
        self.get(key).ok_or_else(|| GgufError::MissingKey(key.into()))
    }

    /// `general.architecture`, e.g. `llama`.
    pub fn architecture(&self) -> Result<&str> {
        self.get_str("general.architecture")
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn require_tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensor(name).ok_or_else(|| GgufError::MissingTensor(name.into()))
    }

    /// Total number of bytes in the tensor data section.
    pub fn data_len(&self) -> u64 {
        self.tensors.iter().map(|t| t.offset + t.n_bytes()).max().unwrap_or(0)
    }

    /// Total number of weight elements across all tensors.
    pub fn n_params(&self) -> u64 {
        self.tensors.iter().map(TensorInfo::n_elements).sum()
    }
}

/// A GGUF file memory-mapped read-only. Tensor data is accessed zero-copy.
pub struct Gguf {
    mmap: memmap2::Mmap,
    file: GgufFile,
}

impl Gguf {
    /// Open and parse `path`. The whole file is mapped but only the header
    /// region is touched.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let f = File::open(path)?;
        // SAFETY: the mapping is read-only and the file is assumed not to be
        // truncated concurrently. This is the standard trade-off for mmap I/O.
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        let file = GgufFile::parse(&mmap, mmap.len() as u64)?;
        Ok(Gguf { mmap, file })
    }

    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Raw bytes of the tensor data section.
    pub fn data_section(&self) -> &[u8] {
        &self.mmap[self.file.data_offset as usize..]
    }

    /// Raw (quantized) bytes of one tensor.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let start = (self.file.data_offset + t.offset) as usize;
        &self.mmap[start..start + t.n_bytes() as usize]
    }
}

impl std::ops::Deref for Gguf {
    type Target = GgufFile;
    fn deref(&self) -> &GgufFile {
        &self.file
    }
}

/// Cursor over the little-endian header bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated { what, offset: self.pos })?;
        if end > self.buf.len() {
            return Err(GgufError::Truncated { what, offset: self.pos });
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }
    fn u16(&mut self, what: &'static str) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }
    fn u32(&mut self, what: &'static str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }
    fn u64(&mut self, what: &'static str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }

    fn string(&mut self, what: &'static str) -> Result<String> {
        let len = self.u64(what)? as usize;
        let start = self.pos;
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::BadString(start))
    }

    fn value(&mut self, vt: u32) -> Result<MetaValue> {
        const W: &str = "metadata value";
        Ok(match vt {
            0 => MetaValue::U8(self.u8(W)?),
            1 => MetaValue::I8(self.u8(W)? as i8),
            2 => MetaValue::U16(self.u16(W)?),
            3 => MetaValue::I16(self.u16(W)? as i16),
            4 => MetaValue::U32(self.u32(W)?),
            5 => MetaValue::I32(self.u32(W)? as i32),
            6 => MetaValue::F32(f32::from_bits(self.u32(W)?)),
            7 => MetaValue::Bool(self.u8(W)? != 0),
            8 => MetaValue::String(self.string(W)?),
            9 => {
                let et = self.u32("array element type")?;
                let n = self.u64("array length")? as usize;
                // Guard against absurd lengths before allocating.
                if n > self.buf.len() {
                    return Err(GgufError::Truncated { what: "array", offset: self.pos });
                }
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64(W)?),
            11 => MetaValue::I64(self.u64(W)? as i64),
            12 => MetaValue::F64(f64::from_bits(self.u64(W)?)),
            other => return Err(GgufError::BadValueType(other)),
        })
    }
}

/// Helper to build small GGUF files in memory (used by tests and tooling).
pub mod writer {
    use super::{GgmlType, MetaValue, GGUF_MAGIC};

    /// Incremental GGUF v3 writer producing a `Vec<u8>`.
    #[derive(Default)]
    pub struct GgufWriter {
        kv: Vec<(String, MetaValue)>,
        tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
        alignment: u64,
    }

    impl GgufWriter {
        pub fn new() -> Self {
            Self { alignment: super::DEFAULT_ALIGNMENT, ..Default::default() }
        }

        pub fn kv(&mut self, key: &str, value: MetaValue) -> &mut Self {
            self.kv.push((key.to_string(), value));
            self
        }

        /// Add a tensor. `dims` is in ggml order (innermost first).
        pub fn tensor(&mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> &mut Self {
            let n: u64 = dims.iter().product();
            assert_eq!(ty.row_bytes(n as usize), data.len(), "tensor {name}: data size mismatch");
            self.tensors.push((name.to_string(), dims.to_vec(), ty, data));
            self
        }

        pub fn finish(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
            for (k, v) in &self.kv {
                write_string(&mut out, k);
                write_value(&mut out, v, true);
            }
            let mut offset = 0u64;
            for (name, dims, ty, data) in &self.tensors {
                write_string(&mut out, name);
                out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&(*ty as u32).to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                offset += data.len() as u64;
                offset = offset.div_ceil(self.alignment) * self.alignment;
            }
            while !(out.len() as u64).is_multiple_of(self.alignment) {
                out.push(0);
            }
            for (_, _, _, data) in &self.tensors {
                out.extend_from_slice(data);
                while !(out.len() as u64).is_multiple_of(self.alignment) {
                    out.push(0);
                }
            }
            out
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn type_id(v: &MetaValue) -> u32 {
        match v {
            MetaValue::U8(_) => 0,
            MetaValue::I8(_) => 1,
            MetaValue::U16(_) => 2,
            MetaValue::I16(_) => 3,
            MetaValue::U32(_) => 4,
            MetaValue::I32(_) => 5,
            MetaValue::F32(_) => 6,
            MetaValue::Bool(_) => 7,
            MetaValue::String(_) => 8,
            MetaValue::Array(_) => 9,
            MetaValue::U64(_) => 10,
            MetaValue::I64(_) => 11,
            MetaValue::F64(_) => 12,
        }
    }

    fn write_value(out: &mut Vec<u8>, v: &MetaValue, with_type: bool) {
        if with_type {
            out.extend_from_slice(&type_id(v).to_le_bytes());
        }
        match v {
            MetaValue::U8(x) => out.push(*x),
            MetaValue::I8(x) => out.push(*x as u8),
            MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::Bool(x) => out.push(*x as u8),
            MetaValue::String(s) => write_string(out, s),
            MetaValue::Array(a) => {
                let et = a.first().map(type_id).unwrap_or(4);
                out.extend_from_slice(&et.to_le_bytes());
                out.extend_from_slice(&(a.len() as u64).to_le_bytes());
                for e in a {
                    assert_eq!(type_id(e), et, "heterogeneous GGUF array");
                    write_value(out, e, false);
                }
            }
            MetaValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::writer::GgufWriter;
    use super::*;

    #[test]
    fn roundtrip() {
        let mut w = GgufWriter::new();
        w.kv("general.architecture", MetaValue::String("llama".into()))
            .kv("llama.block_count", MetaValue::U32(2))
            .kv("tokenizer.ggml.tokens", MetaValue::Array(vec![
                MetaValue::String("a".into()),
                MetaValue::String("b".into()),
            ]))
            .tensor("x", &[4, 2], GgmlType::F32, vec![0u8; 32])
            .tensor("y", &[32], GgmlType::Q4_0, vec![1u8; 18]);
        let bytes = w.finish();
        let f = GgufFile::parse(&bytes, bytes.len() as u64).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(f.architecture().unwrap(), "llama");
        assert_eq!(f.get_u64("llama.block_count").unwrap(), 2);
        assert_eq!(f.get_array("tokenizer.ggml.tokens").unwrap().len(), 2);
        assert_eq!(f.tensors.len(), 2);
        let x = f.tensor("x").unwrap();
        assert_eq!(x.rows(), 2);
        assert_eq!(x.cols(), 4);
        assert_eq!(x.n_bytes(), 32);
        let y = f.tensor("y").unwrap();
        assert_eq!(y.offset, 32);
        assert_eq!(f.data_offset % 32, 0);
        assert_eq!(&bytes[f.data_offset as usize + 32..][..18], &[1u8; 18]);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = [0u8; 32];
        assert!(matches!(GgufFile::parse(&bytes, 32), Err(GgufError::BadMagic(_))));
    }

    #[test]
    fn rejects_truncated() {
        let mut w = GgufWriter::new();
        w.kv("k", MetaValue::U32(1));
        let bytes = w.finish();
        let r = GgufFile::parse(&bytes[..12], 12);
        assert!(matches!(r, Err(GgufError::Truncated { .. })));
    }
}
