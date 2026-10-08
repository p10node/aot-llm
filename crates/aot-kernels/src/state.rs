//! Model dimensions, tensor descriptors and the per-run activation state.

use super::quant::{BlockQ8_0, BlockQ8_K, Kind};

/// Architecture hyper-parameters of a Llama-family model.
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub dim: usize,
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Number of head dimensions rotated by RoPE (usually `head_dim`).
    pub rot_dim: usize,
    pub vocab: usize,
    pub n_ctx_train: usize,
    pub eps: f32,
    pub rope_base: f32,
    /// Linear RoPE scaling factor (positions are divided by it). 1.0 = none.
    pub rope_scale: f32,
}

impl Dims {
    pub const fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub const fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
    /// Query heads per key/value head (grouped-query attention).
    pub const fn gqa(&self) -> usize {
        self.n_head / self.n_kv_head
    }
}

/// Location and shape of one weight tensor inside the weights blob.
#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    pub name: &'static str,
    pub kind: Kind,
    /// Byte offset into the weights blob.
    pub off: usize,
    /// Size in bytes.
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub const fn row_bytes(&self) -> usize {
        self.kind.row_bytes(self.cols)
    }
}

/// Static information about the compiled model, for `--info` and stats.
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub name: &'static str,
    pub arch: &'static str,
    pub file_type: &'static str,
    pub source: &'static str,
    pub n_params: u64,
    pub weights_len: usize,
    pub n_tensors: usize,
    /// Human readable quantization mix, e.g. `q4_K: 155, q6_K: 46, f32: 45`.
    pub quant_mix: &'static str,
    pub compiler_version: &'static str,
}

/// Read-only view of the weights blob (embedded in the executable or
/// memory-mapped from a sidecar file).
pub struct Weights {
    data: &'static [u8],
}

impl Weights {
    pub fn new(data: &'static [u8]) -> Self {
        Weights { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_bytes(&self) -> &'static [u8] {
        self.data
    }

    /// Raw bytes of a tensor.
    #[inline]
    pub fn bytes(&self, t: &Tensor) -> &'static [u8] {
        &self.data[t.off..t.off + t.len]
    }

    /// Raw bytes of row `r` of a matrix tensor.
    #[inline]
    pub fn row(&self, t: &Tensor, r: usize) -> &'static [u8] {
        let rb = t.row_bytes();
        let start = t.off + r * rb;
        &self.data[start..start + rb]
    }

    /// View an f32 tensor. GGUF aligns tensors to 32 bytes and the blob is
    /// 64-byte aligned, so this never copies.
    pub fn f32s(&self, t: &Tensor) -> &'static [f32] {
        debug_assert_eq!(t.kind, Kind::F32);
        let b = self.bytes(t);
        // SAFETY: the slice is 4-byte aligned (checked below) and any bit
        // pattern is a valid f32; the lifetime is tied to the static blob.
        let (pre, mid, post) = unsafe { b.align_to::<f32>() };
        assert!(pre.is_empty() && post.is_empty(), "tensor {} is not 4-byte aligned", t.name);
        mid
    }
}

/// Activation buffers and KV cache for one sequence.
pub struct State {
    pub ctx: usize,
    /// Residual stream.
    pub x: Vec<f32>,
    /// Normalised input to attention / FFN.
    pub xb: Vec<f32>,
    /// Output of attention / FFN before the residual add.
    pub xb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub logits: Vec<f32>,
    /// Attention scores scratch, `n_head * ctx`.
    pub scores: Vec<f32>,
    /// `n_layer * ctx * kv_dim`.
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    /// Quantized activations (Q8_0 blocks) for Q4_0 / Q8_0 weights.
    pub aq0: Vec<BlockQ8_0>,
    /// Quantized activations (Q8_K blocks) for K-quant weights.
    pub aqk: Vec<BlockQ8_K>,
    /// Per-dimension inverse RoPE frequencies, `rot_dim / 2`.
    pub inv_freq: Vec<f32>,
    /// Interleaved (cos, sin) for the current position, `rot_dim`.
    pub rope_cs: Vec<f32>,
}

impl State {
    /// Allocate buffers for a context of `ctx` tokens. Large buffers are
    /// zero-initialised lazily by the allocator, so this is cheap.
    pub fn new(d: &Dims, ctx: usize, rope_freq_factors: Option<&[f32]>) -> State {
        let half = d.rot_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let mut f = d.rope_base.powf(-(2.0 * i as f32) / d.rot_dim as f32);
            if let Some(ff) = rope_freq_factors {
                f /= ff[i];
            }
            inv_freq.push(f);
        }
        State {
            ctx,
            x: vec![0.0; d.dim],
            xb: vec![0.0; d.dim],
            xb2: vec![0.0; d.dim],
            q: vec![0.0; d.q_dim()],
            k: vec![0.0; d.kv_dim()],
            v: vec![0.0; d.kv_dim()],
            hb: vec![0.0; d.hidden],
            hb2: vec![0.0; d.hidden],
            logits: vec![0.0; d.vocab],
            scores: vec![0.0; d.n_head * ctx],
            k_cache: vec![0.0; d.n_layer * ctx * d.kv_dim()],
            v_cache: vec![0.0; d.n_layer * ctx * d.kv_dim()],
            aq0: Vec::new(),
            aqk: Vec::new(),
            inv_freq,
            rope_cs: vec![0.0; d.rot_dim],
        }
    }

    /// Bytes allocated for the KV cache.
    pub fn kv_cache_bytes(&self) -> usize {
        (self.k_cache.len() + self.v_cache.len()) * 4
    }
}
