//! Model dimensions, tensor descriptors and the per-run activation state.

use super::quant::{BlockQ8_0, BlockQ8_K, Kind};
use std::ops::{Deref, DerefMut};

/// Page size assumed for page-aligned buffers (16 KiB covers Apple Silicon;
/// 4 KiB systems only waste a little padding).
pub const PAGE: usize = 16384;

/// Page-aligned, zero-initialised buffer of plain numbers.
///
/// Activation buffers and the KV cache live in these instead of `Vec`s so a
/// GPU backend can map them without copying (Metal's
/// `newBufferWithBytesNoCopy` needs page-aligned memory of a page-multiple
/// length) while the CPU kernels keep using them as ordinary slices. The
/// memory comes straight from `mmap`, so large buffers are zero-filled
/// lazily by the kernel, exactly like a `vec![0.0; n]` from the allocator.
pub struct Buf<T> {
    ptr: *mut T,
    len: usize,
    /// Bytes actually mapped (rounded up to whole pages); 0 when empty.
    bytes: usize,
}

// SAFETY: the buffer owns its mapping; access is through ordinary slices.
unsafe impl<T: Send> Send for Buf<T> {}
unsafe impl<T: Sync> Sync for Buf<T> {}

impl<T> Buf<T> {
    /// Allocate `len` zeroed elements. `T` must be a plain number type for
    /// which the all-zero bit pattern is a valid value.
    pub fn zeroed(len: usize) -> Buf<T> {
        let bytes = (len * std::mem::size_of::<T>()).div_ceil(PAGE) * PAGE;
        if bytes == 0 {
            return Buf { ptr: std::ptr::NonNull::dangling().as_ptr(), len, bytes: 0 };
        }
        #[cfg(unix)]
        {
            const PROT_READ: i32 = 1;
            const PROT_WRITE: i32 = 2;
            const MAP_PRIVATE: i32 = 2;
            #[cfg(target_os = "linux")]
            const MAP_ANON: i32 = 0x20;
            #[cfg(not(target_os = "linux"))]
            const MAP_ANON: i32 = 0x1000;
            extern "C" {
                fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
            }
            // SAFETY: anonymous private mapping; the kernel returns zeroed,
            // page-aligned memory or MAP_FAILED.
            let p = unsafe { mmap(std::ptr::null_mut(), bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0) };
            assert!(p as usize != usize::MAX, "mmap of {bytes} bytes failed: {}", std::io::Error::last_os_error());
            Buf { ptr: p as *mut T, len, bytes }
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::from_size_align(bytes, PAGE).unwrap();
            // SAFETY: non-zero size; zeroed allocation.
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!p.is_null(), "allocation of {bytes} bytes failed");
            Buf { ptr: p as *mut T, len, bytes }
        }
    }

    /// Bytes mapped for this buffer (a multiple of [`PAGE`]).
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

impl<T> Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` elements were allocated and zero-initialised.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for Buf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as above; `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<'a, T> IntoIterator for &'a Buf<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Buf<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Drop for Buf<T> {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        #[cfg(unix)]
        {
            extern "C" {
                fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
            }
            // SAFETY: unmapping exactly what `zeroed` mapped.
            unsafe { munmap(self.ptr as *mut _, self.bytes) };
        }
        #[cfg(not(unix))]
        {
            let layout = std::alloc::Layout::from_size_align(self.bytes, PAGE).unwrap();
            // SAFETY: allocated with the same layout in `zeroed`.
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

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
    /// Sliding-window size of the local attention layers (Gemma 3); 0 when
    /// every layer attends to the whole context.
    pub swa_window: usize,
    /// RoPE base and linear scale of the local (sliding-window) layers; the
    /// global layers use `rope_base` / `rope_scale`.
    pub rope_base_local: f32,
    pub rope_scale_local: f32,
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
///
/// The float buffers are page-aligned [`Buf`]s so a GPU backend can share them
/// with the CPU path without copies (see `metal.rs`).
pub struct State {
    pub ctx: usize,
    /// Residual stream.
    pub x: Buf<f32>,
    /// Normalised input to attention / FFN, and the attention output
    /// (`max(dim, q_dim)` elements: Qwen3 has more query dimensions than
    /// model dimensions).
    pub xb: Buf<f32>,
    /// Output of attention / FFN before the residual add.
    pub xb2: Buf<f32>,
    pub q: Buf<f32>,
    pub k: Buf<f32>,
    pub v: Buf<f32>,
    pub hb: Buf<f32>,
    pub hb2: Buf<f32>,
    pub logits: Buf<f32>,
    /// Attention scores scratch, `n_head * ctx`.
    pub scores: Buf<f32>,
    /// `n_layer * ctx * kv_dim`.
    pub k_cache: Buf<f32>,
    pub v_cache: Buf<f32>,
    /// Quantized activations (Q8_0 blocks) for Q4_0 / Q8_0 weights.
    pub aq0: Vec<BlockQ8_0>,
    /// Quantized activations (Q8_K blocks) for K-quant weights.
    pub aqk: Vec<BlockQ8_K>,
    /// Per-dimension inverse RoPE frequencies, `rot_dim / 2`.
    pub inv_freq: Vec<f32>,
    /// Interleaved (cos, sin) for the current position, `rot_dim`.
    pub rope_cs: Buf<f32>,
    /// The same for the sliding-window layers (empty without `swa_window`).
    pub inv_freq_local: Vec<f32>,
    pub rope_cs_local: Buf<f32>,
    /// Largest number of prompt tokens processed in one batched step.
    pub max_batch: usize,
    /// Batched activations (`max_batch` rows each); empty when `max_batch == 1`.
    pub bx: Buf<f32>,
    pub bxb: Buf<f32>,
    pub bxb2: Buf<f32>,
    pub bq: Buf<f32>,
    pub bk: Buf<f32>,
    pub bv: Buf<f32>,
    pub bhb: Buf<f32>,
    pub bhb2: Buf<f32>,
    pub brope_cs: Buf<f32>,
    pub brope_cs_local: Buf<f32>,
    pub baq0: Vec<BlockQ8_0>,
    pub baqk: Vec<BlockQ8_K>,
}

impl State {
    /// Allocate buffers for a context of `ctx` tokens, `n_threads` workers
    /// and prompt batches of up to `max_batch` tokens. Large buffers are
    /// zero-initialised lazily by the allocator, so this is cheap.
    pub fn new(d: &Dims, ctx: usize, rope_freq_factors: Option<&[f32]>, n_threads: usize, max_batch: usize) -> State {
        let max_batch = max_batch.max(1);
        let b = if max_batch > 1 { max_batch } else { 0 };
        let half = d.rot_dim / 2;
        let table = |base: f32| -> Vec<f32> {
            (0..half)
                .map(|i| {
                    let mut f = base.powf(-(2.0 * i as f32) / d.rot_dim as f32);
                    if let Some(ff) = rope_freq_factors {
                        f /= ff[i];
                    }
                    f
                })
                .collect()
        };
        let inv_freq = table(d.rope_base);
        let (inv_freq_local, local_len) = if d.swa_window > 0 { (table(d.rope_base_local), d.rot_dim) } else { (Vec::new(), 0) };
        State {
            ctx,
            x: Buf::zeroed(d.dim),
            xb: Buf::zeroed(d.dim.max(d.q_dim())),
            xb2: Buf::zeroed(d.dim),
            q: Buf::zeroed(d.q_dim()),
            k: Buf::zeroed(d.kv_dim()),
            v: Buf::zeroed(d.kv_dim()),
            hb: Buf::zeroed(d.hidden),
            hb2: Buf::zeroed(d.hidden),
            logits: Buf::zeroed(d.vocab),
            scores: Buf::zeroed(d.n_head.max(n_threads) * ctx),
            k_cache: Buf::zeroed(d.n_layer * ctx * d.kv_dim()),
            v_cache: Buf::zeroed(d.n_layer * ctx * d.kv_dim()),
            aq0: Vec::new(),
            aqk: Vec::new(),
            inv_freq,
            rope_cs: Buf::zeroed(d.rot_dim),
            inv_freq_local,
            rope_cs_local: Buf::zeroed(local_len),
            max_batch,
            bx: Buf::zeroed(b * d.dim),
            bxb: Buf::zeroed(b * d.dim.max(d.q_dim())),
            bxb2: Buf::zeroed(b * d.dim),
            bq: Buf::zeroed(b * d.q_dim()),
            bk: Buf::zeroed(b * d.kv_dim()),
            bv: Buf::zeroed(b * d.kv_dim()),
            bhb: Buf::zeroed(b * d.hidden),
            bhb2: Buf::zeroed(b * d.hidden),
            brope_cs: Buf::zeroed(b * d.rot_dim),
            brope_cs_local: Buf::zeroed(b * local_len),
            baq0: Vec::new(),
            baqk: Vec::new(),
        }
    }

    /// Bytes allocated for the KV cache.
    pub fn kv_cache_bytes(&self) -> usize {
        (self.k_cache.len() + self.v_cache.len()) * 4
    }
}
