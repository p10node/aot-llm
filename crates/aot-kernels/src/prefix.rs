//! Compile-time KV prefix cache.
//!
//! `aot-llm compile --system "..."` runs the system prompt through the
//! compiled model once, at build time, and embeds the resulting key/value
//! cache in the binary. At run time a prompt that starts with the same
//! tokens loads those rows (a memcpy of a few megabytes) instead of
//! recomputing them, so the first token of a chat reply depends only on the
//! length of the user's message.
//!
//! Blob layout (little-endian):
//!
//! ```text
//! 0   "AOTP"        4  version = 1     8  n_tokens   12 n_layer   16 kv_dim
//! 24  token ids u32[n_tokens]
//! .   padding to a 64-byte boundary
//! .   for each layer: k rows f32[n_tokens * kv_dim], then v rows f32[n_tokens * kv_dim]
//! ```

use super::state::{Dims, State};

pub const MAGIC: &[u8; 4] = b"AOTP";
pub const VERSION: u32 = 1;
const HEADER: usize = 24;

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parsed view of an embedded prefix blob.
pub struct Prefix<'a> {
    pub tokens: Vec<u32>,
    n_layer: usize,
    kv_dim: usize,
    data: &'a [u8],
}

impl<'a> Prefix<'a> {
    /// Parse `blob`; `None` when it is empty or malformed.
    pub fn parse(blob: &'a [u8], d: &Dims) -> Option<Prefix<'a>> {
        if blob.len() < HEADER || &blob[..4] != MAGIC || u32_at(blob, 4) != VERSION {
            return None;
        }
        let n = u32_at(blob, 8) as usize;
        let n_layer = u32_at(blob, 12) as usize;
        let kv_dim = u32_at(blob, 16) as usize;
        if n_layer != d.n_layer || kv_dim != d.kv_dim() || n == 0 {
            return None;
        }
        let tokens: Vec<u32> = (0..n).map(|i| u32_at(blob, HEADER + 4 * i)).collect();
        let data_off = (HEADER + 4 * n).div_ceil(64) * 64;
        let need = data_off + n_layer * 2 * n * kv_dim * 4;
        if blob.len() < need {
            return None;
        }
        Some(Prefix { tokens, n_layer, kv_dim, data: &blob[data_off..need] })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Copy the cached rows into `state` (positions `0..len()`).
    pub fn load_into(&self, state: &mut State) {
        let n = self.tokens.len();
        let kv = self.kv_dim;
        let rows = n * kv;
        for layer in 0..self.n_layer {
            let at = layer * state.ctx * kv;
            let k_off = (layer * 2) * rows * 4;
            let v_off = (layer * 2 + 1) * rows * 4;
            copy_f32(&self.data[k_off..k_off + rows * 4], &mut state.k_cache[at..at + rows]);
            copy_f32(&self.data[v_off..v_off + rows * 4], &mut state.v_cache[at..at + rows]);
        }
    }
}

fn copy_f32(src: &[u8], dst: &mut [f32]) {
    // The blob is 64-byte aligned in the executable, but go through
    // byte-wise decoding to stay alignment-agnostic (sidecar files etc.).
    for (i, v) in dst.iter_mut().enumerate() {
        *v = f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]]);
    }
}

/// Serialise the first `n` cached positions of `state` as a prefix blob.
pub fn serialize(state: &State, d: &Dims, tokens: &[u32]) -> Vec<u8> {
    let n = tokens.len();
    let kv = d.kv_dim();
    let mut out = Vec::with_capacity(HEADER + 4 * n + 64 + d.n_layer * 2 * n * kv * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(d.n_layer as u32).to_le_bytes());
    out.extend_from_slice(&(kv as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for &t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    while out.len() % 64 != 0 {
        out.push(0);
    }
    for layer in 0..d.n_layer {
        let at = layer * state.ctx * kv;
        for v in &state.k_cache[at..at + n * kv] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &state.v_cache[at..at + n * kv] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}
