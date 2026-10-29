//! Compile-time KV prefix cache.
//!
//! `aot-llm compile --system "..."` runs the system prompt through the
//! compiled model once, at build time, and embeds the resulting key/value
//! cache in the binary. At run time a prompt that starts with the same
//! tokens loads those rows (a memcpy of a few megabytes) instead of
//! recomputing them, so the first token of a chat reply depends only on the
//! length of the user's message.
//!
//! The same format serves `--save-kv` / `--load-kv`: the KV cache of a
//! finished run (prompt plus reply) written to a file and continued by the
//! next process, so a multi-turn conversation only ever evaluates the new
//! turn.
//!
//! Blob layout (little-endian):
//!
//! ```text
//! 0   "AOTP"        4  version = 2     8  n_tokens   12 n_layer   16 kv_dim   20 n_cached
//! 24  token ids u32[n_tokens]
//! .   padding to a 64-byte boundary
//! .   for each layer: k rows f32[n_cached * kv_dim], then v rows f32[n_cached * kv_dim]
//! ```
//!
//! `n_cached <= n_tokens`: the first `n_cached` tokens have cache rows, the
//! rest (the last generated token of a saved run, which was never fed back)
//! must be evaluated. Version 1 blobs have `n_cached = n_tokens`.

use super::state::{Dims, State};

pub const MAGIC: &[u8; 4] = b"AOTP";
pub const VERSION: u32 = 2;
const HEADER: usize = 24;

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parsed view of an embedded prefix blob.
pub struct Prefix<'a> {
    pub tokens: Vec<u32>,
    n_cached: usize,
    n_layer: usize,
    kv_dim: usize,
    data: &'a [u8],
}

impl<'a> Prefix<'a> {
    /// Parse `blob`; `None` when it is empty or malformed.
    pub fn parse(blob: &'a [u8], d: &Dims) -> Option<Prefix<'a>> {
        if blob.len() < HEADER || &blob[..4] != MAGIC {
            return None;
        }
        let version = u32_at(blob, 4);
        if version != 1 && version != VERSION {
            return None;
        }
        let n = u32_at(blob, 8) as usize;
        let n_layer = u32_at(blob, 12) as usize;
        let kv_dim = u32_at(blob, 16) as usize;
        let n_cached = if version == 1 { n } else { u32_at(blob, 20) as usize };
        if n_layer != d.n_layer || kv_dim != d.kv_dim() || n == 0 || n_cached > n {
            return None;
        }
        let tokens: Vec<u32> = (0..n).map(|i| u32_at(blob, HEADER + 4 * i)).collect();
        let data_off = (HEADER + 4 * n).div_ceil(64) * 64;
        let need = data_off + n_layer * 2 * n_cached * kv_dim * 4;
        if blob.len() < need {
            return None;
        }
        Some(Prefix { tokens, n_cached, n_layer, kv_dim, data: &blob[data_off..need] })
    }

    /// Number of tokens in the blob (cached ones first).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Tokens whose KV rows are in the blob (`<= len()`).
    pub fn cached(&self) -> usize {
        self.n_cached
    }

    /// Copy the cached rows into `state` (positions `0..cached()`). The
    /// state's context must hold them.
    pub fn load_into(&self, state: &mut State) {
        let n = self.n_cached;
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

/// Serialise `tokens` and the cache rows of their first `n_cached`
/// positions (`n_cached <= tokens.len()`, and those positions must have
/// been evaluated into `state`).
pub fn serialize(state: &State, d: &Dims, tokens: &[u32], n_cached: usize) -> Vec<u8> {
    let n = tokens.len();
    let n_cached = n_cached.min(n);
    let kv = d.kv_dim();
    let mut out = Vec::with_capacity(HEADER + 4 * n + 64 + d.n_layer * 2 * n_cached * kv * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(d.n_layer as u32).to_le_bytes());
    out.extend_from_slice(&(kv as u32).to_le_bytes());
    out.extend_from_slice(&(n_cached as u32).to_le_bytes());
    for &t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    while out.len() % 64 != 0 {
        out.push(0);
    }
    for layer in 0..d.n_layer {
        let at = layer * state.ctx * kv;
        for v in &state.k_cache[at..at + n_cached * kv] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &state.v_cache[at..at + n_cached * kv] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}
