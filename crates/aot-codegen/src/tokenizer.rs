//! Serialises the GGUF vocabulary into the runtime's zero-copy tokenizer
//! blob (format documented in `aot-kernels/src/tokenizer.rs`).

use aot_gguf::{GgufFile, MetaValue};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;

// Mirror of the runtime constants (kept in sync by `tests/blob_roundtrip.rs`).
pub const KIND_SPM: u32 = 0;
pub const KIND_BPE: u32 = 1;
pub const NONE: u32 = u32::MAX;
pub const FLAG_ADD_BOS: u32 = 1 << 0;
pub const FLAG_ADD_EOS: u32 = 1 << 1;
pub const FLAG_ADD_SPACE_PREFIX: u32 = 1 << 2;
pub const FLAG_PRE_LLAMA3: u32 = 1 << 3;
pub const FLAG_IGNORE_MERGES: u32 = 1 << 4;
pub const TTYPE_NORMAL: u8 = 1;
pub const TTYPE_CONTROL: u8 = 3;
pub const TTYPE_USER_DEFINED: u8 = 4;
pub const TTYPE_BYTE: u8 = 6;

pub mod chat_format {
    pub const NONE: u32 = 0;
    pub const LLAMA3: u32 = 1;
    pub const ZEPHYR: u32 = 2;
    pub const CHATML: u32 = 3;
    pub const LLAMA2: u32 = 4;
    pub const GEMMA: u32 = 5;
    pub const MISTRAL: u32 = 6;
}

const HDR_SIZE: usize = 128;

/// Human readable facts about the serialised tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerSummary {
    pub kind: &'static str,
    pub pre: String,
    pub n_tokens: usize,
    pub n_merges: usize,
    pub n_special: usize,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub chat_format: u32,
    pub stop_tokens: Vec<u32>,
    pub warnings: Vec<String>,
}

pub fn chat_format_name(f: u32) -> &'static str {
    match f {
        chat_format::LLAMA3 => "llama3",
        chat_format::ZEPHYR => "zephyr",
        chat_format::CHATML => "chatml",
        chat_format::LLAMA2 => "llama2",
        chat_format::GEMMA => "gemma",
        chat_format::MISTRAL => "mistral",
        _ => "none",
    }
}

fn strings<'a>(g: &'a GgufFile, key: &str) -> Result<Vec<&'a str>> {
    g.get_array(key)?
        .iter()
        .map(|v| v.as_str().with_context(|| format!("{key}: non-string element")))
        .collect()
}

fn opt_u32(g: &GgufFile, key: &str) -> Option<u32> {
    g.get(key).and_then(MetaValue::as_u64).and_then(|v| u32::try_from(v).ok())
}

/// Detect the chat template family from the Jinja template or vocabulary.
fn detect_chat_format(template: Option<&str>, has_token: &dyn Fn(&str) -> bool) -> u32 {
    if let Some(t) = template {
        if t.contains("<|start_header_id|>") {
            return chat_format::LLAMA3;
        }
        if t.contains("<|im_start|>") {
            return chat_format::CHATML;
        }
        if t.contains("<|user|>") {
            return chat_format::ZEPHYR;
        }
        if t.contains("<start_of_turn>") {
            return chat_format::GEMMA;
        }
        if t.contains("[INST]") {
            return if t.contains("<<SYS>>") { chat_format::LLAMA2 } else { chat_format::MISTRAL };
        }
    }
    if has_token("<|start_header_id|>") {
        return chat_format::LLAMA3;
    }
    if has_token("<|im_start|>") {
        return chat_format::CHATML;
    }
    if has_token("<|user|>") {
        return chat_format::ZEPHYR;
    }
    chat_format::NONE
}

struct Blob {
    buf: Vec<u8>,
}

impl Blob {
    fn new() -> Self {
        Blob { buf: vec![0u8; HDR_SIZE] }
    }
    fn set_u32(&mut self, at: usize, v: u32) {
        self.buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn align(&mut self) {
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
    }
    /// Append a section and return its offset.
    fn section(&mut self, bytes: &[u8]) -> u32 {
        self.align();
        let off = self.buf.len() as u32;
        self.buf.extend_from_slice(bytes);
        off
    }
    fn u32s(&mut self, v: &[u32]) -> u32 {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.section(&bytes)
    }
}

/// Build the tokenizer blob from GGUF metadata.
pub fn build_blob(g: &GgufFile) -> Result<(Vec<u8>, TokenizerSummary)> {
    let mut warnings = Vec::new();
    let model = g.get_str("tokenizer.ggml.model").context("reading tokenizer.ggml.model")?;
    let kind = match model {
        "llama" => KIND_SPM,
        "gpt2" => KIND_BPE,
        other => bail!("unsupported tokenizer model \"{other}\" (supported: llama/spm, gpt2/bpe)"),
    };
    let tokens = strings(g, "tokenizer.ggml.tokens")?;
    let n = tokens.len();
    if n == 0 || n >= NONE as usize {
        bail!("vocabulary has {n} tokens");
    }
    let scores: Vec<f32> = match g.get("tokenizer.ggml.scores").and_then(MetaValue::as_array) {
        Some(a) if a.len() == n => a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect(),
        Some(_) => bail!("tokenizer.ggml.scores length mismatch"),
        None => vec![0.0; n],
    };
    let types: Vec<u8> = match g.get("tokenizer.ggml.token_type").and_then(MetaValue::as_array) {
        Some(a) if a.len() == n => a.iter().map(|v| v.as_u64().unwrap_or(1) as u8).collect(),
        Some(_) => bail!("tokenizer.ggml.token_type length mismatch"),
        None => vec![TTYPE_NORMAL; n],
    };
    let pre = g.get_str("tokenizer.ggml.pre").unwrap_or(if kind == KIND_SPM { "spm" } else { "default" }).to_string();

    // Flags.
    let mut flags = 0u32;
    let default_add_bos = kind == KIND_SPM;
    if g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(default_add_bos) {
        flags |= FLAG_ADD_BOS;
    }
    if g.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false) {
        flags |= FLAG_ADD_EOS;
    }
    if kind == KIND_SPM && g.get_bool("tokenizer.ggml.add_space_prefix").unwrap_or(true) {
        flags |= FLAG_ADD_SPACE_PREFIX;
    }
    if kind == KIND_BPE {
        match pre.as_str() {
            "llama-bpe" | "llama3" | "llama-v3" => flags |= FLAG_PRE_LLAMA3 | FLAG_IGNORE_MERGES,
            "default" | "gpt-2" | "gpt2" | "olmo" | "jina-v2-en" | "smollm" => {}
            other => {
                warnings.push(format!("pre-tokenizer \"{other}\" is not implemented; using the GPT-2 regex (tokenization may differ)"));
            }
        }
    }

    // Lookup by bytes.
    let mut by_text: HashMap<&[u8], u32> = HashMap::with_capacity(n);
    for (i, t) in tokens.iter().enumerate() {
        by_text.entry(t.as_bytes()).or_insert(i as u32);
    }
    let find = |s: &str| by_text.get(s.as_bytes()).copied();

    // Special ids.
    let bos = opt_u32(g, "tokenizer.ggml.bos_token_id").or(if kind == KIND_SPM { Some(1) } else { None });
    let eos = opt_u32(g, "tokenizer.ggml.eos_token_id").or(if kind == KIND_SPM { Some(2) } else { None });
    let unk = opt_u32(g, "tokenizer.ggml.unknown_token_id").or(if kind == KIND_SPM { Some(0) } else { None });
    let pad = opt_u32(g, "tokenizer.ggml.padding_token_id");
    let eot = opt_u32(g, "tokenizer.ggml.eot_token_id")
        .or_else(|| ["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>"].iter().find_map(|s| find(s)));
    let eom = opt_u32(g, "tokenizer.ggml.eom_token_id").or_else(|| find("<|eom_id|>"));
    for (name, id) in [("bos", bos), ("eos", eos), ("unk", unk), ("pad", pad), ("eot", eot), ("eom", eom)] {
        if let Some(i) = id {
            if i as usize >= n {
                bail!("{name} token id {i} is out of range (vocab {n})");
            }
        }
    }

    // Stop tokens.
    let mut stop: Vec<u32> = Vec::new();
    for id in [eos, eot, eom].into_iter().flatten() {
        stop.push(id);
    }
    for s in ["<|end_of_text|>", "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "</s>", "<|endoftext|>"] {
        if let Some(id) = find(s) {
            if types[id as usize] != TTYPE_NORMAL {
                stop.push(id);
            }
        }
    }
    stop.sort_unstable();
    stop.dedup();

    // Byte fallback table.
    let mut byte_tokens = [NONE; 256];
    if kind == KIND_SPM {
        for (i, t) in tokens.iter().enumerate() {
            if types[i] == TTYPE_BYTE {
                if let Some(b) = parse_byte_token(t.as_bytes()) {
                    byte_tokens[b as usize] = i as u32;
                }
            }
        }
        if byte_tokens.contains(&NONE) {
            warnings.push("vocabulary has no complete <0xNN> byte fallback set".into());
        }
    }

    // Sorted index.
    let mut sorted: Vec<u32> = (0..n as u32).collect();
    sorted.sort_unstable_by(|&a, &b| tokens[a as usize].as_bytes().cmp(tokens[b as usize].as_bytes()).then(a.cmp(&b)));
    // Remove duplicate texts (keep the first id) so binary search is unambiguous.
    sorted.dedup_by(|b, a| tokens[*a as usize] == tokens[*b as usize]);
    let n_sorted = sorted.len();

    // Merges.
    let mut merges: Vec<[u32; 4]> = Vec::new();
    if kind == KIND_BPE {
        let list = strings(g, "tokenizer.ggml.merges").unwrap_or_default();
        let mut skipped = 0usize;
        for (rank, m) in list.iter().enumerate() {
            let Some((l, r)) = m.split_once(' ') else {
                skipped += 1;
                continue;
            };
            let merged = format!("{l}{r}");
            match (find(l), find(r), find(&merged)) {
                (Some(li), Some(ri), Some(mi)) => merges.push([li, ri, rank as u32, mi]),
                _ => skipped += 1,
            }
        }
        if skipped > 0 {
            warnings.push(format!("{skipped} merges reference tokens outside the vocabulary and were dropped"));
        }
        merges.sort_unstable_by_key(|m| (m[0], m[1], m[2]));
        merges.dedup_by_key(|m| (m[0], m[1]));
        if merges.is_empty() {
            warnings.push("BPE tokenizer has no merges; every word will be split into single characters".into());
        }
    }

    // Special tokens, longest text first for greedy matching.
    let mut special: Vec<u32> = (0..n as u32)
        .filter(|&i| matches!(types[i as usize], TTYPE_CONTROL | TTYPE_USER_DEFINED) && !tokens[i as usize].is_empty())
        .collect();
    special.sort_unstable_by(|&a, &b| tokens[b as usize].len().cmp(&tokens[a as usize].len()).then(a.cmp(&b)));

    let template = g.get_str("tokenizer.chat_template").ok();
    let chat = detect_chat_format(template, &|s| find(s).is_some());

    // Serialise.
    let mut b = Blob::new();
    let mut offsets = Vec::with_capacity(n + 1);
    let mut strings_blob = Vec::new();
    for t in &tokens {
        offsets.push(strings_blob.len() as u32);
        strings_blob.extend_from_slice(t.as_bytes());
    }
    offsets.push(strings_blob.len() as u32);
    let off_tok_offsets = b.u32s(&offsets);
    let off_strings = b.section(&strings_blob);
    let score_bytes: Vec<u8> = scores.iter().flat_map(|s| s.to_le_bytes()).collect();
    let off_scores = b.section(&score_bytes);
    let off_types = b.section(&types);
    let off_sorted = b.u32s(&sorted);
    let merge_flat: Vec<u32> = merges.iter().flatten().copied().collect();
    let off_merges = b.u32s(&merge_flat);
    let off_bytes = b.u32s(&byte_tokens);
    let off_special = b.u32s(&special);
    let off_stop = b.u32s(&stop);

    b.buf[..4].copy_from_slice(b"AOTK");
    b.set_u32(4, 1);
    b.set_u32(8, kind);
    b.set_u32(12, n_sorted as u32);
    b.set_u32(16, merges.len() as u32);
    b.set_u32(20, flags);
    b.set_u32(24, bos.unwrap_or(NONE));
    b.set_u32(28, eos.unwrap_or(NONE));
    b.set_u32(32, eot.unwrap_or(NONE));
    b.set_u32(36, unk.unwrap_or(NONE));
    b.set_u32(40, pad.unwrap_or(NONE));
    b.set_u32(44, eom.unwrap_or(NONE));
    b.set_u32(48, off_tok_offsets);
    b.set_u32(52, off_strings);
    b.set_u32(56, off_scores);
    b.set_u32(60, off_types);
    b.set_u32(64, off_sorted);
    b.set_u32(68, off_merges);
    b.set_u32(72, off_bytes);
    b.set_u32(76, off_special);
    b.set_u32(80, special.len() as u32);
    b.set_u32(84, chat);
    b.set_u32(88, stop.len() as u32);
    b.set_u32(92, off_stop);
    // n_tokens must be the full vocabulary size for id -> text lookups; the
    // sorted index may be shorter when texts are duplicated. Store the full
    // count and let the runtime binary-search over `n_sorted` entries by
    // storing that count right after the header fields.
    b.set_u32(12, n as u32);
    b.set_u32(96, n_sorted as u32);

    let summary = TokenizerSummary {
        kind: if kind == KIND_SPM { "spm" } else { "bpe" },
        pre,
        n_tokens: n,
        n_merges: merges.len(),
        n_special: special.len(),
        bos,
        eos,
        eot,
        chat_format: chat,
        stop_tokens: stop,
        warnings,
    };
    Ok((b.buf, summary))
}

fn parse_byte_token(b: &[u8]) -> Option<u8> {
    if b.len() == 6 && b.starts_with(b"<0x") && b[5] == b'>' {
        u8::from_str_radix(std::str::from_utf8(&b[3..5]).ok()?, 16).ok()
    } else {
        None
    }
}
