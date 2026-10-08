//! Zero-copy tokenizer.
//!
//! The compiler serialises the GGUF vocabulary into a flat blob (format
//! below) that is embedded in the binary. Every lookup table is precomputed
//! (sorted index for string -> id, sorted merge pairs, byte-token table), so
//! the tokenizer is usable the instant the blob is mapped: nothing is parsed
//! or allocated at startup.
//!
//! Supported models:
//! * `spm` - SentencePiece (Llama 1/2, TinyLlama, Mistral): score-based
//!   merges with byte fallback;
//! * `bpe` - byte-level BPE (Llama 3, GPT-2 style): rank-based merges with a
//!   hand-written implementation of the Llama-3 or GPT-2 pre-tokenizer regex.
//!
//! Blob layout (little-endian u32 fields at fixed offsets, see [`hdr`]):
//!
//! ```text
//! 0   "AOTK"                     48  off_tok_offsets  u32[n_tokens+1]
//! 4   version = 1                52  off_strings      concatenated token bytes
//! 8   kind (0 spm, 1 bpe)        56  off_scores       f32[n_tokens]
//! 12  n_tokens                   60  off_types        u8[n_tokens]
//! 16  n_merges                   64  off_sorted       u32[n_tokens], ids sorted by bytes
//! 20  flags                      68  off_merges       u32[4*n_merges] (left,right,rank,merged)
//! 24  bos   28 eos   32 eot      72  off_byte_tokens  u32[256]
//! 36  unk   40 pad   44 eom      76  off_special      u32[n_special], longest first
//!                                80  n_special   84 chat_format
//!                                88  n_stop      92 off_stop  u32[n_stop]
//!                                96  n_sorted (entries in the sorted index)
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub const MAGIC: &[u8; 4] = b"AOTK";
pub const VERSION: u32 = 1;
pub const KIND_SPM: u32 = 0;
pub const KIND_BPE: u32 = 1;
pub const NONE: u32 = u32::MAX;

pub const FLAG_ADD_BOS: u32 = 1 << 0;
pub const FLAG_ADD_EOS: u32 = 1 << 1;
pub const FLAG_ADD_SPACE_PREFIX: u32 = 1 << 2;
pub const FLAG_PRE_LLAMA3: u32 = 1 << 3;
pub const FLAG_IGNORE_MERGES: u32 = 1 << 4;

/// Header field offsets.
pub mod hdr {
    pub const VERSION: usize = 4;
    pub const KIND: usize = 8;
    pub const N_TOKENS: usize = 12;
    pub const N_MERGES: usize = 16;
    pub const FLAGS: usize = 20;
    pub const BOS: usize = 24;
    pub const EOS: usize = 28;
    pub const EOT: usize = 32;
    pub const UNK: usize = 36;
    pub const PAD: usize = 40;
    pub const EOM: usize = 44;
    pub const OFF_TOK_OFFSETS: usize = 48;
    pub const OFF_STRINGS: usize = 52;
    pub const OFF_SCORES: usize = 56;
    pub const OFF_TYPES: usize = 60;
    pub const OFF_SORTED: usize = 64;
    pub const OFF_MERGES: usize = 68;
    pub const OFF_BYTE_TOKENS: usize = 72;
    pub const OFF_SPECIAL: usize = 76;
    pub const N_SPECIAL: usize = 80;
    pub const CHAT_FORMAT: usize = 84;
    pub const N_STOP: usize = 88;
    pub const OFF_STOP: usize = 92;
    /// Number of entries in the sorted index (<= n_tokens when the
    /// vocabulary contains duplicate texts).
    pub const N_SORTED: usize = 96;
    pub const SIZE: usize = 128;
}

/// Token attribute types (same numbering as llama.cpp).
pub mod ttype {
    pub const UNDEFINED: u8 = 0;
    pub const NORMAL: u8 = 1;
    pub const UNKNOWN: u8 = 2;
    pub const CONTROL: u8 = 3;
    pub const USER_DEFINED: u8 = 4;
    pub const UNUSED: u8 = 5;
    pub const BYTE: u8 = 6;
}

/// Chat template families recognised by the compiler.
pub mod chat_format {
    pub const NONE: u32 = 0;
    pub const LLAMA3: u32 = 1;
    pub const ZEPHYR: u32 = 2;
    pub const CHATML: u32 = 3;
    pub const LLAMA2: u32 = 4;
    pub const GEMMA: u32 = 5;
    pub const MISTRAL: u32 = 6;
}

/// SentencePiece's visible space marker.
pub const SPM_SPACE: &[u8] = "\u{2581}".as_bytes();

#[derive(Clone, Copy)]
pub struct Tokenizer {
    blob: &'static [u8],
}

impl Tokenizer {
    /// Wrap a blob, validating the header.
    pub fn new(blob: &'static [u8]) -> Result<Self, String> {
        if blob.len() < hdr::SIZE || &blob[..4] != MAGIC {
            return Err("tokenizer blob: bad magic".into());
        }
        let t = Tokenizer { blob };
        if t.u32(hdr::VERSION) != VERSION {
            return Err(format!("tokenizer blob: unsupported version {}", t.u32(hdr::VERSION)));
        }
        Ok(t)
    }

    #[inline]
    fn u32(&self, at: usize) -> u32 {
        u32::from_le_bytes([self.blob[at], self.blob[at + 1], self.blob[at + 2], self.blob[at + 3]])
    }

    #[inline]
    fn u32_arr(&self, base: usize, i: usize) -> u32 {
        self.u32(base + 4 * i)
    }

    fn opt(&self, at: usize) -> Option<u32> {
        match self.u32(at) {
            NONE => None,
            v => Some(v),
        }
    }

    pub fn kind(&self) -> u32 {
        self.u32(hdr::KIND)
    }
    pub fn is_spm(&self) -> bool {
        self.kind() == KIND_SPM
    }
    pub fn n_tokens(&self) -> usize {
        self.u32(hdr::N_TOKENS) as usize
    }
    pub fn n_merges(&self) -> usize {
        self.u32(hdr::N_MERGES) as usize
    }
    pub fn flags(&self) -> u32 {
        self.u32(hdr::FLAGS)
    }
    pub fn add_bos(&self) -> bool {
        self.flags() & FLAG_ADD_BOS != 0
    }
    pub fn add_eos(&self) -> bool {
        self.flags() & FLAG_ADD_EOS != 0
    }
    pub fn add_space_prefix(&self) -> bool {
        self.flags() & FLAG_ADD_SPACE_PREFIX != 0
    }
    pub fn bos(&self) -> Option<u32> {
        self.opt(hdr::BOS)
    }
    pub fn eos(&self) -> Option<u32> {
        self.opt(hdr::EOS)
    }
    pub fn eot(&self) -> Option<u32> {
        self.opt(hdr::EOT)
    }
    pub fn unk(&self) -> Option<u32> {
        self.opt(hdr::UNK)
    }
    pub fn pad(&self) -> Option<u32> {
        self.opt(hdr::PAD)
    }
    pub fn chat_format(&self) -> u32 {
        self.u32(hdr::CHAT_FORMAT)
    }

    /// Raw bytes of a token as stored in the vocabulary.
    pub fn token_bytes(&self, id: u32) -> &'static [u8] {
        let offs = self.u32(hdr::OFF_TOK_OFFSETS) as usize;
        let strs = self.u32(hdr::OFF_STRINGS) as usize;
        let a = self.u32_arr(offs, id as usize) as usize;
        let b = self.u32_arr(offs, id as usize + 1) as usize;
        &self.blob[strs + a..strs + b]
    }

    pub fn token_type(&self, id: u32) -> u8 {
        self.blob[self.u32(hdr::OFF_TYPES) as usize + id as usize]
    }

    pub fn score(&self, id: u32) -> f32 {
        let at = self.u32(hdr::OFF_SCORES) as usize + 4 * id as usize;
        f32::from_le_bytes([self.blob[at], self.blob[at + 1], self.blob[at + 2], self.blob[at + 3]])
    }

    pub fn is_control(&self, id: u32) -> bool {
        matches!(self.token_type(id), ttype::CONTROL | ttype::UNKNOWN | ttype::UNUSED)
    }

    /// Whether `id` ends generation (EOS, EOT, EOM, ...).
    pub fn is_stop(&self, id: u32) -> bool {
        let n = self.u32(hdr::N_STOP) as usize;
        let off = self.u32(hdr::OFF_STOP) as usize;
        (0..n).any(|i| self.u32_arr(off, i) == id)
    }

    /// Exact lookup of a token by its stored bytes (binary search over the
    /// precomputed sorted index).
    pub fn find(&self, bytes: &[u8]) -> Option<u32> {
        let sorted = self.u32(hdr::OFF_SORTED) as usize;
        let (mut lo, mut hi) = (0usize, self.u32(hdr::N_SORTED) as usize);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let id = self.u32_arr(sorted, mid);
            match self.token_bytes(id).cmp(bytes) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(id),
            }
        }
        None
    }

    /// BPE merge lookup: `(rank, merged_id)` for the pair `(left, right)`.
    pub fn merge(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let base = self.u32(hdr::OFF_MERGES) as usize;
        let (mut lo, mut hi) = (0usize, self.n_merges());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let l = self.u32_arr(base, 4 * mid);
            let r = self.u32_arr(base, 4 * mid + 1);
            match (l, r).cmp(&(left, right)) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some((self.u32_arr(base, 4 * mid + 2), self.u32_arr(base, 4 * mid + 3))),
            }
        }
        None
    }

    /// The byte-fallback token for `b` (`<0xNN>` in SentencePiece vocabularies).
    pub fn byte_token(&self, b: u8) -> Option<u32> {
        match self.u32_arr(self.u32(hdr::OFF_BYTE_TOKENS) as usize, b as usize) {
            NONE => None,
            v => Some(v),
        }
    }

    /// Special (control / user-defined) token ids, longest text first.
    pub fn specials(&self) -> impl Iterator<Item = u32> + '_ {
        let n = self.u32(hdr::N_SPECIAL) as usize;
        let off = self.u32(hdr::OFF_SPECIAL) as usize;
        (0..n).map(move |i| self.u32_arr(off, i))
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Tokenize `text`. With `parse_special`, occurrences of special token
    /// strings (e.g. `<|eot_id|>`) in the text are mapped to their ids.
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_bos {
            if let Some(b) = self.bos() {
                out.push(b);
            }
        }
        let bytes = text.as_bytes();
        let mut prev_special = true;
        let mut i = 0usize;
        let mut seg_start = 0usize;
        while i < bytes.len() {
            let mut matched = None;
            if parse_special {
                for id in self.specials() {
                    let tb = self.token_bytes(id);
                    if !tb.is_empty() && bytes[i..].starts_with(tb) {
                        matched = Some((id, tb.len()));
                        break;
                    }
                }
            }
            if let Some((id, len)) = matched {
                if seg_start < i {
                    self.encode_fragment(&text[seg_start..i], prev_special, &mut out);
                }
                out.push(id);
                prev_special = true;
                i += len;
                seg_start = i;
            } else {
                // Advance by one UTF-8 character.
                i += utf8_len(bytes[i]);
            }
        }
        if seg_start < bytes.len() {
            self.encode_fragment(&text[seg_start..], prev_special, &mut out);
        }
        if self.add_eos() {
            if let Some(e) = self.eos() {
                out.push(e);
            }
        }
        out
    }

    fn encode_fragment(&self, s: &str, prev_special: bool, out: &mut Vec<u32>) {
        if s.is_empty() {
            return;
        }
        if self.is_spm() {
            self.encode_spm(s, prev_special, out);
        } else {
            self.encode_bpe(s, out);
        }
    }

    /// SentencePiece: greedy highest-score bigram merging with byte fallback
    /// (mirrors `llm_tokenizer_spm` in llama.cpp).
    fn encode_spm(&self, s: &str, prev_special: bool, out: &mut Vec<u32>) {
        let mut text: Vec<u8> = Vec::with_capacity(s.len() + 4);
        if prev_special && self.add_space_prefix() {
            text.extend_from_slice(SPM_SPACE);
        }
        for ch in s.chars() {
            if ch == ' ' {
                text.extend_from_slice(SPM_SPACE);
            } else {
                let mut buf = [0u8; 4];
                text.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        // Initial symbols: one per UTF-8 character.
        let mut syms: Vec<Sym> = Vec::new();
        let mut i = 0;
        while i < text.len() {
            let len = utf8_len(text[i]).min(text.len() - i);
            syms.push(Sym { start: i, len, id: NONE });
            i += len;
        }
        let n = syms.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 < n as isize { i + 1 } else { -1 }).collect();

        #[derive(PartialEq)]
        struct Bigram {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl Eq for Bigram {}
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Bigram {
            fn cmp(&self, o: &Self) -> Ordering {
                // Highest score first, then leftmost.
                self.score.total_cmp(&o.score).then_with(|| o.left.cmp(&self.left))
            }
        }
        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Bigram>, syms: &[Sym], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            let a = &syms[l];
            let b = &syms[r];
            let piece = &text[a.start..b.start + b.len];
            if let Some(id) = self.find(piece) {
                heap.push(Bigram { score: self.score(id), left: l, right: r, size: a.len + b.len });
            }
        };
        for i in 1..n {
            try_add(&mut heap, &syms, i as isize - 1, i as isize);
        }
        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left, bg.right);
            if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != bg.size {
                continue;
            }
            syms[l].len += syms[r].len;
            syms[r].len = 0;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            try_add(&mut heap, &syms, prev[l], l as isize);
            try_add(&mut heap, &syms, l as isize, next[l]);
        }
        let mut i: isize = if n > 0 { 0 } else { -1 };
        while i >= 0 {
            let s = &syms[i as usize];
            let piece = &text[s.start..s.start + s.len];
            match self.find(piece) {
                Some(id) => out.push(id),
                None => {
                    for &b in piece {
                        match self.byte_token(b).or_else(|| self.unk()) {
                            Some(id) => out.push(id),
                            None => {}
                        }
                    }
                }
            }
            i = next[i as usize];
        }
    }

    /// Byte-level BPE: pre-tokenize, map bytes to the GPT-2 unicode alphabet,
    /// then merge by rank (mirrors `llm_tokenizer_bpe` in llama.cpp).
    fn encode_bpe(&self, s: &str, out: &mut Vec<u32>) {
        let mut words = Vec::new();
        if self.flags() & FLAG_PRE_LLAMA3 != 0 {
            pretokenize_llama3(s, &mut words);
        } else {
            pretokenize_gpt2(s, &mut words);
        }
        let ignore_merges = self.flags() & FLAG_IGNORE_MERGES != 0;
        let mut mapped: Vec<u8> = Vec::new();
        for &(a, b) in &words {
            mapped.clear();
            for &byte in &s.as_bytes()[a..b] {
                let mut buf = [0u8; 4];
                mapped.extend_from_slice(byte_to_char(byte).encode_utf8(&mut buf).as_bytes());
            }
            if ignore_merges {
                if let Some(id) = self.find(&mapped) {
                    out.push(id);
                    continue;
                }
            }
            self.bpe_word(&mapped, out);
        }
    }

    fn bpe_word(&self, text: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<Sym> = Vec::new();
        let mut i = 0;
        while i < text.len() {
            let len = utf8_len(text[i]).min(text.len() - i);
            let id = self.find(&text[i..i + len]).unwrap_or(NONE);
            syms.push(Sym { start: i, len, id });
            i += len;
        }
        let n = syms.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 < n as isize { i + 1 } else { -1 }).collect();

        #[derive(PartialEq, Eq)]
        struct Bigram {
            rank: u32,
            left: usize,
            right: usize,
            size: usize,
            merged: u32,
        }
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Bigram {
            fn cmp(&self, o: &Self) -> Ordering {
                // Lowest rank first, then leftmost.
                o.rank.cmp(&self.rank).then_with(|| o.left.cmp(&self.left))
            }
        }
        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Bigram>, syms: &[Sym], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            if syms[l].id == NONE || syms[r].id == NONE {
                return;
            }
            if let Some((rank, merged)) = self.merge(syms[l].id, syms[r].id) {
                heap.push(Bigram { rank, left: l, right: r, size: syms[l].len + syms[r].len, merged });
            }
        };
        for i in 1..n {
            try_add(&mut heap, &syms, i as isize - 1, i as isize);
        }
        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left, bg.right);
            if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != bg.size {
                continue;
            }
            syms[l].len += syms[r].len;
            syms[l].id = bg.merged;
            syms[r].len = 0;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            try_add(&mut heap, &syms, prev[l], l as isize);
            try_add(&mut heap, &syms, l as isize, next[l]);
        }
        let mut i: isize = if n > 0 { 0 } else { -1 };
        while i >= 0 {
            let s = &syms[i as usize];
            if s.id != NONE {
                out.push(s.id);
            } else {
                // Unknown character: emit it byte by byte through the alphabet.
                let piece = &text[s.start..s.start + s.len];
                for &b in piece {
                    let mut buf = [0u8; 4];
                    let c = byte_to_char(b).encode_utf8(&mut buf);
                    match self.find(c.as_bytes()).or_else(|| self.unk()) {
                        Some(id) => out.push(id),
                        None => {}
                    }
                }
            }
            i = next[i as usize];
        }
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Append the bytes of token `id` to `out`. Control tokens produce no
    /// output unless `show_special` is set.
    pub fn decode_piece(&self, id: u32, out: &mut Vec<u8>, show_special: bool) {
        if id as usize >= self.n_tokens() {
            return;
        }
        let ty = self.token_type(id);
        let bytes = self.token_bytes(id);
        if self.is_spm() {
            match ty {
                ttype::BYTE => {
                    if let Some(b) = parse_byte_token(bytes) {
                        out.push(b);
                    }
                }
                ttype::CONTROL | ttype::UNKNOWN | ttype::UNUSED => {
                    if show_special {
                        out.extend_from_slice(bytes);
                    }
                }
                _ => {
                    let mut i = 0;
                    while i < bytes.len() {
                        if bytes[i..].starts_with(SPM_SPACE) {
                            out.push(b' ');
                            i += SPM_SPACE.len();
                        } else {
                            out.push(bytes[i]);
                            i += 1;
                        }
                    }
                }
            }
        } else {
            match ty {
                ttype::CONTROL | ttype::UNKNOWN | ttype::UNUSED | ttype::USER_DEFINED if ty != ttype::USER_DEFINED || self.is_stop(id) => {
                    if show_special {
                        out.extend_from_slice(bytes);
                    }
                }
                _ => {
                    // Map the GPT-2 unicode alphabet back to bytes.
                    match std::str::from_utf8(bytes) {
                        Ok(s) => {
                            for c in s.chars() {
                                match char_to_byte(c) {
                                    Some(b) => out.push(b),
                                    None => {
                                        let mut buf = [0u8; 4];
                                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                                    }
                                }
                            }
                        }
                        Err(_) => out.extend_from_slice(bytes),
                    }
                }
            }
        }
    }

    /// Decode a whole sequence to a string (lossy on invalid UTF-8).
    pub fn decode(&self, ids: &[u32], show_special: bool) -> String {
        let mut v = Vec::new();
        for &id in ids {
            self.decode_piece(id, &mut v, show_special);
        }
        String::from_utf8_lossy(&v).into_owned()
    }
}

#[derive(Clone, Copy)]
struct Sym {
    start: usize,
    len: usize,
    id: u32,
}

#[inline]
fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1 // continuation byte in isolation: treat as a single byte
    }
}

/// Parse `<0xNN>` byte tokens.
pub fn parse_byte_token(b: &[u8]) -> Option<u8> {
    if b.len() == 6 && b.starts_with(b"<0x") && b[5] == b'>' {
        let hex = std::str::from_utf8(&b[3..5]).ok()?;
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte <-> unicode alphabet
// ---------------------------------------------------------------------------

const fn is_printable_byte(b: u8) -> bool {
    (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174
}

const fn build_byte_to_char() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut n = 0u32;
    let mut b = 0usize;
    while b < 256 {
        if is_printable_byte(b as u8) {
            t[b] = b as u32;
        } else {
            t[b] = 256 + n;
            n += 1;
        }
        b += 1;
    }
    t
}

static BYTE_TO_CHAR: [u32; 256] = build_byte_to_char();

/// Map a byte to its GPT-2 alphabet character.
#[inline]
pub fn byte_to_char(b: u8) -> char {
    // SAFETY-free: every value is < 0x400 and not a surrogate.
    char::from_u32(BYTE_TO_CHAR[b as usize]).unwrap_or('\u{FFFD}')
}

/// Inverse of [`byte_to_char`].
pub fn char_to_byte(c: char) -> Option<u8> {
    let v = c as u32;
    if v < 256 && is_printable_byte(v as u8) {
        return Some(v as u8);
    }
    if (256..256 + 68).contains(&v) {
        let mut n = v - 256;
        for b in 0..256u32 {
            if !is_printable_byte(b as u8) {
                if n == 0 {
                    return Some(b as u8);
                }
                n -= 1;
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Pre-tokenizers (hand-written equivalents of the tokenizer regexes)
// ---------------------------------------------------------------------------

#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

#[inline]
fn is_number(c: char) -> bool {
    c.is_numeric()
}

#[inline]
fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// `'s|'t|'re|'ve|'m|'ll|'d`, optionally case-insensitive. Returns the
/// number of characters matched.
fn match_contraction(chars: &[(usize, char)], i: usize, ci: bool) -> Option<usize> {
    if chars[i].1 != '\'' {
        return None;
    }
    let lc = |k: usize| -> Option<char> { chars.get(k).map(|&(_, c)| if ci { c.to_ascii_lowercase() } else { c }) };
    let c1 = lc(i + 1)?;
    let c2 = lc(i + 2);
    match (c1, c2) {
        ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l')) => Some(3),
        ('s', _) | ('t', _) | ('m', _) | ('d', _) => Some(2),
        _ => None,
    }
}

/// Llama-3 pre-tokenizer:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
pub fn pretokenize_llama3(s: &str, out: &mut Vec<(usize, usize)>) {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let n = chars.len();
    let end_of = |k: usize| if k < n { chars[k].0 } else { s.len() };
    let at = |k: usize| chars.get(k).map(|&(_, c)| c);
    let mut i = 0;
    while i < n {
        let start = chars[i].0;
        let c = chars[i].1;
        if let Some(len) = match_contraction(&chars, i, true) {
            out.push((start, end_of(i + len)));
            i += len;
            continue;
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        {
            let mut j = i;
            if !is_letter(c) && !is_number(c) && !is_newline(c) {
                j = i + 1;
            }
            if at(j).is_some_and(is_letter) {
                while at(j).is_some_and(is_letter) {
                    j += 1;
                }
                out.push((start, end_of(j)));
                i = j;
                continue;
            }
        }
        // \p{N}{1,3}
        if is_number(c) {
            let mut j = i;
            while j < n && j - i < 3 && is_number(chars[j].1) {
                j += 1;
            }
            out.push((start, end_of(j)));
            i = j;
            continue;
        }
        // ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            let mut j = i;
            if c == ' ' {
                j = i + 1;
            }
            let is_punct = |x: char| !x.is_whitespace() && !is_letter(x) && !is_number(x);
            if at(j).is_some_and(is_punct) {
                while at(j).is_some_and(is_punct) {
                    j += 1;
                }
                while at(j).is_some_and(is_newline) {
                    j += 1;
                }
                out.push((start, end_of(j)));
                i = j;
                continue;
            }
        }
        if c.is_whitespace() {
            let mut j = i;
            while at(j).is_some_and(|x| x.is_whitespace()) {
                j += 1;
            }
            // \s*[\r\n]+ : ends at the last newline of the run.
            if let Some(k) = (i..j).rev().find(|&k| is_newline(chars[k].1)) {
                out.push((start, end_of(k + 1)));
                i = k + 1;
                continue;
            }
            // \s+(?!\S) : leave the last space to attach to the next word.
            if j < n && j - i > 1 {
                out.push((start, end_of(j - 1)));
                i = j - 1;
                continue;
            }
            out.push((start, end_of(j)));
            i = j;
            continue;
        }
        out.push((start, end_of(i + 1)));
        i += 1;
    }
}

/// GPT-2 pre-tokenizer:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
pub fn pretokenize_gpt2(s: &str, out: &mut Vec<(usize, usize)>) {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let n = chars.len();
    let end_of = |k: usize| if k < n { chars[k].0 } else { s.len() };
    let at = |k: usize| chars.get(k).map(|&(_, c)| c);
    let mut i = 0;
    while i < n {
        let start = chars[i].0;
        let c = chars[i].1;
        if let Some(len) = match_contraction(&chars, i, false) {
            out.push((start, end_of(i + len)));
            i += len;
            continue;
        }
        let j0 = if c == ' ' { i + 1 } else { i };
        let is_punct = |x: char| !x.is_whitespace() && !is_letter(x) && !is_number(x);
        if at(j0).is_some_and(is_letter) {
            let mut j = j0;
            while at(j).is_some_and(is_letter) {
                j += 1;
            }
            out.push((start, end_of(j)));
            i = j;
            continue;
        }
        if at(j0).is_some_and(is_number) {
            let mut j = j0;
            while at(j).is_some_and(is_number) {
                j += 1;
            }
            out.push((start, end_of(j)));
            i = j;
            continue;
        }
        if at(j0).is_some_and(is_punct) {
            let mut j = j0;
            while at(j).is_some_and(is_punct) {
                j += 1;
            }
            out.push((start, end_of(j)));
            i = j;
            continue;
        }
        if c.is_whitespace() {
            let mut j = i;
            while at(j).is_some_and(|x| x.is_whitespace()) {
                j += 1;
            }
            if j < n && j - i > 1 {
                out.push((start, end_of(j - 1)));
                i = j - 1;
                continue;
            }
            out.push((start, end_of(j)));
            i = j;
            continue;
        }
        out.push((start, end_of(i + 1)));
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(f: fn(&str, &mut Vec<(usize, usize)>), s: &str) -> Vec<&str> {
        let mut v = Vec::new();
        f(s, &mut v);
        v.into_iter().map(|(a, b)| &s[a..b]).collect()
    }

    #[test]
    fn llama3_pretokenizer() {
        assert_eq!(pieces(pretokenize_llama3, "Hello world"), vec!["Hello", " world"]);
        assert_eq!(pieces(pretokenize_llama3, "I'm here, it's 12345!"), vec!["I", "'m", " here", ",", " it", "'s", " ", "123", "45", "!"]);
        assert_eq!(pieces(pretokenize_llama3, "a  b"), vec!["a", " ", " b"]);
        assert_eq!(pieces(pretokenize_llama3, "x\n\n  y"), vec!["x", "\n\n", " ", " y"]);
        assert_eq!(pieces(pretokenize_llama3, "end.\n"), vec!["end", ".\n"]);
        assert_eq!(pieces(pretokenize_llama3, "  "), vec!["  "]);
        assert_eq!(pieces(pretokenize_llama3, "Xin chào thế giới"), vec!["Xin", " chào", " thế", " giới"]);
    }

    #[test]
    fn gpt2_pretokenizer() {
        assert_eq!(pieces(pretokenize_gpt2, "Hello world!!"), vec!["Hello", " world", "!!"]);
        assert_eq!(pieces(pretokenize_gpt2, "it's 2024"), vec!["it", "'s", " 2024"]);
        assert_eq!(pieces(pretokenize_gpt2, "a   b"), vec!["a", "  ", " b"]);
    }

    #[test]
    fn byte_alphabet_roundtrip() {
        for b in 0..=255u8 {
            let c = byte_to_char(b);
            assert_eq!(char_to_byte(c), Some(b), "byte {b}");
        }
        assert_eq!(byte_to_char(b' '), '\u{120}');
        assert_eq!(byte_to_char(b'a'), 'a');
    }

    #[test]
    fn byte_token_parse() {
        assert_eq!(parse_byte_token(b"<0x0A>"), Some(10));
        assert_eq!(parse_byte_token(b"<0xFF>"), Some(255));
        assert_eq!(parse_byte_token(b"<0x0>"), None);
        assert_eq!(parse_byte_token(b"hello"), None);
    }
}
