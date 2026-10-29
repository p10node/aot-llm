//! Output constraints for generation: stop strings and a fixed set of
//! allowed answers (`--stop`, `--choices`).
//!
//! Both are model independent and shared by the CLI driver and the C API.

use super::tokenizer::Tokenizer;
use std::collections::BTreeMap;

/// Detects stop strings in the generated text while holding back the bytes
/// that could still be the beginning of one, so nothing of a stop string is
/// ever emitted.
pub struct StopMatcher {
    stops: Vec<Vec<u8>>,
    pending: Vec<u8>,
    /// Longest stop string (bound on the bytes that have to be held back).
    max_len: usize,
    /// Index of the stop string that matched, once one did.
    pub matched: Option<usize>,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> StopMatcher {
        let stops: Vec<Vec<u8>> = stops.iter().filter(|s| !s.is_empty()).map(|s| s.as_bytes().to_vec()).collect();
        let max_len = stops.iter().map(Vec::len).max().unwrap_or(0);
        StopMatcher { stops, pending: Vec::new(), max_len, matched: None }
    }

    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Add the bytes of one generated token and return the bytes that are
    /// safe to output. `matched` is set when a stop string was found; the
    /// returned bytes then end right before it.
    pub fn feed(&mut self, piece: &[u8]) -> Vec<u8> {
        if self.stops.is_empty() {
            return piece.to_vec();
        }
        self.pending.extend_from_slice(piece);
        // Earliest occurrence of any stop string.
        let mut hit: Option<(usize, usize)> = None;
        for (si, s) in self.stops.iter().enumerate() {
            if let Some(at) = find(&self.pending, s) {
                if hit.is_none_or(|(h, _)| at < h) {
                    hit = Some((at, si));
                }
            }
        }
        if let Some((at, si)) = hit {
            self.matched = Some(si);
            let out = self.pending[..at].to_vec();
            self.pending.clear();
            return out;
        }
        // Hold back the longest suffix that is a proper prefix of a stop string.
        let n = self.pending.len();
        let mut hold = 0;
        for s in &self.stops {
            let max = s.len().min(n).saturating_sub(0);
            for k in (1..=max.min(s.len() - 1)).rev() {
                if self.pending[n - k..] == s[..k] {
                    hold = hold.max(k);
                    break;
                }
            }
        }
        let hold = hold.min(self.max_len);
        let out = self.pending[..n - hold].to_vec();
        self.pending.drain(..n - hold);
        out
    }

    /// Bytes still held back (output them when generation ends without a
    /// stop string).
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Restricts generation to one of a fixed set of strings: a trie over the
/// token sequences of the choices; at every step only the tokens that
/// continue some choice are allowed, so the output is always one of them
/// exactly (spelled with the tokenizer's own split of the choice text).
pub struct Choices {
    nodes: Vec<Node>,
    cur: usize,
    texts: Vec<String>,
    /// Scratch for the allowed-token mask.
    mask: Vec<bool>,
}

#[derive(Default)]
struct Node {
    children: BTreeMap<u32, usize>,
    /// Index of the choice that ends at this node.
    terminal: Option<usize>,
}

impl Choices {
    /// Tokenize every choice (no BOS, special tokens as text) and build the
    /// trie. Empty choices are ignored.
    pub fn new(tok: &Tokenizer, choices: &[String]) -> Choices {
        let mut nodes = vec![Node::default()];
        let mut texts = Vec::new();
        for c in choices {
            if c.is_empty() {
                continue;
            }
            // SentencePiece models may spell the answer with or without the
            // leading `▁` depending on what precedes it; accept both paths.
            let mut spellings = vec![tok.encode(c, false, false)];
            let bare = tok.encode_ext(c, false, false, false);
            if bare != spellings[0] {
                spellings.push(bare);
            }
            for ids in spellings {
                let mut at = 0usize;
                for id in ids {
                    let next = nodes.len();
                    let e = *nodes[at].children.entry(id).or_insert(next);
                    if e == next {
                        nodes.push(Node::default());
                    }
                    at = e;
                }
                if nodes[at].terminal.is_none() {
                    nodes[at].terminal = Some(texts.len());
                }
            }
            texts.push(c.clone());
        }
        Choices { nodes, cur: 0, texts, mask: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Tokens allowed next from the current position.
    pub fn allowed(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes[self.cur].children.keys().copied()
    }

    /// Whether a choice ends at the current position.
    pub fn at_terminal(&self) -> Option<usize> {
        self.nodes[self.cur].terminal
    }

    /// `true` when the current position ends a choice and nothing can
    /// follow it: generation is complete.
    pub fn done(&self) -> bool {
        let n = &self.nodes[self.cur];
        n.terminal.is_some() && n.children.is_empty()
    }

    /// Set every logit to `-inf` except the allowed tokens. At a terminal
    /// node that is also a prefix of a longer choice, `stop_tokens` stay
    /// allowed so the model can end the shorter answer.
    pub fn mask(&mut self, logits: &mut [f32], stop_tokens: &[u32]) {
        self.mask.clear();
        self.mask.resize(logits.len(), false);
        for t in self.nodes[self.cur].children.keys() {
            if (*t as usize) < logits.len() {
                self.mask[*t as usize] = true;
            }
        }
        if self.nodes[self.cur].terminal.is_some() {
            for &t in stop_tokens {
                if (t as usize) < logits.len() {
                    self.mask[t as usize] = true;
                }
            }
        }
        for (l, &ok) in logits.iter_mut().zip(&self.mask) {
            if !ok {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Move along the trie; returns `false` if `token` was not allowed
    /// (the caller then stops: the choice ended with a stop token).
    pub fn advance(&mut self, token: u32) -> bool {
        match self.nodes[self.cur].children.get(&token) {
            Some(&n) => {
                self.cur = n;
                true
            }
            None => false,
        }
    }

    /// Text of the choice selected so far (when at a terminal node).
    pub fn selected(&self) -> Option<&str> {
        self.at_terminal().map(|i| self.texts[i].as_str())
    }
}

/// Interpret `\n`, `\t`, `\r`, `\0` and `\\` in a command-line argument so
/// `--stop '\n'` works in any shell.
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('\\') => out.push('\\'),
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(m: &mut StopMatcher, pieces: &[&str]) -> (String, bool) {
        let mut out = Vec::new();
        for p in pieces {
            out.extend(m.feed(p.as_bytes()));
            if m.matched.is_some() {
                return (String::from_utf8(out).unwrap(), true);
            }
        }
        out.extend(m.finish());
        (String::from_utf8(out).unwrap(), false)
    }

    #[test]
    fn stop_string_split_across_tokens() {
        let mut m = StopMatcher::new(&["</answer>".into()]);
        let (out, stopped) = feed_all(&mut m, &["Paris", " is", " nice", "</", "ans", "wer>", " more"]);
        assert_eq!(out, "Paris is nice");
        assert!(stopped);
    }

    #[test]
    fn partial_prefix_is_held_back_then_released() {
        let mut m = StopMatcher::new(&["\n\n".into()]);
        let first = m.feed(b"line\n");
        assert_eq!(first, b"line"); // the newline could start "\n\n"
        let second = m.feed(b"more");
        assert_eq!(second, b"\nmore");
        assert!(m.matched.is_none());
        let (out, stopped) = feed_all(&mut m, &["\n", "\n", "tail"]);
        assert_eq!(out, "");
        assert!(stopped);
    }

    #[test]
    fn no_stops_passes_through() {
        let mut m = StopMatcher::new(&[]);
        assert_eq!(m.feed(b"abc"), b"abc");
        assert!(m.finish().is_empty());
    }

    #[test]
    fn unescape_sequences() {
        assert_eq!(unescape("a\\nb\\t\\\\"), "a\nb\t\\");
        assert_eq!(unescape("plain"), "plain");
    }
}
