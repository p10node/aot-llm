//! Round-trips vocabularies through the blob builder and the runtime tokenizer.

use aot_codegen::tokenizer::build_blob;
use aot_gguf::writer::GgufWriter;
use aot_gguf::{GgufFile, MetaValue};
use aot_kernels::tokenizer::Tokenizer;

fn strs(v: &[&str]) -> MetaValue {
    MetaValue::Array(v.iter().map(|s| MetaValue::String(s.to_string())).collect())
}

fn i32s(v: &[i32]) -> MetaValue {
    MetaValue::Array(v.iter().map(|&x| MetaValue::I32(x)).collect())
}

fn f32s(v: &[f32]) -> MetaValue {
    MetaValue::Array(v.iter().map(|&x| MetaValue::F32(x)).collect())
}

fn leak(v: Vec<u8>) -> &'static [u8] {
    Box::leak(v.into_boxed_slice())
}

#[test]
fn spm_vocab_roundtrip() {
    let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    let mut types = vec![2, 3, 3];
    let mut scores = vec![0.0f32, 0.0, 0.0];
    for b in 0..=255u8 {
        tokens.push(format!("<0x{b:02X}>"));
        types.push(6);
        scores.push(0.0);
    }
    // SentencePiece vocabularies contain every single character plus the
    // prefix chain of each word, so greedy merging can reach the whole word.
    for c in "abcdefghijklmnopqrstuvwxyz\u{2581}".chars() {
        tokens.push(c.to_string());
        types.push(1);
        scores.push(-100.0);
    }
    for word in ["\u{2581}hello", "\u{2581}world"] {
        let chars: Vec<char> = word.chars().collect();
        for n in 2..=chars.len() {
            tokens.push(chars[..n].iter().collect());
            types.push(1);
            scores.push(-(20.0 - n as f32));
        }
    }
    let tok_refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let mut w = GgufWriter::new();
    w.kv("general.architecture", MetaValue::String("llama".into()))
        .kv("tokenizer.ggml.model", MetaValue::String("llama".into()))
        .kv("tokenizer.ggml.tokens", strs(&tok_refs))
        .kv("tokenizer.ggml.scores", f32s(&scores))
        .kv("tokenizer.ggml.token_type", i32s(&types))
        .kv("tokenizer.ggml.bos_token_id", MetaValue::U32(1))
        .kv("tokenizer.ggml.eos_token_id", MetaValue::U32(2))
        .kv("tokenizer.chat_template", MetaValue::String("{% for m in messages %}<|user|>\n{{ m.content }}</s>{% endfor %}".into()));
    let bytes = w.finish();
    let g = GgufFile::parse(&bytes, bytes.len() as u64).unwrap();
    let (blob, summary) = build_blob(&g).unwrap();
    assert_eq!(summary.kind, "spm");
    assert_eq!(summary.chat_format, aot_codegen::tokenizer::chat_format::ZEPHYR);
    let t = Tokenizer::new(leak(blob)).unwrap();
    assert!(t.is_spm());
    assert_eq!(t.bos(), Some(1));
    assert!(t.is_stop(2));
    let hello = t.find("\u{2581}hello".as_bytes()).unwrap();
    let world = t.find("\u{2581}world".as_bytes()).unwrap();
    // Best-scoring merges win: "▁hello" (-1) beats "▁hell" + "o".
    assert_eq!(t.encode("hello world", true, true), vec![1, hello, world]);
    // Byte fallback for characters outside the vocabulary (UTF-8 of 'é' = C3 A9);
    // the leading space prefix stays a separate "▁" token.
    let sp = t.find("\u{2581}".as_bytes()).unwrap();
    let ids = t.encode("é", false, true);
    assert_eq!(ids, vec![sp, 3 + 0xC3, 3 + 0xA9]);
    assert_eq!(t.decode(&ids, false), " é");
    // Special tokens in text are parsed.
    assert_eq!(t.encode("hello</s>", false, true), vec![hello, 2]);
    assert_eq!(t.decode(&[hello, world], false), " hello world");
}

#[test]
fn bpe_vocab_roundtrip() {
    // GPT-2 alphabet: space is U+0120 (Ġ).
    let tokens = ["<|endoftext|>", "a", "b", "c", "ab", "abc", "\u{120}", "\u{120}a", "\u{120}abc", "!"];
    let types = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let merges = ["a b", "ab c", "\u{120} a", "\u{120} abc"];
    let mut w = GgufWriter::new();
    w.kv("general.architecture", MetaValue::String("llama".into()))
        .kv("tokenizer.ggml.model", MetaValue::String("gpt2".into()))
        .kv("tokenizer.ggml.pre", MetaValue::String("default".into()))
        .kv("tokenizer.ggml.tokens", strs(&tokens))
        .kv("tokenizer.ggml.token_type", i32s(&types))
        .kv("tokenizer.ggml.merges", strs(&merges))
        .kv("tokenizer.ggml.eos_token_id", MetaValue::U32(0))
        .kv("tokenizer.ggml.add_bos_token", MetaValue::Bool(false));
    let bytes = w.finish();
    let g = GgufFile::parse(&bytes, bytes.len() as u64).unwrap();
    let (blob, summary) = build_blob(&g).unwrap();
    assert_eq!(summary.kind, "bpe");
    assert_eq!(summary.n_merges, 4);
    let t = Tokenizer::new(leak(blob)).unwrap();
    assert!(!t.add_bos());
    assert_eq!(t.merge(1, 2), Some((0, 4)));
    assert_eq!(t.encode("abc a!", false, true), vec![5, 7, 9]);
    assert_eq!(t.encode("abc abc<|endoftext|>", false, true), vec![5, 8, 0]);
    assert_eq!(t.decode(&[5, 7, 9], false), "abc a!");
    assert!(t.is_stop(0));
}
