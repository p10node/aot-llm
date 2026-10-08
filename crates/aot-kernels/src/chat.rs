//! Chat prompt formatting for the template families detected at compile
//! time. Special tokens are written as text and resolved by the tokenizer
//! with `parse_special = true`.
//!
//! A prompt is `system_prefix(..) + user_turn(..)`. The split matters for the
//! compile-time KV prefix cache: the compiler bakes the KV cache of
//! `system_prefix` into the binary, and at run time a prompt whose tokens
//! start with that prefix skips straight to the user turn.

use super::tokenizer::chat_format::*;

/// The part of the prompt that precedes the user's message (system turn and
/// any template preamble). Empty for templates that have no system slot
/// when no system prompt is given.
pub fn system_prefix(fmt: u32, system: Option<&str>) -> Option<String> {
    let mut s = String::new();
    match fmt {
        LLAMA3 => {
            if let Some(sys) = system {
                s.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                s.push_str(sys);
                s.push_str("<|eot_id|>");
            }
        }
        ZEPHYR => {
            if let Some(sys) = system {
                s.push_str("<|system|>\n");
                s.push_str(sys);
                s.push_str("</s>\n");
            }
        }
        CHATML => {
            if let Some(sys) = system {
                s.push_str("<|im_start|>system\n");
                s.push_str(sys);
                s.push_str("<|im_end|>\n");
            }
        }
        LLAMA2 => {
            s.push_str("[INST] ");
            if let Some(sys) = system {
                s.push_str("<<SYS>>\n");
                s.push_str(sys);
                s.push_str("\n<</SYS>>\n\n");
            }
        }
        MISTRAL => {
            s.push_str("[INST] ");
            if let Some(sys) = system {
                s.push_str(sys);
                s.push_str("\n\n");
            }
        }
        GEMMA => {
            s.push_str("<start_of_turn>user\n");
            if let Some(sys) = system {
                s.push_str(sys);
                s.push_str("\n\n");
            }
        }
        _ => return None,
    }
    Some(s)
}

/// The user's message plus the assistant header that prompts a reply.
pub fn user_turn(fmt: u32, user: &str) -> Option<String> {
    let mut s = String::new();
    match fmt {
        LLAMA3 => {
            s.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
            s.push_str(user);
            s.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ZEPHYR => {
            s.push_str("<|user|>\n");
            s.push_str(user);
            s.push_str("</s>\n<|assistant|>\n");
        }
        CHATML => {
            s.push_str("<|im_start|>user\n");
            s.push_str(user);
            s.push_str("<|im_end|>\n<|im_start|>assistant\n");
        }
        LLAMA2 | MISTRAL => {
            s.push_str(user);
            s.push_str(" [/INST]");
        }
        GEMMA => {
            s.push_str(user);
            s.push_str("<end_of_turn>\n<start_of_turn>model\n");
        }
        _ => return None,
    }
    Some(s)
}

/// Build a single-turn chat prompt ending with the assistant header.
pub fn format(fmt: u32, system: Option<&str>, user: &str) -> Option<String> {
    Some(system_prefix(fmt, system)? + &user_turn(fmt, user)?)
}

pub fn name(fmt: u32) -> &'static str {
    match fmt {
        LLAMA3 => "llama3",
        ZEPHYR => "zephyr",
        CHATML => "chatml",
        LLAMA2 => "llama2",
        MISTRAL => "mistral",
        GEMMA => "gemma",
        _ => "none",
    }
}
