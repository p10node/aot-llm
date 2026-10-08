//! Chat prompt formatting for the template families detected at compile
//! time. Special tokens are written as text and resolved by the tokenizer
//! with `parse_special = true`.

use super::tokenizer::chat_format::*;

/// Build a single-turn chat prompt ending with the assistant header.
pub fn format(fmt: u32, system: Option<&str>, user: &str) -> Option<String> {
    let mut s = String::new();
    match fmt {
        LLAMA3 => {
            if let Some(sys) = system {
                s.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                s.push_str(sys);
                s.push_str("<|eot_id|>");
            }
            s.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
            s.push_str(user);
            s.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        ZEPHYR => {
            if let Some(sys) = system {
                s.push_str("<|system|>\n");
                s.push_str(sys);
                s.push_str("</s>\n");
            }
            s.push_str("<|user|>\n");
            s.push_str(user);
            s.push_str("</s>\n<|assistant|>\n");
        }
        CHATML => {
            if let Some(sys) = system {
                s.push_str("<|im_start|>system\n");
                s.push_str(sys);
                s.push_str("<|im_end|>\n");
            }
            s.push_str("<|im_start|>user\n");
            s.push_str(user);
            s.push_str("<|im_end|>\n<|im_start|>assistant\n");
        }
        LLAMA2 => {
            s.push_str("[INST] ");
            if let Some(sys) = system {
                s.push_str("<<SYS>>\n");
                s.push_str(sys);
                s.push_str("\n<</SYS>>\n\n");
            }
            s.push_str(user);
            s.push_str(" [/INST]");
        }
        MISTRAL => {
            s.push_str("[INST] ");
            if let Some(sys) = system {
                s.push_str(sys);
                s.push_str("\n\n");
            }
            s.push_str(user);
            s.push_str(" [/INST]");
        }
        GEMMA => {
            s.push_str("<start_of_turn>user\n");
            if let Some(sys) = system {
                s.push_str(sys);
                s.push_str("\n\n");
            }
            s.push_str(user);
            s.push_str("<end_of_turn>\n<start_of_turn>model\n");
        }
        _ => return None,
    }
    Some(s)
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
