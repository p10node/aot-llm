//! Extraction of the Llama computational graph from GGUF metadata.

use aot_gguf::{file_type_name, GgmlType, Gguf, TensorInfo};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// Weight element types the runtime has kernels for.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl Kind {
    pub fn from_ggml(t: GgmlType) -> Option<Kind> {
        Some(match t {
            GgmlType::F32 => Kind::F32,
            GgmlType::F16 => Kind::F16,
            GgmlType::BF16 => Kind::BF16,
            GgmlType::Q4_0 => Kind::Q4_0,
            GgmlType::Q8_0 => Kind::Q8_0,
            GgmlType::Q4_K => Kind::Q4_K,
            GgmlType::Q5_K => Kind::Q5_K,
            GgmlType::Q6_K => Kind::Q6_K,
            _ => return None,
        })
    }

    /// Variant name in the runtime's `Kind` enum.
    pub fn rust_name(self) -> &'static str {
        match self {
            Kind::F32 => "F32",
            Kind::F16 => "F16",
            Kind::BF16 => "BF16",
            Kind::Q4_0 => "Q4_0",
            Kind::Q8_0 => "Q8_0",
            Kind::Q4_K => "Q4_K",
            Kind::Q5_K => "Q5_K",
            Kind::Q6_K => "Q6_K",
        }
    }

    /// Suffix of the runtime kernel functions (`matvec_<suffix>`).
    pub fn fn_suffix(self) -> &'static str {
        match self {
            Kind::F32 => "f32",
            Kind::F16 => "f16",
            Kind::BF16 => "bf16",
            Kind::Q4_0 => "q4_0",
            Kind::Q8_0 => "q8_0",
            Kind::Q4_K => "q4_k",
            Kind::Q5_K => "q5_k",
            Kind::Q6_K => "q6_k",
        }
    }

    pub fn ggml_name(self) -> &'static str {
        match self {
            Kind::F32 => "f32",
            Kind::F16 => "f16",
            Kind::BF16 => "bf16",
            Kind::Q4_0 => "q4_0",
            Kind::Q8_0 => "q8_0",
            Kind::Q4_K => "q4_K",
            Kind::Q5_K => "q5_K",
            Kind::Q6_K => "q6_K",
        }
    }

    /// Which quantized activation format the matmul consumes.
    pub fn activation(self) -> Activation {
        match self {
            Kind::Q4_0 | Kind::Q8_0 => Activation::Q8_0,
            Kind::Q4_K | Kind::Q5_K | Kind::Q6_K => Activation::Q8_K,
            Kind::F32 | Kind::F16 | Kind::BF16 => Activation::F32,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Kind::F32 | Kind::F16 | Kind::BF16 => 1,
            Kind::Q4_0 | Kind::Q8_0 => 32,
            Kind::Q4_K | Kind::Q5_K | Kind::Q6_K => 256,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Activation {
    F32,
    Q8_0,
    Q8_K,
}

/// One weight tensor as the generated code will see it.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub kind: Kind,
    /// Byte offset within the tensor data section.
    pub off: u64,
    pub len: u64,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub attn_norm: TensorSpec,
    pub wq: TensorSpec,
    pub wk: TensorSpec,
    pub wv: TensorSpec,
    pub wo: TensorSpec,
    pub ffn_norm: TensorSpec,
    pub gate: TensorSpec,
    pub up: TensorSpec,
    pub down: TensorSpec,
}

#[derive(Debug, Clone)]
pub struct DimsSpec {
    pub dim: usize,
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
    pub vocab: usize,
    pub n_ctx_train: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_scale: f32,
}

/// Everything the emitter needs to generate a model module.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub arch: String,
    pub file_type: String,
    pub source: String,
    pub dims: DimsSpec,
    pub token_embd: TensorSpec,
    pub output_norm: TensorSpec,
    /// Logits head; `None` when tied to `token_embd`.
    pub output: Option<TensorSpec>,
    pub rope_freqs: Option<TensorSpec>,
    pub layers: Vec<LayerSpec>,
    pub n_params: u64,
    pub n_tensors: usize,
    pub quant_mix: String,
    /// Absolute offset of the tensor data section in the GGUF file.
    pub data_offset: u64,
    /// Length of the tensor data section.
    pub data_len: u64,
    pub warnings: Vec<String>,
}

fn tensor_spec(t: &TensorInfo) -> Result<TensorSpec> {
    let kind = Kind::from_ggml(t.ty)
        .ok_or_else(|| anyhow!("tensor {} uses unsupported type {}; supported: f32, f16, bf16, q4_0, q8_0, q4_K, q5_K, q6_K", t.name, t.ty))?;
    let cols = t.cols() as usize;
    if !cols.is_multiple_of(kind.block_size()) {
        bail!("tensor {}: inner dimension {cols} is not a multiple of the {} block size", t.name, kind.ggml_name());
    }
    Ok(TensorSpec { name: t.name.clone(), kind, off: t.offset, len: t.n_bytes(), rows: t.rows() as usize, cols })
}

fn matrix(g: &Gguf, name: &str, rows: usize, cols: usize) -> Result<TensorSpec> {
    let t = g.require_tensor(name)?;
    let s = tensor_spec(t)?;
    if s.rows != rows || s.cols != cols {
        bail!("tensor {name}: expected shape [{cols} x {rows}] (ggml order), found {:?}", t.dims);
    }
    Ok(s)
}

fn vector(g: &Gguf, name: &str, len: usize) -> Result<TensorSpec> {
    let t = g.require_tensor(name)?;
    let s = tensor_spec(t)?;
    if s.kind != Kind::F32 {
        bail!("tensor {name}: norm weights must be f32, found {}", s.kind.ggml_name());
    }
    if t.n_elements() as usize != len {
        bail!("tensor {name}: expected {len} elements, found {}", t.n_elements());
    }
    Ok(s)
}

impl ModelSpec {
    /// Read hyper-parameters and the tensor table of a Llama-architecture GGUF.
    pub fn from_gguf(g: &Gguf, source: &Path) -> Result<ModelSpec> {
        let arch = g.architecture().context("reading general.architecture")?.to_string();
        if arch != "llama" {
            bail!("unsupported architecture \"{arch}\": only \"llama\" (Llama 1/2/3, TinyLlama, Mistral) is supported");
        }
        let mut warnings = Vec::new();
        let k = |s: &str| format!("{arch}.{s}");
        let dim = g.get_u64(&k("embedding_length"))? as usize;
        let hidden = g.get_u64(&k("feed_forward_length"))? as usize;
        let n_layer = g.get_u64(&k("block_count"))? as usize;
        let n_head = g.get_u64(&k("attention.head_count"))? as usize;
        let n_kv_head = g.get_u64(&k("attention.head_count_kv")).unwrap_or(n_head as u64) as usize;
        let head_dim = g.get_u64(&k("attention.key_length")).map(|v| v as usize).unwrap_or(dim / n_head);
        let rot_dim = g.get_u64(&k("rope.dimension_count")).map(|v| v as usize).unwrap_or(head_dim);
        let eps = g.get_f64(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5) as f32;
        let rope_base = g.get_f64(&k("rope.freq_base")).unwrap_or(10000.0) as f32;
        let n_ctx_train = g.get_u64(&k("context_length")).unwrap_or(2048) as usize;
        let mut rope_scale = 1.0f32;
        if let Ok(t) = g.get_str(&k("rope.scaling.type")) {
            let factor = g.get_f64(&k("rope.scaling.factor")).unwrap_or(1.0) as f32;
            match t {
                "none" | "" => {}
                "linear" => rope_scale = factor,
                other => warnings.push(format!("rope scaling type \"{other}\" (factor {factor}) is not supported; positions are used unscaled")),
            }
        }
        if !n_head.is_multiple_of(n_kv_head) {
            bail!("head_count {n_head} is not a multiple of head_count_kv {n_kv_head}");
        }
        if rot_dim > head_dim || !rot_dim.is_multiple_of(2) {
            bail!("invalid rope dimension count {rot_dim} for head_dim {head_dim}");
        }
        let vocab = match g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
            Some(a) => a.len(),
            None => g.get_u64(&k("vocab_size")).context("no tokenizer.ggml.tokens and no vocab_size")? as usize,
        };
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;

        let token_embd = matrix(g, "token_embd.weight", vocab, dim)?;
        let output_norm = vector(g, "output_norm.weight", dim)?;
        let output = match g.tensor("output.weight") {
            Some(_) => Some(matrix(g, "output.weight", vocab, dim)?),
            None => None,
        };
        let rope_freqs = match g.tensor("rope_freqs.weight") {
            Some(t) => {
                let s = tensor_spec(t)?;
                if s.kind != Kind::F32 || t.n_elements() as usize != rot_dim / 2 {
                    bail!("rope_freqs.weight must be f32[{}]", rot_dim / 2);
                }
                Some(s)
            }
            None => None,
        };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(LayerSpec {
                attn_norm: vector(g, &p("attn_norm"), dim)?,
                wq: matrix(g, &p("attn_q"), q_dim, dim)?,
                wk: matrix(g, &p("attn_k"), kv_dim, dim)?,
                wv: matrix(g, &p("attn_v"), kv_dim, dim)?,
                wo: matrix(g, &p("attn_output"), dim, q_dim)?,
                ffn_norm: vector(g, &p("ffn_norm"), dim)?,
                gate: matrix(g, &p("ffn_gate"), hidden, dim)?,
                up: matrix(g, &p("ffn_up"), hidden, dim)?,
                down: matrix(g, &p("ffn_down"), dim, hidden)?,
            });
        }
        for t in &g.tensors {
            if t.name.contains("bias") {
                bail!("tensor {} : bias tensors are not supported by the llama graph", t.name);
            }
        }
        let used: usize = 3 + 9 * n_layer + rope_freqs.is_some() as usize - output.is_none() as usize;
        if used != g.tensors.len() {
            warnings.push(format!("{} tensors in file, {used} used by the graph", g.tensors.len()));
        }

        let mut mix: BTreeMap<&'static str, usize> = BTreeMap::new();
        for t in &g.tensors {
            *mix.entry(Kind::from_ggml(t.ty).map(Kind::ggml_name).unwrap_or("other")).or_default() += 1;
        }
        let quant_mix = mix.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", ");
        let file_type = g.get("general.file_type").and_then(|v| v.as_u64()).map(|v| file_type_name(v as u32).to_string()).unwrap_or_else(|| "unknown".into());
        let name = g.get_str("general.name").map(|s| s.to_string()).unwrap_or_else(|_| source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into()));

        Ok(ModelSpec {
            name,
            arch,
            file_type,
            source: source.display().to_string(),
            dims: DimsSpec { dim, hidden, n_layer, n_head, n_kv_head, head_dim, rot_dim, vocab, n_ctx_train, eps, rope_base, rope_scale },
            token_embd,
            output_norm,
            output,
            rope_freqs,
            layers,
            n_params: g.n_params(),
            n_tensors: g.tensors.len(),
            quant_mix,
            data_offset: g.data_offset,
            data_len: g.data_len(),
            warnings,
        })
    }

    /// All tensors referenced by the graph.
    pub fn tensors(&self) -> Vec<&TensorSpec> {
        let mut v = vec![&self.token_embd, &self.output_norm];
        v.extend(self.output.iter());
        v.extend(self.rope_freqs.iter());
        for l in &self.layers {
            v.extend([&l.attn_norm, &l.wq, &l.wk, &l.wv, &l.wo, &l.ffn_norm, &l.gate, &l.up, &l.down]);
        }
        v
    }
}
