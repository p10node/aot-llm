//! GGML tensor element types as encoded in GGUF files.

use std::fmt;

/// Tensor element / quantization type (`ggml_type`).
///
/// The discriminants match the `ggml_type` enum in ggml so they can be read
/// straight out of a GGUF tensor descriptor.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
}

impl GgmlType {
    /// Decode a raw `ggml_type` id.
    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            _ => return None,
        })
    }

    /// Number of elements stored per quantization block.
    pub fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | I8 | I16 | I32 | I64 | F64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL => 32,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K | IQ2_XXS | IQ2_XS | IQ3_XXS | IQ1_S
            | IQ3_S | IQ2_S | IQ4_XS | IQ1_M | TQ1_0 | TQ2_0 => 256,
        }
    }

    /// Number of bytes occupied by one block.
    pub fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            I8 => 1,
            I64 | F64 => 8,
            Q4_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36,
            Q2_K => 84,
            Q3_K => 110,
            Q4_K => 144,
            Q5_K => 176,
            Q6_K => 210,
            Q8_K => 292,
            IQ2_XXS => 66,
            IQ2_XS => 74,
            IQ3_XXS => 98,
            IQ1_S => 50,
            IQ4_NL => 18,
            IQ3_S => 110,
            IQ2_S => 82,
            IQ4_XS => 136,
            IQ1_M => 56,
            TQ1_0 => 54,
            TQ2_0 => 66,
        }
    }

    /// Number of bytes needed to store `n_elements` values of this type.
    /// `n_elements` must be a multiple of `block_size()`.
    pub fn row_bytes(self, n_elements: usize) -> usize {
        n_elements / self.block_size() * self.type_size()
    }

    /// Average bits per weight.
    pub fn bits_per_weight(self) -> f64 {
        self.type_size() as f64 * 8.0 / self.block_size() as f64
    }

    /// Canonical lowercase ggml name (e.g. `q4_K`).
    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2_K => "q2_K",
            Q3_K => "q3_K",
            Q4_K => "q4_K",
            Q5_K => "q5_K",
            Q6_K => "q6_K",
            Q8_K => "q8_K",
            IQ2_XXS => "iq2_xxs",
            IQ2_XS => "iq2_xs",
            IQ3_XXS => "iq3_xxs",
            IQ1_S => "iq1_s",
            IQ4_NL => "iq4_nl",
            IQ3_S => "iq3_s",
            IQ2_S => "iq2_s",
            IQ4_XS => "iq4_xs",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            IQ1_M => "iq1_m",
            BF16 => "bf16",
            TQ1_0 => "tq1_0",
            TQ2_0 => "tq2_0",
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// File-level quantization type (`general.file_type`), informational only.
pub fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "ALL_F32",
        1 => "MOSTLY_F16",
        2 => "MOSTLY_Q4_0",
        3 => "MOSTLY_Q4_1",
        7 => "MOSTLY_Q8_0",
        8 => "MOSTLY_Q5_0",
        9 => "MOSTLY_Q5_1",
        10 => "MOSTLY_Q2_K",
        11 => "MOSTLY_Q3_K_S",
        12 => "MOSTLY_Q3_K_M",
        13 => "MOSTLY_Q3_K_L",
        14 => "MOSTLY_Q4_K_S",
        15 => "MOSTLY_Q4_K_M",
        16 => "MOSTLY_Q5_K_S",
        17 => "MOSTLY_Q5_K_M",
        18 => "MOSTLY_Q6_K",
        32 => "MOSTLY_BF16",
        _ => "UNKNOWN",
    }
}
