//! Quantization block layouts, half-float helpers, activation quantizers and
//! scalar (reference) dequantizers.
//!
//! Weight blocks are never cast to structs: all weight access goes through
//! byte slices with explicit little-endian decoding, so there are no alignment
//! requirements on the memory-mapped data. Activation blocks ([`BlockQ8_0`],
//! [`BlockQ8_K`]) are owned by the runtime and therefore use plain structs.
//!
//! Layouts follow ggml (`ggml-common.h`):
//!
//! | type | block | bytes | layout                                               |
//! |------|-------|-------|------------------------------------------------------|
//! | Q4_0 | 32    | 18    | f16 d, 16 x u8 nibbles (lo = elem j, hi = elem j+16)  |
//! | Q8_0 | 32    | 34    | f16 d, 32 x i8                                        |
//! | Q4_K | 256   | 144   | f16 d, f16 dmin, 12 x u8 scales/mins, 128 x u8 nibbles|
//! | Q5_K | 256   | 176   | f16 d, f16 dmin, 12 x u8 scales, 32 x u8 qh, 128 x qs |
//! | Q6_K | 256   | 210   | 128 x u8 ql, 64 x u8 qh, 16 x i8 scales, f16 d        |

/// Elements per Q4_0 / Q8_0 block.
pub const QK8_0: usize = 32;
/// Elements per K-quant super-block.
pub const QK_K: usize = 256;

pub const Q4_0_SIZE: usize = 18;
pub const Q8_0_SIZE: usize = 34;
pub const Q4_K_SIZE: usize = 144;
pub const Q5_K_SIZE: usize = 176;
pub const Q6_K_SIZE: usize = 210;

/// Weight element types supported by the runtime.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub const fn block_size(self) -> usize {
        match self {
            Kind::F32 | Kind::F16 | Kind::BF16 => 1,
            Kind::Q4_0 | Kind::Q8_0 => QK8_0,
            Kind::Q4_K | Kind::Q5_K | Kind::Q6_K => QK_K,
        }
    }

    pub const fn type_size(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F16 | Kind::BF16 => 2,
            Kind::Q4_0 => Q4_0_SIZE,
            Kind::Q8_0 => Q8_0_SIZE,
            Kind::Q4_K => Q4_K_SIZE,
            Kind::Q5_K => Q5_K_SIZE,
            Kind::Q6_K => Q6_K_SIZE,
        }
    }

    /// Bytes per row of `n` elements.
    pub const fn row_bytes(self, n: usize) -> usize {
        n / self.block_size() * self.type_size()
    }

    pub const fn name(self) -> &'static str {
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
}

// ---------------------------------------------------------------------------
// Half precision helpers
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 binary16 bit pattern to `f32` (handles subnormals,
/// infinities and NaNs).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: value = mant * 2^-24. Normalise by shifting the
            // mantissa until the implicit bit is in place.
            // A half subnormal with top bit at position k equals
            // 1.xxx * 2^(k-24); the f32 exponent field is k + 103. Shifting
            // the mantissa up to bit 10 takes (10 - k) steps from 113.
            let mut e = 113i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Convert `f32` to binary16 with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xFF) as i32;
    let mant = x & 0x7F_FFFF;
    if exp == 0xFF {
        // Inf or NaN.
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    exp -= 127;
    if exp > 15 {
        return sign | 0x7C00; // overflow -> inf
    }
    if exp < -25 {
        return sign; // underflow -> zero
    }
    if exp < -14 {
        // Subnormal half: shift mantissa (with implicit bit) right.
        let m = mant | 0x80_0000;
        let shift = (-14 - exp) as u32 + 13;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = (((exp + 15) as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // may carry into the exponent, which is the correct rounding
    }
    sign | r as u16
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(f: f32) -> u16 {
    // Round to nearest even.
    let x = f.to_bits();
    if (x & 0x7F80_0000) == 0x7F80_0000 {
        return (x >> 16) as u16 | if x & 0x7F_FFFF != 0 { 0x40 } else { 0 };
    }
    let lsb = (x >> 16) & 1;
    ((x + 0x7FFF + lsb) >> 16) as u16
}

#[inline]
pub fn read_f16(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

#[inline]
pub fn read_f32(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

// ---------------------------------------------------------------------------
// Activation blocks
// ---------------------------------------------------------------------------

/// 32 int8 values with one f32 scale (activation side of Q4_0/Q8_0 dots).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f32,
    pub qs: [i8; QK8_0],
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self { d: 0.0, qs: [0; QK8_0] }
    }
}

/// 256 int8 values with one f32 scale and 16 partial sums of 16 values each
/// (activation side of K-quant dots; `bsums` lets kernels apply the block
/// minimums without touching individual elements).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8_K {
    fn default() -> Self {
        Self { d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16] }
    }
}

#[inline]
fn round_i(x: f32) -> i32 {
    // Round half away from zero, like ggml's nearest_int for our value range.
    x.round() as i32
}

/// Quantize `x` (length multiple of 32) into Q8_0 blocks, reusing `out`.
pub fn quantize_q8_0(x: &[f32], out: &mut Vec<BlockQ8_0>) {
    debug_assert_eq!(x.len() % QK8_0, 0);
    let nb = x.len() / QK8_0;
    out.resize(nb, BlockQ8_0::default());
    for (i, blk) in out.iter_mut().enumerate() {
        let src = &x[i * QK8_0..(i + 1) * QK8_0];
        let amax = src.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = d;
        for (q, &v) in blk.qs.iter_mut().zip(src) {
            *q = round_i(v * id).clamp(-127, 127) as i8;
        }
    }
}

/// Quantize `x` (length multiple of 256) into Q8_K blocks, reusing `out`.
pub fn quantize_q8_k(x: &[f32], out: &mut Vec<BlockQ8_K>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;
    out.resize(nb, BlockQ8_K::default());
    for (i, blk) in out.iter_mut().enumerate() {
        let src = &x[i * QK_K..(i + 1) * QK_K];
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in src {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            blk.d = 0.0;
            blk.qs = [0; QK_K];
            blk.bsums = [0; QK_K / 16];
            continue;
        }
        // Same convention as ggml: scale by the signed extreme so that it maps
        // to -127 exactly, which slightly improves accuracy.
        let iscale = -127.0 / max;
        for (q, &v) in blk.qs.iter_mut().zip(src) {
            *q = round_i(iscale * v).min(127) as i8;
        }
        for j in 0..QK_K / 16 {
            let mut s = 0i32;
            for k in 0..16 {
                s += blk.qs[j * 16 + k] as i32;
            }
            blk.bsums[j] = s as i16;
        }
        blk.d = 1.0 / iscale;
    }
}

// ---------------------------------------------------------------------------
// Scalar reference dequantizers (one row)
// ---------------------------------------------------------------------------

/// Decode the 6-bit scale and minimum of sub-block `j` from the packed
/// 12-byte K-quant scale array (`get_scale_min_k4` in ggml).
#[inline]
pub fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

pub fn dequant_row_f32(row: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = read_f32(row, i * 4);
    }
}

pub fn dequant_row_f16(row: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = read_f16(row, i * 2);
    }
}

pub fn dequant_row_bf16(row: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = bf16_to_f32(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]));
    }
}

pub fn dequant_row_q4_0(row: &[u8], out: &mut [f32]) {
    for (i, chunk) in out.chunks_exact_mut(QK8_0).enumerate() {
        let b = &row[i * Q4_0_SIZE..(i + 1) * Q4_0_SIZE];
        let d = read_f16(b, 0);
        for j in 0..16 {
            let q = b[2 + j];
            chunk[j] = ((q & 0x0F) as i32 - 8) as f32 * d;
            chunk[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
        }
    }
}

pub fn dequant_row_q8_0(row: &[u8], out: &mut [f32]) {
    for (i, chunk) in out.chunks_exact_mut(QK8_0).enumerate() {
        let b = &row[i * Q8_0_SIZE..(i + 1) * Q8_0_SIZE];
        let d = read_f16(b, 0);
        for j in 0..QK8_0 {
            chunk[j] = (b[2 + j] as i8) as f32 * d;
        }
    }
}

pub fn dequant_row_q4_k(row: &[u8], out: &mut [f32]) {
    for (i, chunk) in out.chunks_exact_mut(QK_K).enumerate() {
        let b = &row[i * Q4_K_SIZE..(i + 1) * Q4_K_SIZE];
        let d = read_f16(b, 0);
        let dmin = read_f16(b, 2);
        let scales = &b[4..16];
        let qs = &b[16..144];
        for j in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let m1 = dmin * m1 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                let q = qs[32 * j + l];
                chunk[64 * j + l] = d1 * (q & 0xF) as f32 - m1;
                chunk[64 * j + 32 + l] = d2 * (q >> 4) as f32 - m2;
            }
        }
    }
}

pub fn dequant_row_q5_k(row: &[u8], out: &mut [f32]) {
    for (i, chunk) in out.chunks_exact_mut(QK_K).enumerate() {
        let b = &row[i * Q5_K_SIZE..(i + 1) * Q5_K_SIZE];
        let d = read_f16(b, 0);
        let dmin = read_f16(b, 2);
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        for j in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let m1 = dmin * m1 as f32;
            let m2 = dmin * m2 as f32;
            let u1 = 1u8 << (2 * j);
            let u2 = 2u8 << (2 * j);
            for l in 0..32 {
                let q = qs[32 * j + l];
                let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                chunk[64 * j + l] = d1 * ((q & 0xF) + h1) as f32 - m1;
                chunk[64 * j + 32 + l] = d2 * ((q >> 4) + h2) as f32 - m2;
            }
        }
    }
}

pub fn dequant_row_q6_k(row: &[u8], out: &mut [f32]) {
    for (i, chunk) in out.chunks_exact_mut(QK_K).enumerate() {
        let b = &row[i * Q6_K_SIZE..(i + 1) * Q6_K_SIZE];
        let d = read_f16(b, 208);
        for n in 0..2 {
            let ql = &b[64 * n..64 * n + 64];
            let qh = &b[128 + 32 * n..128 + 32 * n + 32];
            let sc = &b[192 + 8 * n..192 + 8 * n + 8];
            let y = &mut chunk[128 * n..128 * n + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l]) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

/// Dequantize one row of `kind` into `out`.
pub fn dequant_row(kind: Kind, row: &[u8], out: &mut [f32]) {
    match kind {
        Kind::F32 => dequant_row_f32(row, out),
        Kind::F16 => dequant_row_f16(row, out),
        Kind::BF16 => dequant_row_bf16(row, out),
        Kind::Q4_0 => dequant_row_q4_0(row, out),
        Kind::Q8_0 => dequant_row_q8_0(row, out),
        Kind::Q4_K => dequant_row_q4_k(row, out),
        Kind::Q5_K => dequant_row_q5_k(row, out),
        Kind::Q6_K => dequant_row_q6_k(row, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 65504.0, 1e-5, 6.1e-5, 3.14159, -0.333, 1e-7] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {h:#x} -> {back}");
        }
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(1e-8), 0);
    }

    #[test]
    fn q8_0_activation() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 7.0).collect();
        let mut out = Vec::new();
        quantize_q8_0(&x, &mut out);
        assert_eq!(out.len(), 2);
        for (i, blk) in out.iter().enumerate() {
            for j in 0..32 {
                let v = blk.qs[j] as f32 * blk.d;
                assert!((v - x[i * 32 + j]).abs() < 0.05);
            }
        }
    }

    #[test]
    fn q8_k_activation_bsums() {
        let x: Vec<f32> = (0..256).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
        let mut out = Vec::new();
        quantize_q8_k(&x, &mut out);
        let blk = &out[0];
        for j in 0..16 {
            let s: i32 = blk.qs[j * 16..j * 16 + 16].iter().map(|&q| q as i32).sum();
            assert_eq!(s, blk.bsums[j] as i32);
        }
    }
}
