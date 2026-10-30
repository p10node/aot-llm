// Metal compute kernels of the aot-llm runtime (see metal.rs for the host side).
//
// Weights are read in their GGUF quantized layouts straight from the
// executable's weight section; activations are f32. One simdgroup (32 lanes)
// computes NR output rows of a matrix product at a time: every lane owns a
// slice of the input vector, dequantizes the matching bytes of each row and
// accumulates a partial dot product; `simd_sum` joins the lanes.

#include <metal_stdlib>
using namespace metal;

struct MmArgs {
    uint rows;        // output rows (weight matrix rows)
    uint cols;        // weight matrix columns = input length
    uint m;           // number of input vectors (tokens)
    uint accumulate;  // 1: y += W x instead of y = W x
    uint x_stride;    // floats between consecutive input vectors
    uint y_stride;    // floats between consecutive output vectors
    uint rope;        // 1: rotate output pairs with the (cos, sin) table (q / k projections)
    uint head_dim;    // for rope: output row r is dimension r % head_dim of its head
    uint rot_dim;     // for rope: dimensions >= rot_dim are left alone
    uint bias;        // 1: add bias[row] to every output row (projection biases)
    uint gelu;        // gate kernel: 1 = gelu(gate) * up (Gemma), 0 = silu(gate) * up
};
/// Three matrix-vector products sharing one input (the q / k / v projections)
/// in one dispatch: the simdgroups are spread over the concatenated rows.
struct Mv3Args {
    uint cols;
    uint rows0, rows1, rows2;
    uint rope;        // bit i: rotate the output pairs of matrix i (Llama q / k)
    uint head_dim, rot_dim;
    uint bias;        // bit i: add bias_i[row] to matrix i
};
struct NormArgs { uint n; uint m; float eps; uint accumulate; };
struct RopeArgs { uint m; uint n_head; uint n_kv_head; uint head_dim; uint rot_dim; uint q_stride; uint k_stride; uint neox; };
struct AttnArgs { uint m; uint pos0; uint n_head; uint gqa; uint head_dim; uint kv_dim; uint q_stride; uint ctx; float scale; uint window; };
struct SiluArgs { uint n; uint gelu; };

/// GELU, tanh approximation (ggml's `gelu`).
static inline float gelu_f(float v) { return 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))); }

static inline float f16f(ushort u) { return float(as_type<half>(u)); }
static inline ushort u16at(device const uchar* p) { return (ushort)p[0] | ((ushort)p[1] << 8); }
/// The four bytes of `u` as floats (little-endian order).
static inline float4 bytes4(uint u) { return float4(u & 0xFFu, (u >> 8) & 0xFFu, (u >> 16) & 0xFFu, u >> 24); }
static inline float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }

// ---------------------------------------------------------------------------
// Weight formats. Each struct describes how the 32 lanes of a simdgroup split
// one "iteration" of ITER consecutive columns: lane_elem() is the first column
// a lane handles, x_off(lane, k) the column of its k-th float4 of input, W the
// dequantization state loaded from a row, and dot() the lane's partial product.
// ---------------------------------------------------------------------------

/// K-quant lane mapping: 4 lanes per 256-element super-block, each owning
/// one pair of 32-element sub-blocks (64 columns); 8 super-blocks per iteration.
static inline uint kq_lane_elem(uint lane) { return (lane / 4) * 256 + (lane % 4) * 64; }

/// Byte `i` (0..12) of the 12 scale bytes held in words y, z, w of a super-block header.
static inline uint hdr_byte(uint4 h, uint i) {
    const uint word = i < 4 ? h.y : (i < 8 ? h.z : h.w);
    return (word >> (8 * (i % 4))) & 0xFFu;
}

/// Decode the 6-bit (scale, min) pair `j` from a 16-byte Q4_K / Q5_K header
/// (f16 d, f16 dmin, 12 packed scale bytes) loaded as one uint4.
static inline void kq_scale_min_h(uint4 h, uint j, thread float& s, thread float& m) {
    if (j < 4) {
        s = float(hdr_byte(h, j) & 63u);
        m = float(hdr_byte(h, j + 4) & 63u);
    } else {
        s = float((hdr_byte(h, j + 4) & 0xFu) | ((hdr_byte(h, j - 4) >> 6) << 4));
        m = float((hdr_byte(h, j + 4) >> 4) | ((hdr_byte(h, j) >> 6) << 4));
    }
}

/// Decode the 6-bit (scale, min) pair `j` of a Q4_K / Q5_K super-block.
static inline void kq_scale_min(device const uchar* sc, uint j, thread float& s, thread float& m) {
    if (j < 4) {
        s = float(sc[j] & 63u);
        m = float(sc[j + 4] & 63u);
    } else {
        s = float((sc[j + 4] & 0xFu) | ((sc[j - 4] >> 6) << 4));
        m = float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
}

struct Q4K {
    enum : uint { BLOCK = 256, BLOCK_BYTES = 144, ITER = 2048, NX = 16 };
    static uint lane_elem(uint lane) { return kq_lane_elem(lane); }
    static uint x_off(uint lane, uint k) { return kq_lane_elem(lane) + 4 * k; }
    struct W { float d, dmin, s1, m1, s2, m2; uint4 q0, q1; };
    static W load(device const uchar* row, uint elem) {
        device const uchar* b = row + (elem / 256) * 144;
        const uint jp = (elem % 256) / 64;
        const uint4 h = *(device const uint4*)b;
        W w;
        w.d = f16f(ushort(h.x & 0xFFFFu));
        w.dmin = f16f(ushort(h.x >> 16));
        kq_scale_min_h(h, 2 * jp, w.s1, w.m1);
        kq_scale_min_h(h, 2 * jp + 1, w.s2, w.m2);
        device const uint4* q = (device const uint4*)(b + 16 + 32 * jp);
        w.q0 = q[0];
        w.q1 = q[1];
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) {
        float sl = 0.0f, sh = 0.0f, xsl = 0.0f, xsh = 0.0f;
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q0[c];
            sl += metal::dot(bytes4(u & 0x0F0F0F0Fu), xv[c]);
            sh += metal::dot(bytes4((u >> 4) & 0x0F0F0F0Fu), xv[8 + c]);
            xsl += sum4(xv[c]);
            xsh += sum4(xv[8 + c]);
        }
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q1[c];
            sl += metal::dot(bytes4(u & 0x0F0F0F0Fu), xv[4 + c]);
            sh += metal::dot(bytes4((u >> 4) & 0x0F0F0F0Fu), xv[12 + c]);
            xsl += sum4(xv[4 + c]);
            xsh += sum4(xv[12 + c]);
        }
        return w.d * (w.s1 * sl + w.s2 * sh) - w.dmin * (w.m1 * xsl + w.m2 * xsh);
    }
    static void dequant(thread const W& w, thread float4* out) {
        const float d1 = w.d * w.s1, d2 = w.d * w.s2, m1 = w.dmin * w.m1, m2 = w.dmin * w.m2;
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q0[c];
            out[c] = d1 * bytes4(u & 0x0F0F0F0Fu) - m1;
            out[8 + c] = d2 * bytes4((u >> 4) & 0x0F0F0F0Fu) - m2;
        }
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q1[c];
            out[4 + c] = d1 * bytes4(u & 0x0F0F0F0Fu) - m1;
            out[12 + c] = d2 * bytes4((u >> 4) & 0x0F0F0F0Fu) - m2;
        }
    }
    /// Columns [col, col + 8) of a row as floats (col is a multiple of 8).
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const uchar* b = row + (col / 256) * 144;
        const uint within = col % 256;
        const uint j = within / 32;
        const float d = f16f(u16at(b)), dmin = f16f(u16at(b + 2));
        float sc, mn;
        kq_scale_min(b + 4, j, sc, mn);
        device const uint2* qs = (device const uint2*)(b + 16 + 32 * (j / 2) + within % 32);
        const uint shift = (j & 1) * 4;
        const uint2 q = *qs;
        const float ds = d * sc, dm = dmin * mn;
        const float4 lo = ds * bytes4((q.x >> shift) & 0x0F0F0F0Fu) - dm;
        const float4 hi = ds * bytes4((q.y >> shift) & 0x0F0F0F0Fu) - dm;
        out[0] = lo.x; out[1] = lo.y; out[2] = lo.z; out[3] = lo.w;
        out[4] = hi.x; out[5] = hi.y; out[6] = hi.z; out[7] = hi.w;
    }
};

struct Q5K {
    enum : uint { BLOCK = 256, BLOCK_BYTES = 176, ITER = 2048, NX = 16 };
    static uint lane_elem(uint lane) { return kq_lane_elem(lane); }
    static uint x_off(uint lane, uint k) { return kq_lane_elem(lane) + 4 * k; }
    struct W { float d, dmin, s1, m1, s2, m2; uint j1, j2; uint4 q0, q1, h0, h1; };
    static W load(device const uchar* row, uint elem) {
        device const uchar* b = row + (elem / 256) * 176;
        const uint jp = (elem % 256) / 64;
        const uint4 h = *(device const uint4*)b;
        W w;
        w.d = f16f(ushort(h.x & 0xFFFFu));
        w.dmin = f16f(ushort(h.x >> 16));
        w.j1 = 2 * jp;
        w.j2 = 2 * jp + 1;
        kq_scale_min_h(h, w.j1, w.s1, w.m1);
        kq_scale_min_h(h, w.j2, w.s2, w.m2);
        device const uint4* qh = (device const uint4*)(b + 16);
        w.h0 = qh[0];
        w.h1 = qh[1];
        device const uint4* q = (device const uint4*)(b + 48 + 32 * jp);
        w.q0 = q[0];
        w.q1 = q[1];
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) {
        float sl = 0.0f, sh = 0.0f, xsl = 0.0f, xsh = 0.0f;
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q0[c], h = w.h0[c];
            const uint lo = (u & 0x0F0F0F0Fu) | (((h >> w.j1) & 0x01010101u) << 4);
            const uint hi = ((u >> 4) & 0x0F0F0F0Fu) | (((h >> w.j2) & 0x01010101u) << 4);
            sl += metal::dot(bytes4(lo), xv[c]);
            sh += metal::dot(bytes4(hi), xv[8 + c]);
            xsl += sum4(xv[c]);
            xsh += sum4(xv[8 + c]);
        }
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q1[c], h = w.h1[c];
            const uint lo = (u & 0x0F0F0F0Fu) | (((h >> w.j1) & 0x01010101u) << 4);
            const uint hi = ((u >> 4) & 0x0F0F0F0Fu) | (((h >> w.j2) & 0x01010101u) << 4);
            sl += metal::dot(bytes4(lo), xv[4 + c]);
            sh += metal::dot(bytes4(hi), xv[12 + c]);
            xsl += sum4(xv[4 + c]);
            xsh += sum4(xv[12 + c]);
        }
        return w.d * (w.s1 * sl + w.s2 * sh) - w.dmin * (w.m1 * xsl + w.m2 * xsh);
    }
    static void dequant(thread const W& w, thread float4* out) {
        const float d1 = w.d * w.s1, d2 = w.d * w.s2, m1 = w.dmin * w.m1, m2 = w.dmin * w.m2;
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q0[c], h = w.h0[c];
            out[c] = d1 * bytes4((u & 0x0F0F0F0Fu) | (((h >> w.j1) & 0x01010101u) << 4)) - m1;
            out[8 + c] = d2 * bytes4(((u >> 4) & 0x0F0F0F0Fu) | (((h >> w.j2) & 0x01010101u) << 4)) - m2;
        }
        for (uint c = 0; c < 4; c++) {
            const uint u = w.q1[c], h = w.h1[c];
            out[4 + c] = d1 * bytes4((u & 0x0F0F0F0Fu) | (((h >> w.j1) & 0x01010101u) << 4)) - m1;
            out[12 + c] = d2 * bytes4(((u >> 4) & 0x0F0F0F0Fu) | (((h >> w.j2) & 0x01010101u) << 4)) - m2;
        }
    }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const uchar* b = row + (col / 256) * 176;
        const uint within = col % 256;
        const uint j = within / 32;
        const uint l0 = within % 32;
        const float d = f16f(u16at(b)), dmin = f16f(u16at(b + 2));
        float sc, mn;
        kq_scale_min(b + 4, j, sc, mn);
        const uint2 h = *(device const uint2*)(b + 16 + l0);
        const uint2 q = *(device const uint2*)(b + 48 + 32 * (j / 2) + l0);
        const uint shift = (j & 1) * 4;
        const float ds = d * sc, dm = dmin * mn;
        const float4 lo = ds * bytes4(((q.x >> shift) & 0x0F0F0F0Fu) | (((h.x >> j) & 0x01010101u) << 4)) - dm;
        const float4 hi = ds * bytes4(((q.y >> shift) & 0x0F0F0F0Fu) | (((h.y >> j) & 0x01010101u) << 4)) - dm;
        out[0] = lo.x; out[1] = lo.y; out[2] = lo.z; out[3] = lo.w;
        out[4] = hi.x; out[5] = hi.y; out[6] = hi.z; out[7] = hi.w;
    }
};

/// Q6_K: a lane owns 16 of the 32 `l` positions of one 128-element half
/// (n = jp / 2, l in [16 * (jp % 2), +16)), i.e. four runs of 16 columns at
/// 128 n + l + {0, 32, 64, 96}. Blocks are 210 bytes, so rows are only
/// 2-byte aligned and the quantized bytes are loaded as packed 4-byte groups.
struct Q6K {
    enum : uint { BLOCK = 256, BLOCK_BYTES = 210, ITER = 2048, NX = 16 };
    static uint lane_elem(uint lane) { return kq_lane_elem(lane); }
    static uint x_off(uint lane, uint k) {
        const uint jp = lane % 4;
        return (lane / 4) * 256 + 128 * (jp / 2) + 16 * (jp % 2) + 32 * (k / 4) + 4 * (k % 4);
    }
    struct W { float d, sc0, sc2, sc4, sc6; uchar4 ql[4], qlh[4], qh[4]; };
    static W load(device const uchar* row, uint elem) {
        device const uchar* b = row + (elem / 256) * 210;
        const uint jp = (elem % 256) / 64;
        const uint n = jp / 2;
        const uint lh = 16 * (jp % 2);
        const uint is = jp % 2;
        W w;
        w.d = f16f(u16at(b + 208));
        device const char* sc = (device const char*)(b + 192 + 8 * n);
        w.sc0 = float(sc[is]);
        w.sc2 = float(sc[is + 2]);
        w.sc4 = float(sc[is + 4]);
        w.sc6 = float(sc[is + 6]);
        device const packed_uchar4* pl = (device const packed_uchar4*)(b + 64 * n + lh);
        device const packed_uchar4* plh = (device const packed_uchar4*)(b + 64 * n + 32 + lh);
        device const packed_uchar4* ph = (device const packed_uchar4*)(b + 128 + 32 * n + lh);
        for (uint i = 0; i < 4; i++) {
            w.ql[i] = pl[i];
            w.qlh[i] = plh[i];
            w.qh[i] = ph[i];
        }
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) {
        float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
        for (uint i = 0; i < 4; i++) {
            const uchar4 lo = w.ql[i], hi = w.qlh[i], h = w.qh[i];
            const float4 q1 = float4(int4((lo & 0xF) | ((h & 3) << 4))) - 32.0f;
            const float4 q2 = float4(int4((hi & 0xF) | (((h >> 2) & 3) << 4))) - 32.0f;
            const float4 q3 = float4(int4((lo >> 4) | (((h >> 4) & 3) << 4))) - 32.0f;
            const float4 q4 = float4(int4((hi >> 4) | (((h >> 6) & 3) << 4))) - 32.0f;
            s1 += metal::dot(q1, xv[i]);
            s2 += metal::dot(q2, xv[4 + i]);
            s3 += metal::dot(q3, xv[8 + i]);
            s4 += metal::dot(q4, xv[12 + i]);
        }
        return w.d * (w.sc0 * s1 + w.sc2 * s2 + w.sc4 * s3 + w.sc6 * s4);
    }
    static void dequant(thread const W& w, thread float4* out) {
        const float d0 = w.d * w.sc0, d2 = w.d * w.sc2, d4 = w.d * w.sc4, d6 = w.d * w.sc6;
        for (uint i = 0; i < 4; i++) {
            const uchar4 lo = w.ql[i], hi = w.qlh[i], h = w.qh[i];
            out[i] = d0 * (float4(int4((lo & 0xF) | ((h & 3) << 4))) - 32.0f);
            out[4 + i] = d2 * (float4(int4((hi & 0xF) | (((h >> 2) & 3) << 4))) - 32.0f);
            out[8 + i] = d4 * (float4(int4((lo >> 4) | (((h >> 4) & 3) << 4))) - 32.0f);
            out[12 + i] = d6 * (float4(int4((hi >> 4) | (((h >> 6) & 3) << 4))) - 32.0f);
        }
    }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const uchar* b = row + (col / 256) * 210;
        const uint within = col % 256;
        const uint n = within / 128;
        const uint r = within % 128;
        const uint quarter = r / 32;
        const uint l = r % 32;
        device const packed_uchar4* pl = (device const packed_uchar4*)(b + 64 * n + l);
        device const packed_uchar4* plh = (device const packed_uchar4*)(b + 64 * n + 32 + l);
        device const packed_uchar4* ph = (device const packed_uchar4*)(b + 128 + 32 * n + l);
        device const char* sc = (device const char*)(b + 192 + 8 * n);
        const float scale = f16f(u16at(b + 208)) * float(sc[l / 16 + 2 * quarter]);
        // The quarter selects which nibble of ql and which 2 bits of qh form the value.
        const uint hs = 2 * quarter;
        const bool high = quarter >= 2;
        for (uint i = 0; i < 2; i++) {
            const uchar4 lo = pl[i], hi = plh[i], h = ph[i];
            const uchar4 base = (quarter & 1) ? hi : lo;
            const uchar4 nib = high ? (base >> 4) : (base & 0xF);
            const float4 q = float4(int4(nib | (((h >> hs) & 3) << 4))) - 32.0f;
            const float4 v = scale * q;
            out[4 * i] = v.x; out[4 * i + 1] = v.y; out[4 * i + 2] = v.z; out[4 * i + 3] = v.w;
        }
    }
};

/// Q4_0: one 32-element block (18 bytes) per lane and iteration.
struct Q40 {
    enum : uint { BLOCK = 32, BLOCK_BYTES = 18, ITER = 1024, NX = 8 };
    static uint lane_elem(uint lane) { return lane * 32; }
    static uint x_off(uint lane, uint k) { return lane * 32 + 4 * k; }
    struct W { float d; uchar4 q[4]; };
    static W load(device const uchar* row, uint elem) {
        device const uchar* b = row + (elem / 32) * 18;
        W w;
        w.d = f16f(u16at(b));
        device const packed_uchar4* q = (device const packed_uchar4*)(b + 2);
        for (uint i = 0; i < 4; i++) w.q[i] = q[i];
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) {
        float s = 0.0f, xs = 0.0f;
        for (uint i = 0; i < 4; i++) {
            s += metal::dot(float4(w.q[i] & 0xF), xv[i]) + metal::dot(float4(w.q[i] >> 4), xv[4 + i]);
            xs += sum4(xv[i]) + sum4(xv[4 + i]);
        }
        return w.d * (s - 8.0f * xs);
    }
    static void dequant(thread const W& w, thread float4* out) {
        for (uint i = 0; i < 4; i++) {
            out[i] = w.d * (float4(w.q[i] & 0xF) - 8.0f);
            out[4 + i] = w.d * (float4(w.q[i] >> 4) - 8.0f);
        }
    }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const uchar* b = row + (col / 32) * 18;
        const uint within = col % 32;
        const float d = f16f(u16at(b));
        device const packed_uchar4* qs = (device const packed_uchar4*)(b + 2 + (within < 16 ? within : within - 16));
        for (uint i = 0; i < 2; i++) {
            const uchar4 q = qs[i];
            const float4 v = d * (float4(within < 16 ? (q & 0xF) : (q >> 4)) - 8.0f);
            out[4 * i] = v.x; out[4 * i + 1] = v.y; out[4 * i + 2] = v.z; out[4 * i + 3] = v.w;
        }
    }
};

/// Q5_0: one 32-element block (22 bytes: f16 d, u32 of fifth bits, 16
/// nibble bytes) per lane and iteration.
struct Q50 {
    enum : uint { BLOCK = 32, BLOCK_BYTES = 22, ITER = 1024, NX = 8 };
    static uint lane_elem(uint lane) { return lane * 32; }
    static uint x_off(uint lane, uint k) { return lane * 32 + 4 * k; }
    struct W { float d; uint qh; uchar4 q[4]; };
    static W load(device const uchar* row, uint elem) {
        device const uchar* b = row + (elem / 32) * 22;
        W w;
        w.d = f16f(u16at(b));
        w.qh = (uint)b[2] | ((uint)b[3] << 8) | ((uint)b[4] << 16) | ((uint)b[5] << 24);
        device const packed_uchar4* q = (device const packed_uchar4*)(b + 6);
        for (uint i = 0; i < 4; i++) w.q[i] = q[i];
        return w;
    }
    /// Elements 4i..4i+3 (low nibbles, bits 4i..) and 16+4i.. (high nibbles, bits 16+4i..), still offset by 16.
    static void elems(thread const W& w, uint i, thread float4& lo, thread float4& hi) {
        const uint4 sh = uint4(0, 1, 2, 3) + 4 * i;
        const uint4 hl = ((uint4(w.qh) >> sh) & 1u) << 4;
        const uint4 hh = ((uint4(w.qh) >> (sh + 16)) & 1u) << 4;
        lo = float4(uint4(w.q[i] & 0xF) | hl);
        hi = float4(uint4(w.q[i] >> 4) | hh);
    }
    static float dot(thread const W& w, thread const float4* xv) {
        float s = 0.0f, xs = 0.0f;
        for (uint i = 0; i < 4; i++) {
            float4 lo, hi;
            elems(w, i, lo, hi);
            s += metal::dot(lo, xv[i]) + metal::dot(hi, xv[4 + i]);
            xs += sum4(xv[i]) + sum4(xv[4 + i]);
        }
        return w.d * (s - 16.0f * xs);
    }
    static void dequant(thread const W& w, thread float4* out) {
        for (uint i = 0; i < 4; i++) {
            float4 lo, hi;
            elems(w, i, lo, hi);
            out[i] = w.d * (lo - 16.0f);
            out[4 + i] = w.d * (hi - 16.0f);
        }
    }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        const W w = load(row, col);
        const uint within = col % 32;
        for (uint i = 0; i < 2; i++) {
            const uint k = (within < 16 ? within : within - 16) / 4 + i;
            float4 lo, hi;
            elems(w, k, lo, hi);
            const float4 v = w.d * ((within < 16 ? lo : hi) - 16.0f);
            out[4 * i] = v.x; out[4 * i + 1] = v.y; out[4 * i + 2] = v.z; out[4 * i + 3] = v.w;
        }
    }
};

/// Q8_0: one 32-element block (34 bytes) per lane and iteration.
struct Q80 {
    enum : uint { BLOCK = 32, BLOCK_BYTES = 34, ITER = 1024, NX = 8 };
    static uint lane_elem(uint lane) { return lane * 32; }
    static uint x_off(uint lane, uint k) { return lane * 32 + 4 * k; }
    struct W { float d; char4 q[8]; };
    static W load(device const uchar* row, uint elem) {
        device const uchar* b = row + (elem / 32) * 34;
        W w;
        w.d = f16f(u16at(b));
        device const packed_char4* q = (device const packed_char4*)(b + 2);
        for (uint i = 0; i < 8; i++) w.q[i] = q[i];
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) {
        float s = 0.0f;
        for (uint i = 0; i < 8; i++) s += metal::dot(float4(w.q[i]), xv[i]);
        return w.d * s;
    }
    static void dequant(thread const W& w, thread float4* out) {
        for (uint i = 0; i < 8; i++) out[i] = w.d * float4(w.q[i]);
    }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const uchar* b = row + (col / 32) * 34;
        const float d = f16f(u16at(b));
        device const packed_char4* q = (device const packed_char4*)(b + 2 + col % 32);
        for (uint i = 0; i < 2; i++) {
            const float4 v = d * float4(q[i]);
            out[4 * i] = v.x; out[4 * i + 1] = v.y; out[4 * i + 2] = v.z; out[4 * i + 3] = v.w;
        }
    }
};

/// Float formats: four columns per lane and iteration.
struct F16W {
    enum : uint { BLOCK = 1, BLOCK_BYTES = 2, ITER = 128, NX = 1 };
    static uint lane_elem(uint lane) { return lane * 4; }
    static uint x_off(uint lane, uint) { return lane * 4; }
    struct W { float4 v; };
    static W load(device const uchar* row, uint elem) {
        W w;
        w.v = float4(*(device const half4*)(row + elem * 2));
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) { return metal::dot(w.v, xv[0]); }
    static void dequant(thread const W& w, thread float4* out) { out[0] = w.v; }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const half* h = (device const half*)(row + col * 2);
        for (uint i = 0; i < 8; i++) out[i] = float(h[i]);
    }
};

struct BF16W {
    enum : uint { BLOCK = 1, BLOCK_BYTES = 2, ITER = 128, NX = 1 };
    static uint lane_elem(uint lane) { return lane * 4; }
    static uint x_off(uint lane, uint) { return lane * 4; }
    struct W { float4 v; };
    static W load(device const uchar* row, uint elem) {
        W w;
        const ushort4 u = *(device const ushort4*)(row + elem * 2);
        w.v = as_type<float4>(uint4(u) << 16);
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) { return metal::dot(w.v, xv[0]); }
    static void dequant(thread const W& w, thread float4* out) { out[0] = w.v; }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const ushort* u = (device const ushort*)(row + col * 2);
        for (uint i = 0; i < 8; i++) out[i] = as_type<float>(uint(u[i]) << 16);
    }
};

struct F32W {
    enum : uint { BLOCK = 1, BLOCK_BYTES = 4, ITER = 128, NX = 1 };
    static uint lane_elem(uint lane) { return lane * 4; }
    static uint x_off(uint lane, uint) { return lane * 4; }
    struct W { float4 v; };
    static W load(device const uchar* row, uint elem) {
        W w;
        w.v = *(device const float4*)(row + elem * 4);
        return w;
    }
    static float dot(thread const W& w, thread const float4* xv) { return metal::dot(w.v, xv[0]); }
    static void dequant(thread const W& w, thread float4* out) { out[0] = w.v; }
    static void dequant8(device const uchar* row, uint col, thread float* out) {
        device const float* f = (device const float*)(row + col * 4);
        for (uint i = 0; i < 8; i++) out[i] = f[i];
    }
};

// ---------------------------------------------------------------------------
// Matrix-vector product, y[row] (+)= sum_c W[row][c] * x[c].
//
// Threadgroup = 4 simdgroups = 4 * NR rows. NR trades parallelism (small
// matrices want many simdgroups) against reuse of the input (every simdgroup
// streams the whole vector once). With `rope`, lane 0 rotates each output
// pair in place before writing it: the q / k projections then land in their
// buffers (the KV cache for k) already position-encoded.
//
// The kernel is bound by the vector loads (every lane reads 256 bytes of x
// per iteration), so nothing else is folded into it: an attempt to apply
// RMSNorm on the fly (a second 8 KB vector read per simdgroup) made the
// products 2-3x slower than a separate 12 us RMSNorm dispatch.
// ---------------------------------------------------------------------------
template <typename Q, int NR>
kernel void kernel_mv(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MmArgs& a [[buffer(3)]],
    device const float* cs [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = (tgpig.x * 4 + sgitg) * NR;
    if (row0 >= a.rows) return;
    const uint row_bytes = a.cols / Q::BLOCK * Q::BLOCK_BYTES;
    const uint lane_elem = Q::lane_elem(tiisg);
    float sums[NR];
    for (int i = 0; i < NR; i++) sums[i] = 0.0f;
    for (uint base = 0; base + lane_elem < a.cols; base += Q::ITER) {
        const uint elem = base + lane_elem;
        typename Q::W wv[NR];
        for (int i = 0; i < NR; i++) wv[i] = Q::load(w + min(row0 + i, a.rows - 1) * row_bytes, elem);
        float4 xv[Q::NX];
        for (uint k = 0; k < Q::NX; k++) xv[k] = *(device const float4*)(x + base + Q::x_off(tiisg, k));
        for (int i = 0; i < NR; i++) sums[i] += Q::dot(wv[i], xv);
    }
    float v[NR];
    for (int i = 0; i < NR; i++) v[i] = simd_sum(sums[i]);
    if (tiisg != 0) return;
    if (a.bias) {
        for (int i = 0; i < NR; i++) v[i] += bias[min(row0 + i, a.rows - 1)];
    }
    if (a.rope) {
        for (int i = 0; i + 1 < NR; i += 2) {
            const uint dim = (row0 + i) % a.head_dim;
            if (dim < a.rot_dim) {
                const float c = cs[dim], sn = cs[dim + 1];
                const float x0 = v[i], x1 = v[i + 1];
                v[i] = x0 * c - x1 * sn;
                v[i + 1] = x0 * sn + x1 * c;
            }
        }
    }
    for (int i = 0; i < NR; i++) {
        if (row0 + i < a.rows) {
            device float* yp = y + row0 + i;
            *yp = a.accumulate ? (*yp + v[i]) : v[i];
        }
    }
}

// Fused q / k / v: like kernel_mv over three matrices of the same format.
// Each simdgroup's NR rows lie inside one matrix (the host checks that every
// row count is a multiple of NR), so matrix selection is per simdgroup.
template <typename Q, int NR>
kernel void kernel_mv3(
    device const uchar* w0 [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y0 [[buffer(2)]],
    constant Mv3Args& a [[buffer(3)]],
    device const float* cs [[buffer(4)]],
    device const float* b0 [[buffer(5)]],
    device const uchar* w1 [[buffer(6)]],
    device const uchar* w2 [[buffer(7)]],
    device float* y1 [[buffer(8)]],
    device float* y2 [[buffer(9)]],
    device const float* b1 [[buffer(10)]],
    device const float* b2 [[buffer(11)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint row0 = (tgpig.x * 4 + sgitg) * NR;
    if (row0 >= a.rows0 + a.rows1 + a.rows2) return;
    uint mat = 0;
    uint rows = a.rows0;
    device const uchar* w = w0;
    device float* y = y0;
    device const float* bias = b0;
    if (row0 >= a.rows0 + a.rows1) {
        mat = 2; row0 -= a.rows0 + a.rows1; rows = a.rows2; w = w2; y = y2; bias = b2;
    } else if (row0 >= a.rows0) {
        mat = 1; row0 -= a.rows0; rows = a.rows1; w = w1; y = y1; bias = b1;
    }
    const uint row_bytes = a.cols / Q::BLOCK * Q::BLOCK_BYTES;
    const uint lane_elem = Q::lane_elem(tiisg);
    float sums[NR];
    for (int i = 0; i < NR; i++) sums[i] = 0.0f;
    for (uint base = 0; base + lane_elem < a.cols; base += Q::ITER) {
        const uint elem = base + lane_elem;
        typename Q::W wv[NR];
        for (int i = 0; i < NR; i++) wv[i] = Q::load(w + min(row0 + i, rows - 1) * row_bytes, elem);
        float4 xv[Q::NX];
        for (uint k = 0; k < Q::NX; k++) xv[k] = *(device const float4*)(x + base + Q::x_off(tiisg, k));
        for (int i = 0; i < NR; i++) sums[i] += Q::dot(wv[i], xv);
    }
    float v[NR];
    for (int i = 0; i < NR; i++) v[i] = simd_sum(sums[i]);
    if (tiisg != 0) return;
    if ((a.bias >> mat) & 1u) {
        for (int i = 0; i < NR; i++) v[i] += bias[min(row0 + i, rows - 1)];
    }
    if ((a.rope >> mat) & 1u) {
        for (int i = 0; i + 1 < NR; i += 2) {
            const uint dim = (row0 + i) % a.head_dim;
            if (dim < a.rot_dim) {
                const float c = cs[dim], sn = cs[dim + 1];
                const float x0 = v[i], x1 = v[i + 1];
                v[i] = x0 * c - x1 * sn;
                v[i + 1] = x0 * sn + x1 * c;
            }
        }
    }
    for (int i = 0; i < NR; i++) {
        if (row0 + i < rows) y[row0 + i] = v[i];
    }
}

// ---------------------------------------------------------------------------
// Batched product, y[t][row] (+)= sum_c W[row][c] * x[t][c] for t < m
// (prompt processing): a tiled GEMM on the simdgroup matrix units.
//
// A threadgroup (4 simdgroups) owns a 32 x 32 tile of (rows x inputs). For
// every chunk of 32 columns the threads dequantize the 32 x 32 weight tile
// and copy the 32 x 32 input tile into threadgroup memory, then each
// simdgroup multiplies its 16 x 16 quarter with 8 x 8 matrix instructions.
// Weights are decoded once per 32 inputs and inputs are read once per 32
// rows, so the product is compute bound like a dense GEMM.
// ---------------------------------------------------------------------------
constant constexpr uint MM_T = 32;

template <typename Q>
kernel void kernel_mm(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MmArgs& a [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sa[MM_T * MM_T];  // weights [row][k], later the output tile [row][input]
    threadgroup float sb[MM_T * MM_T];  // inputs [input][k]
    const uint row0 = tgpig.x * MM_T;
    const uint in0 = tgpig.y * MM_T;
    if (row0 >= a.rows || in0 >= a.m) return;
    const uint row_bytes = a.cols / Q::BLOCK * Q::BLOCK_BYTES;
    // Tile filling: thread -> (row or input fr, 8 columns from fc).
    const uint fr = tid / 4;
    const uint fc = (tid % 4) * 8;
    device const uchar* wr = w + min(row0 + fr, a.rows - 1) * row_bytes;
    const bool in_ok = in0 + fr < a.m;
    device const float* xr = x + min(in0 + fr, a.m - 1) * a.x_stride;
    // Simdgroup quarter: rows sm.., inputs sn.. (16 each = 2 x 2 tiles of 8 x 8).
    const uint sm = (sgitg / 2) * 16;
    const uint sn = (sgitg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint k0 = 0; k0 < a.cols; k0 += MM_T) {
        float av[8];
        Q::dequant8(wr, k0 + fc, av);
        for (uint i = 0; i < 8; i++) sa[fr * MM_T + fc + i] = av[i];
        if (in_ok) {
            const float4 b0 = *(device const float4*)(xr + k0 + fc);
            const float4 b1 = *(device const float4*)(xr + k0 + fc + 4);
            *(threadgroup float4*)(sb + fr * MM_T + fc) = b0;
            *(threadgroup float4*)(sb + fr * MM_T + fc + 4) = b1;
        } else {
            *(threadgroup float4*)(sb + fr * MM_T + fc) = float4(0.0f);
            *(threadgroup float4*)(sb + fr * MM_T + fc + 4) = float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < MM_T; kk += 8) {
            simdgroup_float8x8 ma[2], mb[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(ma[i], sa + (sm + 8 * i) * MM_T + kk, MM_T);
            // sb is [input][k]; the product needs B as [k][input].
            for (uint j = 0; j < 2; j++) simdgroup_load(mb[j], sb + (sn + 8 * j) * MM_T + kk, MM_T, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Stage the 32 x 32 output tile through sa, then write it with bounds checks.
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], sa + (sm + 8 * i) * MM_T + sn + 8 * j, MM_T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = row0 + fr;
    if (row < a.rows) {
        const float b = a.bias ? bias[row] : 0.0f;
        for (uint i = 0; i < 8; i++) {
            const uint t = in0 + fc + i;
            if (t < a.m) {
                device float* yp = y + t * a.y_stride + row;
                const float v = sa[fr * MM_T + fc + i] + b;
                *yp = a.accumulate ? (*yp + v) : v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused feed-forward gate: y[row] = silu(G[row] . x) * (U[row] . x) for two
// matrices of the same format and shape, one row of each per simdgroup (the
// same load per lane as two rows of one matrix, the best point for Q4_K).
// ---------------------------------------------------------------------------
template <typename Q>
kernel void kernel_mv_gate(
    device const uchar* wg [[buffer(0)]],
    device const uchar* wu [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant MmArgs& a [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgpig.x * 4 + sgitg;
    if (row >= a.rows) return;
    const uint row_bytes = a.cols / Q::BLOCK * Q::BLOCK_BYTES;
    const uint lane_elem = Q::lane_elem(tiisg);
    device const uchar* gr = wg + row * row_bytes;
    device const uchar* ur = wu + row * row_bytes;
    float sg = 0.0f, su = 0.0f;
    for (uint base = 0; base + lane_elem < a.cols; base += Q::ITER) {
        const uint elem = base + lane_elem;
        const typename Q::W gv = Q::load(gr, elem);
        const typename Q::W uv = Q::load(ur, elem);
        float4 xv[Q::NX];
        for (uint k = 0; k < Q::NX; k++) xv[k] = *(device const float4*)(x + base + Q::x_off(tiisg, k));
        sg += Q::dot(gv, xv);
        su += Q::dot(uv, xv);
    }
    const float g = simd_sum(sg);
    const float u = simd_sum(su);
    if (tiisg == 0) y[row] = (a.gelu ? gelu_f(g) : g / (1.0f + exp(-g))) * u;
}

typedef decltype(kernel_mv<Q4K, 4>) mv_t;
typedef decltype(kernel_mv3<Q4K, 4>) mv3_t;
typedef decltype(kernel_mm<Q4K>) mm_t;
typedef decltype(kernel_mv_gate<Q4K>) mv_gate_t;

#define MM_KERNELS(Q, name) \
    template [[host_name("mv1_" name)]] kernel mv_t kernel_mv<Q, 1>; \
    template [[host_name("mv2_" name)]] kernel mv_t kernel_mv<Q, 2>; \
    template [[host_name("mv4_" name)]] kernel mv_t kernel_mv<Q, 4>; \
    template [[host_name("mv8_" name)]] kernel mv_t kernel_mv<Q, 8>; \
    template [[host_name("mv3x1_" name)]] kernel mv3_t kernel_mv3<Q, 1>; \
    template [[host_name("mv3x2_" name)]] kernel mv3_t kernel_mv3<Q, 2>; \
    template [[host_name("mv3x4_" name)]] kernel mv3_t kernel_mv3<Q, 4>; \
    template [[host_name("mm_" name)]] kernel mm_t kernel_mm<Q>; \
    template [[host_name("mvgate_" name)]] kernel mv_gate_t kernel_mv_gate<Q>;

MM_KERNELS(F32W, "f32")
MM_KERNELS(F16W, "f16")
MM_KERNELS(BF16W, "bf16")
MM_KERNELS(Q40, "q4_0")
MM_KERNELS(Q50, "q5_0")
MM_KERNELS(Q80, "q8_0")
MM_KERNELS(Q4K, "q4_k")
MM_KERNELS(Q5K, "q5_k")
MM_KERNELS(Q6K, "q6_k")

// ---------------------------------------------------------------------------
// RMSNorm: one threadgroup of 256 threads per row.
// ---------------------------------------------------------------------------
kernel void kernel_rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant NormArgs& a [[buffer(3)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[8];
    if (tgpig >= a.m) return;
    device const float* xr = x + tgpig * a.n;
    device float* yr = y + tgpig * a.n;
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 256) {
        const float v = xr[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (tiisg == 0) part[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8; i++) tot += part[i];
    const float r = 1.0f / sqrt(tot / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += 256) {
        const float v = xr[i] * r * w[i];
        yr[i] = a.accumulate ? (yr[i] + v) : v;
    }
}

// ---------------------------------------------------------------------------
// RoPE over the query heads of q and the key heads of k (both m rows), one
// thread per rotated pair; `cs` holds interleaved (cos, sin) per row.
// ---------------------------------------------------------------------------
kernel void kernel_rope(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* cs [[buffer(2)]],
    constant RopeArgs& a [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_rot = a.rot_dim / 2;
    const uint heads = a.n_head + a.n_kv_head;
    if (gid >= a.m * heads * half_rot) return;
    const uint i = gid % half_rot;
    const uint rest = gid / half_rot;
    const uint h = rest % heads;
    const uint t = rest / heads;
    device float* v = (h < a.n_head) ? (q + t * a.q_stride + h * a.head_dim) : (k + t * a.k_stride + (h - a.n_head) * a.head_dim);
    const float c = cs[t * a.rot_dim + 2 * i];
    const float s = cs[t * a.rot_dim + 2 * i + 1];
    // Adjacent pairs (Llama) or (i, i + rot_dim / 2) pairs (NEOX: Qwen).
    const uint i0 = a.neox ? i : 2 * i;
    const uint i1 = a.neox ? i + half_rot : 2 * i + 1;
    const float x0 = v[i0];
    const float x1 = v[i1];
    v[i0] = x0 * c - x1 * s;
    v[i1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------------------
// Causal attention of m query rows (positions pos0 + t) over the cache of one
// layer. One threadgroup (4 simdgroups) per (row, head); the simdgroups take
// turns over blocks of 32 cached positions with an online softmax (running
// max / sum) and are merged through threadgroup memory. Each lane owns
// head_dim / 32 output dimensions. head_dim must be a multiple of 32, at most 128.
// ---------------------------------------------------------------------------
kernel void kernel_attention(
    device const float* q [[buffer(0)]],
    device const float* kc [[buffer(1)]],
    device const float* vc [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttnArgs& a [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tq[256];
    threadgroup float tacc[4 * 256];
    threadgroup float tM[4];
    threadgroup float tL[4];
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint hd = a.head_dim;
    const uint dpl = hd / 32;
    const uint kvh = h / a.gqa;
    const uint n = a.pos0 + t + 1;
    // Sliding-window layers see only the last `window` positions.
    const uint start = (a.window != 0 && n > a.window) ? n - a.window : 0;
    device const float* qh = q + t * a.q_stride + h * hd;
    for (uint i = tid; i < hd; i += 128) tq[i] = qh[i] * a.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float L = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint nblocks = (n + 31) / 32;
    for (uint b = start / 32 + sgitg; b < nblocks; b += 4) {
        const uint p = b * 32 + tiisg;
        float s = -INFINITY;
        if (p < n && p >= start) {
            device const float* kr = kc + p * a.kv_dim + kvh * hd;
            float d = 0.0f;
            for (uint i = 0; i < hd; i += 4) {
                const float4 k4 = *(device const float4*)(kr + i);
                d += k4.x * tq[i] + k4.y * tq[i + 1] + k4.z * tq[i + 2] + k4.w * tq[i + 3];
            }
            s = d;
        }
        const float Mn = max(M, simd_max(s));
        const float pe = (p < n && p >= start) ? exp(s - Mn) : 0.0f;
        const float alpha = exp(M - Mn);
        L = L * alpha + simd_sum(pe);
        for (uint i = 0; i < dpl; i++) acc[i] *= alpha;
        // Positions past the end get weight 0 and read a clamped row, so the
        // loads of a block are independent and pipeline instead of branching.
        for (uint j = 0; j < 32; j++) {
            const float pj = simd_shuffle(pe, j);
            device const float* vr = vc + min(b * 32 + j, n - 1) * a.kv_dim + kvh * hd + tiisg * dpl;
            for (uint i = 0; i < dpl; i++) acc[i] += pj * vr[i];
        }
        M = Mn;
    }
    if (tiisg == 0) {
        tM[sgitg] = M;
        tL[sgitg] = L;
    }
    for (uint i = 0; i < dpl; i++) tacc[sgitg * 256 + tiisg * dpl + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float Mg = max(max(tM[0], tM[1]), max(tM[2], tM[3]));
    float sc[4];
    float Lg = 0.0f;
    for (uint g = 0; g < 4; g++) {
        sc[g] = (tM[g] == -INFINITY) ? 0.0f : exp(tM[g] - Mg);
        Lg += tL[g] * sc[g];
    }
    for (uint i = tid; i < hd; i += 128) {
        float o = 0.0f;
        for (uint g = 0; g < 4; g++) o += tacc[g * 256 + i] * sc[g];
        out[t * a.q_stride + h * hd + i] = o / Lg;
    }
}

// ---------------------------------------------------------------------------
// SwiGLU gating: h = silu(h) * up.
// ---------------------------------------------------------------------------
kernel void kernel_silu_mul(
    device float* h [[buffer(0)]],
    device const float* u [[buffer(1)]],
    constant SiluArgs& a [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    const float v = h[gid];
    h[gid] = (a.gelu ? gelu_f(v) : v / (1.0f + exp(-v))) * u[gid];
}
