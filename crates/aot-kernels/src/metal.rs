//! Metal GPU backend (macOS on Apple Silicon).
//!
//! Design, in the spirit of the rest of the runtime:
//!
//! * **No dependencies, nothing linked.** The Objective-C runtime and
//!   `Metal.framework` are opened with `dlopen` on first use and every call
//!   goes through `objc_msgSend`. A binary compiled with `--gpu metal` still
//!   links only `libSystem`, and a run with `--device cpu` never loads Metal.
//! * **Zero-copy weights.** The weights section of the executable (or the
//!   sidecar file) is handed to Metal with `newBufferWithBytesNoCopy`; Apple
//!   Silicon has unified memory, so the GPU reads the same mapped pages the
//!   CPU would. `payload::weights` maps the section `MAP_SHARED` for this.
//! * **Shared activations.** The activation buffers and KV cache of
//!   [`State`] are page-aligned [`Buf`]s wrapped the same way, so the CPU
//!   embedding lookup, the GPU transformer blocks and the CPU sampler all
//!   work on the same memory. Nothing is copied in either direction.
//! * **One command buffer per forward pass.** The generated `forward_gpu`
//!   encodes every operator of a token (8 dispatches per layer: RoPE, the
//!   SwiGLU product and the residual adds are fused into the matrix kernels)
//!   into a single command buffer and waits once; the logits are then
//!   readable from `state.logits`.
//! * **Shader source embedded.** `shaders.metal` is compiled by the system
//!   Metal compiler the first time a binary runs on a machine (~0.3 s) and
//!   served from the OS shader cache afterwards (< 1 ms), so building a model
//!   needs no Metal toolchain.
//!
//! Numerics: the GPU kernels use f32 activations and dequantize weights on the
//! fly (like ggml's Metal path), whereas the CPU path quantizes activations to
//! Q8. Outputs therefore match the CPU path closely but not bit for bit.

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use super::quant::Kind;
use super::state::{Buf, Dims, State, Tensor, PAGE};
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr::null_mut;
use std::time::Instant;

/// The compute kernels; see the file for the lane mapping of each format.
pub const SHADER_SOURCE: &str = include_str!("shaders.metal");

/// Tile edge of the batched matrix kernel (`MM_T` in `shaders.metal`).
const MM_TILE: usize = 32;
/// Threads per threadgroup of the matrix kernels (4 simdgroups).
const MM_THREADS: usize = 128;

type Id = *mut c_void;

extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: i32 = 0x2;
const RTLD_GLOBAL: i32 = 0x8;

#[repr(C)]
#[derive(Clone, Copy)]
struct MTLSize {
    width: usize,
    height: usize,
    depth: usize,
}

const MTL_COMMAND_BUFFER_STATUS_COMPLETED: usize = 4;
/// `MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache`.
const MTL_RESOURCE_SHARED: usize = 0;

unsafe fn sym(handle: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let c = CString::new(name).unwrap();
    let p = dlsym(handle, c.as_ptr());
    if p.is_null() {
        Err(format!("symbol {name} not found"))
    } else {
        Ok(p)
    }
}

/// The Objective-C runtime entry points, resolved with `dlsym` so nothing is
/// linked. Messages are sent by casting `objc_msgSend` to the C signature of
/// each method (the arm64 ABI passes every argument type the same way).
struct Objc {
    msg_send: *const c_void,
    get_class: unsafe extern "C" fn(*const c_char) -> Id,
    register_sel: unsafe extern "C" fn(*const c_char) -> Id,
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
}

impl Objc {
    unsafe fn load() -> Result<Objc, String> {
        let lib = dlopen(c"/usr/lib/libobjc.A.dylib".as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        if lib.is_null() {
            return Err("cannot load the Objective-C runtime".into());
        }
        Ok(Objc {
            msg_send: sym(lib, "objc_msgSend")?,
            get_class: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_char) -> Id>(sym(lib, "objc_getClass")?),
            register_sel: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_char) -> Id>(sym(lib, "sel_registerName")?),
            pool_push: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> *mut c_void>(sym(lib, "objc_autoreleasePoolPush")?),
            pool_pop: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_void)>(sym(lib, "objc_autoreleasePoolPop")?),
        })
    }

    unsafe fn sel(&self, name: &str) -> Id {
        let c = CString::new(name).unwrap();
        (self.register_sel)(c.as_ptr())
    }

    unsafe fn class(&self, name: &str) -> Id {
        let c = CString::new(name).unwrap();
        (self.get_class)(c.as_ptr())
    }

    #[inline]
    unsafe fn send0<R>(&self, recv: Id, sel: Id) -> R {
        let f: unsafe extern "C" fn(Id, Id) -> R = std::mem::transmute(self.msg_send);
        f(recv, sel)
    }

    #[inline]
    unsafe fn send1<A, R>(&self, recv: Id, sel: Id, a: A) -> R {
        let f: unsafe extern "C" fn(Id, Id, A) -> R = std::mem::transmute(self.msg_send);
        f(recv, sel, a)
    }

    #[inline]
    unsafe fn send2<A, B, R>(&self, recv: Id, sel: Id, a: A, b: B) -> R {
        let f: unsafe extern "C" fn(Id, Id, A, B) -> R = std::mem::transmute(self.msg_send);
        f(recv, sel, a, b)
    }

    #[inline]
    unsafe fn send3<A, B, C, R>(&self, recv: Id, sel: Id, a: A, b: B, c: C) -> R {
        let f: unsafe extern "C" fn(Id, Id, A, B, C) -> R = std::mem::transmute(self.msg_send);
        f(recv, sel, a, b, c)
    }

    #[inline]
    unsafe fn send4<A, B, C, D, R>(&self, recv: Id, sel: Id, a: A, b: B, c: C, d: D) -> R {
        let f: unsafe extern "C" fn(Id, Id, A, B, C, D) -> R = std::mem::transmute(self.msg_send);
        f(recv, sel, a, b, c, d)
    }

    unsafe fn nsstring(&self, s: &str) -> Id {
        let c = CString::new(s).unwrap();
        self.send1(self.class("NSString"), self.sel("stringWithUTF8String:"), c.as_ptr())
    }

    unsafe fn to_string(&self, ns: Id) -> String {
        if ns.is_null() {
            return "<nil>".into();
        }
        let p: *const c_char = self.send0(ns, self.sel("UTF8String"));
        if p.is_null() {
            "<nil>".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }

    unsafe fn describe_error(&self, err: Id) -> String {
        if err.is_null() {
            "unknown error".into()
        } else {
            self.to_string(self.send0(err, self.sel("localizedDescription")))
        }
    }
}

/// Selectors used on the encoding path, registered once.
struct Sels {
    command_buffer: Id,
    compute_command_encoder: Id,
    set_pipeline: Id,
    set_buffer: Id,
    set_bytes: Id,
    dispatch_threadgroups: Id,
    end_encoding: Id,
    commit: Id,
    wait_until_completed: Id,
    status: Id,
    error: Id,
}

impl Sels {
    unsafe fn new(o: &Objc) -> Sels {
        Sels {
            command_buffer: o.sel("commandBuffer"),
            compute_command_encoder: o.sel("computeCommandEncoder"),
            set_pipeline: o.sel("setComputePipelineState:"),
            set_buffer: o.sel("setBuffer:offset:atIndex:"),
            set_bytes: o.sel("setBytes:length:atIndex:"),
            dispatch_threadgroups: o.sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            end_encoding: o.sel("endEncoding"),
            commit: o.sel("commit"),
            wait_until_completed: o.sel("waitUntilCompleted"),
            status: o.sel("status"),
            error: o.sel("error"),
        }
    }
}

/// The activation buffers of a [`State`] as seen by the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufId {
    X,
    Xb,
    Xb2,
    Q,
    K,
    V,
    Hb,
    Hb2,
    Logits,
    KCache,
    VCache,
    RopeCs,
    Bx,
    Bxb,
    Bxb2,
    Bq,
    Bk,
    Bv,
    Bhb,
    Bhb2,
    BropeCs,
}
const N_BUFS: usize = 21;

fn state_bufs(s: &State) -> [&Buf<f32>; N_BUFS] {
    [&s.x, &s.xb, &s.xb2, &s.q, &s.k, &s.v, &s.hb, &s.hb2, &s.logits, &s.k_cache, &s.v_cache, &s.rope_cs, &s.bx, &s.bxb, &s.bxb2, &s.bq, &s.bk, &s.bv, &s.bhb, &s.bhb2, &s.brope_cs]
}

/// Index of a weight format in the pipeline tables (matches `KIND_NAMES`).
fn kind_idx(k: Kind) -> usize {
    match k {
        Kind::F32 => 0,
        Kind::F16 => 1,
        Kind::BF16 => 2,
        Kind::Q4_0 => 3,
        Kind::Q8_0 => 4,
        Kind::Q4_K => 5,
        Kind::Q5_K => 6,
        Kind::Q6_K => 7,
        Kind::Q5_0 => 8,
    }
}
const N_KINDS: usize = 9;
const KIND_NAMES: [&str; N_KINDS] = ["f32", "f16", "bf16", "q4_0", "q8_0", "q4_k", "q5_k", "q6_k", "q5_0"];

// Argument blocks, laid out exactly like the structs in `shaders.metal`.
#[repr(C)]
struct MmArgs {
    rows: u32,
    cols: u32,
    m: u32,
    accumulate: u32,
    x_stride: u32,
    y_stride: u32,
    rope: u32,
    head_dim: u32,
    rot_dim: u32,
    bias: u32,
}

/// Fusions available on a matrix-vector product (decode step); see
/// `kernel_mv` in `shaders.metal`.
#[derive(Clone, Copy, Default)]
pub struct MvOpts<'a> {
    /// `dst += W x` instead of `dst = W x` (residual connections).
    pub accumulate: bool,
    /// Rotate the output pairs with this (cos, sin) row (q / k projections,
    /// adjacent-pair layout only; NEOX models use [`Gpu::rope_ex`]).
    pub rope: Option<BufId>,
    /// Add this f32 vector (one value per output row) to the result.
    pub bias: Option<&'a Tensor>,
}
#[repr(C)]
struct NormArgs {
    n: u32,
    m: u32,
    eps: f32,
}
#[repr(C)]
struct RopeArgs {
    m: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
    rot_dim: u32,
    q_stride: u32,
    k_stride: u32,
    neox: u32,
}
#[repr(C)]
struct AttnArgs {
    m: u32,
    pos0: u32,
    n_head: u32,
    gqa: u32,
    head_dim: u32,
    kv_dim: u32,
    q_stride: u32,
    ctx: u32,
    scale: f32,
}
#[repr(C)]
struct SiluArgs {
    n: u32,
}

/// A Metal device with the compiled kernels, the weights mapped without a
/// copy and the activation buffers of one [`State`].
pub struct Gpu {
    objc: Objc,
    s: Sels,
    device: Id,
    queue: Id,
    /// Matrix-vector kernels per format, for 1, 2, 4 and 8 rows per simdgroup.
    pso_mv: [[Id; 4]; N_KINDS],
    pso_mm: [Id; N_KINDS],
    pso_gate: [Id; N_KINDS],
    pso_rmsnorm: Id,
    pso_rope: Id,
    pso_attention: Id,
    pso_silu: Id,
    wbuf: Id,
    bufs: [Id; N_BUFS],
    cb: Id,
    enc: Id,
    pool: *mut c_void,
    /// Device name as reported by Metal.
    pub name: String,
    /// Wall time of [`Gpu::new`] in milliseconds.
    pub init_ms: f64,
    /// Kernel dispatches encoded so far.
    pub dispatches: usize,
    /// Forced rows-per-simdgroup for matrix-vector products (tuning aid).
    mv_rows_override: Option<usize>,
}

impl Gpu {
    /// Open the first Metal device, compile the kernels, and map `weights`
    /// (page aligned, e.g. the executable's weight section) and the buffers
    /// of `state` for the GPU. `state` must outlive the returned value and
    /// must not be reallocated.
    pub fn new(weights: &'static [u8], state: &State, dims: &Dims) -> Result<Gpu, String> {
        let t0 = Instant::now();
        let mv_rows_override = std::env::var_os("AOT_GPU_MV_ROWS").and_then(|v| v.to_str()?.parse::<usize>().ok()).filter(|v| [1, 2, 4, 8].contains(v));
        if !dims.head_dim.is_multiple_of(32) || dims.head_dim > 128 {
            return Err(format!("the Metal attention kernel needs a head size that is a multiple of 32 and at most 128 (model has {})", dims.head_dim));
        }
        if !(weights.as_ptr() as usize).is_multiple_of(PAGE) {
            return Err("the weights are not page aligned; Metal cannot map them without copying".into());
        }
        unsafe {
            let objc = Objc::load()?;
            let metal = dlopen(c"/System/Library/Frameworks/Metal.framework/Metal".as_ptr(), RTLD_NOW | RTLD_GLOBAL);
            if metal.is_null() {
                return Err("cannot load Metal.framework".into());
            }
            let copy_all_devices: unsafe extern "C" fn() -> Id = std::mem::transmute(sym(metal, "MTLCopyAllDevices")?);
            let pool = (objc.pool_push)();
            let result = Self::init(&objc, copy_all_devices, weights, state);
            (objc.pool_pop)(pool);
            let (device, queue, name, psos, wbuf, bufs) = result?;
            let s = Sels::new(&objc);
            Ok(Gpu {
                objc,
                s,
                device,
                queue,
                pso_mv: psos.0,
                pso_mm: psos.1,
                pso_gate: psos.2,
                pso_rmsnorm: psos.3,
                pso_rope: psos.4,
                pso_attention: psos.5,
                pso_silu: psos.6,
                wbuf,
                bufs,
                cb: null_mut(),
                enc: null_mut(),
                pool: null_mut(),
                name,
                init_ms: t0.elapsed().as_secs_f64() * 1e3,
                dispatches: 0,
                mv_rows_override,
            })
        }
    }

    #[allow(clippy::type_complexity)]
    unsafe fn init(o: &Objc, copy_all_devices: unsafe extern "C" fn() -> Id, weights: &'static [u8], state: &State) -> Result<(Id, Id, String, ([[Id; 4]; N_KINDS], [Id; N_KINDS], [Id; N_KINDS], Id, Id, Id, Id), Id, [Id; N_BUFS]), String> {
        // MTLCopyAllDevices is several times faster than
        // MTLCreateSystemDefaultDevice (which talks to the window server).
        let devices = copy_all_devices();
        let count: usize = o.send0(devices, o.sel("count"));
        if devices.is_null() || count == 0 {
            return Err("no Metal device".into());
        }
        let device: Id = o.send1(devices, o.sel("objectAtIndex:"), 0usize);
        let device: Id = o.send0(device, o.sel("retain"));
        let name = o.to_string(o.send0(device, o.sel("name")));
        let queue: Id = o.send0(device, o.sel("newCommandQueue"));
        if queue.is_null() {
            return Err("cannot create a Metal command queue".into());
        }

        // Compile the kernels (served from the system shader cache after the first run).
        let opts: Id = o.send0(o.send0(o.class("MTLCompileOptions"), o.sel("alloc")), o.sel("init"));
        o.send1::<bool, ()>(opts, o.sel("setFastMathEnabled:"), false);
        let mut err: Id = null_mut();
        let lib: Id = o.send3(device, o.sel("newLibraryWithSource:options:error:"), o.nsstring(SHADER_SOURCE), opts, &mut err as *mut Id);
        if lib.is_null() {
            return Err(format!("compiling the Metal kernels: {}", o.describe_error(err)));
        }
        let pipeline = |name: &str| -> Result<Id, String> {
            let f: Id = o.send1(lib, o.sel("newFunctionWithName:"), o.nsstring(name));
            if f.is_null() {
                return Err(format!("Metal kernel {name} not found"));
            }
            let mut err: Id = null_mut();
            let p: Id = o.send2(device, o.sel("newComputePipelineStateWithFunction:error:"), f, &mut err as *mut Id);
            if p.is_null() {
                return Err(format!("creating the pipeline for {name}: {}", o.describe_error(err)));
            }
            Ok(p)
        };
        let mut pso_mv = [[null_mut(); 4]; N_KINDS];
        let mut pso_mm = [null_mut(); N_KINDS];
        let mut pso_gate = [null_mut(); N_KINDS];
        for (i, n) in KIND_NAMES.iter().enumerate() {
            for (j, nr) in [1, 2, 4, 8].iter().enumerate() {
                pso_mv[i][j] = pipeline(&format!("mv{nr}_{n}"))?;
            }
            pso_mm[i] = pipeline(&format!("mm_{n}"))?;
            pso_gate[i] = pipeline(&format!("mvgate_{n}"))?;
        }
        let psos = (pso_mv, pso_mm, pso_gate, pipeline("kernel_rmsnorm")?, pipeline("kernel_rope")?, pipeline("kernel_attention")?, pipeline("kernel_silu_mul")?);

        // Weights and activations, without copying.
        let no_copy = o.sel("newBufferWithBytesNoCopy:length:options:deallocator:");
        let wrap = |ptr: *const u8, bytes: usize| -> Id { o.send4(device, no_copy, ptr as *const c_void, bytes, MTL_RESOURCE_SHARED, null_mut::<c_void>()) };
        let wlen = weights.len().div_ceil(PAGE) * PAGE;
        let wbuf = wrap(weights.as_ptr(), wlen);
        if wbuf.is_null() {
            return Err(format!("Metal refused to map the {} MB weights without copying", weights.len() >> 20));
        }
        let mut bufs = [null_mut(); N_BUFS];
        for (i, b) in state_bufs(state).iter().enumerate() {
            if b.byte_len() == 0 {
                continue;
            }
            bufs[i] = wrap(b.as_ptr() as *const u8, b.byte_len());
            if bufs[i].is_null() {
                return Err("Metal refused to map an activation buffer".into());
            }
        }
        Ok((device, queue, name, psos, wbuf, bufs))
    }

    /// Start encoding a command buffer. Every operator issued until
    /// [`Gpu::finish`] runs in order on the GPU.
    pub fn begin(&mut self) {
        unsafe {
            self.pool = (self.objc.pool_push)();
            self.cb = self.objc.send0(self.queue, self.s.command_buffer);
            self.enc = self.objc.send0(self.cb, self.s.compute_command_encoder);
            assert!(!self.enc.is_null(), "cannot create a Metal command encoder");
        }
    }

    /// Submit the command buffer and wait for it; the results are then
    /// visible in the shared activation buffers.
    pub fn finish(&mut self) {
        unsafe {
            self.objc.send0::<()>(self.enc, self.s.end_encoding);
            self.objc.send0::<()>(self.cb, self.s.commit);
            self.objc.send0::<()>(self.cb, self.s.wait_until_completed);
            let status: usize = self.objc.send0(self.cb, self.s.status);
            if status != MTL_COMMAND_BUFFER_STATUS_COMPLETED {
                let err: Id = self.objc.send0(self.cb, self.s.error);
                let msg = self.objc.describe_error(err);
                (self.objc.pool_pop)(self.pool);
                panic!("Metal command buffer failed (status {status}): {msg}");
            }
            (self.objc.pool_pop)(self.pool);
            self.cb = null_mut();
            self.enc = null_mut();
        }
    }

    unsafe fn set_pipeline(&self, pso: Id) {
        self.objc.send1::<Id, ()>(self.enc, self.s.set_pipeline, pso);
    }

    unsafe fn set_buf(&self, buf: Id, byte_off: usize, index: usize) {
        debug_assert!(!buf.is_null(), "buffer {index} is not mapped");
        self.objc.send3::<Id, usize, usize, ()>(self.enc, self.s.set_buffer, buf, byte_off, index);
    }

    unsafe fn set_args<T>(&self, args: &T, index: usize) {
        self.objc.send3::<*const c_void, usize, usize, ()>(self.enc, self.s.set_bytes, args as *const T as *const c_void, std::mem::size_of::<T>(), index);
    }

    unsafe fn dispatch(&mut self, groups_x: usize, groups_y: usize, threads: usize) {
        let g = MTLSize { width: groups_x, height: groups_y, depth: 1 };
        let t = MTLSize { width: threads, height: 1, depth: 1 };
        self.objc.send2::<MTLSize, MTLSize, ()>(self.enc, self.s.dispatch_threadgroups, g, t);
        self.dispatches += 1;
    }

    /// `dst[t] = src[t] / rms(src[t]) * w` for `m` rows of `n` floats; `src`
    /// starts at element `src_off`.
    pub fn rmsnorm(&mut self, dst: BufId, src: BufId, src_off: usize, w: &Tensor, n: usize, m: usize, eps: f32) {
        self.rmsnorm_at(dst, 0, src, src_off, w, n, m, eps);
    }

    /// [`Gpu::rmsnorm`] writing from element `dst_off`. `dst == src` with
    /// equal offsets normalises in place (each element is read and written
    /// by the same thread), which is how the per-head q / k norms of Qwen3
    /// run: `n = head_dim`, `m = n_heads`.
    pub fn rmsnorm_at(&mut self, dst: BufId, dst_off: usize, src: BufId, src_off: usize, w: &Tensor, n: usize, m: usize, eps: f32) {
        debug_assert_eq!(w.kind, Kind::F32);
        let args = NormArgs { n: n as u32, m: m as u32, eps };
        unsafe {
            self.set_pipeline(self.pso_rmsnorm);
            self.set_buf(self.bufs[src as usize], src_off * 4, 0);
            self.set_buf(self.wbuf, w.off, 1);
            self.set_buf(self.bufs[dst as usize], dst_off * 4, 2);
            self.set_args(&args, 3);
            self.dispatch(m, 1, 256);
        }
    }

    /// `dst[t][r] (+)= sum_c W[r][c] * src[t][c]` for `m` input vectors of
    /// `w.cols` floats (packed back to back from element `src_off`) into `m`
    /// output vectors of `w.rows` floats from element `dst_off`.
    pub fn matmul(&mut self, dst: BufId, dst_off: usize, w: &Tensor, src: BufId, src_off: usize, m: usize, accumulate: bool) {
        self.matmul_bias(dst, dst_off, w, src, src_off, m, accumulate, None);
    }

    /// [`Gpu::matmul`] with an optional bias vector added to every output row.
    pub fn matmul_bias(&mut self, dst: BufId, dst_off: usize, w: &Tensor, src: BufId, src_off: usize, m: usize, accumulate: bool, bias: Option<&Tensor>) {
        if m == 1 {
            self.mv(dst, dst_off, w, src, src_off, None, MvOpts { accumulate, bias, ..MvOpts::default() });
        } else {
            self.mm(dst, dst_off, w, src, src_off, m, accumulate, bias);
        }
    }

    /// [`Gpu::matmul`] of one vector whose output pairs are then rotated with
    /// the (cos, sin) row in `cs`: the q / k projections with RoPE applied.
    pub fn matmul_rope(&mut self, dst: BufId, dst_off: usize, w: &Tensor, src: BufId, src_off: usize, cs: BufId, d: &Dims) {
        self.mv(dst, dst_off, w, src, src_off, Some(d), MvOpts { rope: Some(cs), ..MvOpts::default() });
    }

    /// Rows per simdgroup for a matrix-vector product. Two rows share the
    /// input vector and keep enough loads in flight per lane; Q6_K rows
    /// (many small loads each) run best one per simdgroup. Measured on an
    /// M1 Pro with the `bench_ops` test; `AOT_GPU_MV_ROWS` overrides it.
    fn mv_rows(&self, kind: Kind, rope: bool) -> usize {
        if let Some(nr) = self.mv_rows_override {
            return nr.max(if rope { 2 } else { 1 });
        }
        if kind == Kind::Q6_K && !rope {
            1
        } else {
            2
        }
    }

    /// Force `rows` (1, 2, 4 or 8) rows per simdgroup for matrix-vector
    /// products, or `None` for the built-in choice. `AOT_GPU_MV_ROWS` sets
    /// the same at start-up. A tuning aid for benchmarks.
    pub fn set_mv_rows(&mut self, rows: Option<usize>) {
        self.mv_rows_override = rows.filter(|v| [1, 2, 4, 8].contains(v));
    }

    /// `dst (+)= W src` for one vector, with the fusions in `opts`. `d` is
    /// needed for `rope`.
    pub fn mv(&mut self, dst: BufId, dst_off: usize, w: &Tensor, src: BufId, src_off: usize, d: Option<&Dims>, opts: MvOpts) {
        let k = kind_idx(w.kind);
        let nr = self.mv_rows(w.kind, opts.rope.is_some());
        let (head_dim, rot_dim) = match (opts.rope, d) {
            (Some(_), Some(d)) => (d.head_dim as u32, d.rot_dim as u32),
            (Some(_), None) => panic!("rope needs the model dimensions"),
            _ => (1, 0),
        };
        let groups = w.rows.div_ceil(4 * nr);
        let args = MmArgs {
            rows: w.rows as u32,
            cols: w.cols as u32,
            m: 1,
            accumulate: opts.accumulate as u32,
            x_stride: w.cols as u32,
            y_stride: w.rows as u32,
            rope: opts.rope.is_some() as u32,
            head_dim,
            rot_dim,
            bias: opts.bias.is_some() as u32,
        };
        if let Some(b) = opts.bias {
            debug_assert!(b.kind == Kind::F32 && b.rows * b.cols >= w.rows, "bias must be f32 with one value per output row");
        }
        unsafe {
            self.set_pipeline(self.pso_mv[k][nr.trailing_zeros() as usize]);
            self.set_buf(self.wbuf, w.off, 0);
            self.set_buf(self.bufs[src as usize], src_off * 4, 1);
            self.set_buf(self.bufs[dst as usize], dst_off * 4, 2);
            self.set_args(&args, 3);
            self.set_buf(self.bufs[opts.rope.unwrap_or(src) as usize], 0, 4);
            self.set_buf(self.wbuf, opts.bias.map(|b| b.off).unwrap_or(0), 5);
            self.dispatch(groups, 1, MM_THREADS);
        }
    }

    /// Batched product (see [`Gpu::matmul`]), `m >= 2`.
    fn mm(&mut self, dst: BufId, dst_off: usize, w: &Tensor, src: BufId, src_off: usize, m: usize, accumulate: bool, bias: Option<&Tensor>) {
        let args = MmArgs { rows: w.rows as u32, cols: w.cols as u32, m: m as u32, accumulate: accumulate as u32, x_stride: w.cols as u32, y_stride: w.rows as u32, rope: 0, head_dim: 1, rot_dim: 0, bias: bias.is_some() as u32 };
        debug_assert!(w.cols.is_multiple_of(MM_TILE), "batched product needs a column count that is a multiple of 32");
        unsafe {
            self.set_pipeline(self.pso_mm[kind_idx(w.kind)]);
            self.set_buf(self.wbuf, w.off, 0);
            self.set_buf(self.bufs[src as usize], src_off * 4, 1);
            self.set_buf(self.bufs[dst as usize], dst_off * 4, 2);
            self.set_args(&args, 3);
            self.set_buf(self.wbuf, bias.map(|b| b.off).unwrap_or(0), 4);
            self.dispatch(w.rows.div_ceil(MM_TILE), m.div_ceil(MM_TILE), MM_THREADS);
        }
    }

    /// `dst[r] = silu(gate[r] . src) * (up[r] . src)`: the feed-forward gate
    /// and up projections plus the SwiGLU product in one dispatch. Both
    /// matrices must have the same format and shape.
    pub fn gate_up_silu(&mut self, dst: BufId, gate: &Tensor, up: &Tensor, src: BufId) {
        debug_assert!(gate.kind == up.kind && gate.rows == up.rows && gate.cols == up.cols);
        let args = MmArgs { rows: gate.rows as u32, cols: gate.cols as u32, m: 1, accumulate: 0, x_stride: gate.cols as u32, y_stride: gate.rows as u32, rope: 0, head_dim: 1, rot_dim: 0, bias: 0 };
        unsafe {
            self.set_pipeline(self.pso_gate[kind_idx(gate.kind)]);
            self.set_buf(self.wbuf, gate.off, 0);
            self.set_buf(self.wbuf, up.off, 1);
            self.set_buf(self.bufs[src as usize], 0, 2);
            self.set_buf(self.bufs[dst as usize], 0, 3);
            self.set_args(&args, 4);
            self.dispatch(gate.rows.div_ceil(4), 1, MM_THREADS);
        }
    }

    /// RoPE in place over `m` rows of query heads in `q` and key heads in
    /// `k` (from element `k_off`, rows `kv_dim` apart) with the (cos, sin)
    /// rows of `cs`.
    pub fn rope(&mut self, q: BufId, k: BufId, k_off: usize, m: usize, cs: BufId, d: &Dims) {
        self.rope_ex(q, k, k_off, m, cs, d, false);
    }

    /// [`Gpu::rope`] with a choice of pair layout: `neox` rotates
    /// `(i, i + rot_dim / 2)` (Qwen2 / Qwen3) instead of adjacent elements.
    pub fn rope_ex(&mut self, q: BufId, k: BufId, k_off: usize, m: usize, cs: BufId, d: &Dims, neox: bool) {
        let args = RopeArgs { m: m as u32, n_head: d.n_head as u32, n_kv_head: d.n_kv_head as u32, head_dim: d.head_dim as u32, rot_dim: d.rot_dim as u32, q_stride: d.q_dim() as u32, k_stride: d.kv_dim() as u32, neox: neox as u32 };
        let total = m * (d.n_head + d.n_kv_head) * (d.rot_dim / 2);
        unsafe {
            self.set_pipeline(self.pso_rope);
            self.set_buf(self.bufs[q as usize], 0, 0);
            self.set_buf(self.bufs[k as usize], k_off * 4, 1);
            self.set_buf(self.bufs[cs as usize], 0, 2);
            self.set_args(&args, 3);
            self.dispatch(total.div_ceil(256), 1, 256);
        }
    }

    /// Causal attention of the `m` query rows in `q` (positions `pos0..`)
    /// over the KV cache of `layer`, written to `out` (rows of `q_dim`).
    pub fn attention(&mut self, out: BufId, q: BufId, layer: usize, pos0: usize, m: usize, d: &Dims, ctx: usize) {
        let base = layer * ctx * d.kv_dim();
        let args = AttnArgs {
            m: m as u32,
            pos0: pos0 as u32,
            n_head: d.n_head as u32,
            gqa: d.gqa() as u32,
            head_dim: d.head_dim as u32,
            kv_dim: d.kv_dim() as u32,
            q_stride: d.q_dim() as u32,
            ctx: ctx as u32,
            scale: 1.0 / (d.head_dim as f32).sqrt(),
        };
        unsafe {
            self.set_pipeline(self.pso_attention);
            self.set_buf(self.bufs[q as usize], 0, 0);
            self.set_buf(self.bufs[BufId::KCache as usize], base * 4, 1);
            self.set_buf(self.bufs[BufId::VCache as usize], base * 4, 2);
            self.set_buf(self.bufs[out as usize], 0, 3);
            self.set_args(&args, 4);
            self.dispatch(m, d.n_head, 128);
        }
    }

    /// `h[i] = silu(h[i]) * up[i]` for `n` elements.
    pub fn silu_mul(&mut self, h: BufId, up: BufId, n: usize) {
        let args = SiluArgs { n: n as u32 };
        unsafe {
            self.set_pipeline(self.pso_silu);
            self.set_buf(self.bufs[h as usize], 0, 0);
            self.set_buf(self.bufs[up as usize], 0, 1);
            self.set_args(&args, 2);
            self.dispatch(n.div_ceil(256), 1, 256);
        }
    }

    /// The Metal device object (for diagnostics).
    pub fn device(&self) -> *mut c_void {
        self.device
    }
}
