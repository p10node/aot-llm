//! C ABI of a model library (`aot-llm compile --lib`).
//!
//! The generated `lib.rs` invokes [`export_c_api!`] with its [`Runtime`];
//! the macro expands to the `#[no_mangle] extern "C"` functions declared in
//! `aot_llm.h`, which are thin wrappers over the functions in this module.
//! Everything here is model independent: the model reaches it through the
//! `Runtime` handed to [`open`].
//!
//! Handles are heap objects behind raw pointers. The model is a process-wide
//! singleton (there is one model in the library); contexts and samplers are
//! created freely. Evaluations are serialised by a mutex because the thread
//! pool runs one job at a time, so contexts may be used from any thread.

use super::app::{Gpu, Runtime, DEFAULT_BATCH};
use super::chat;
use super::constrain::{Choices, StopMatcher};
use super::matvec;
use super::pool::Pool;
use super::prefix::Prefix;
pub use super::sampler::Sampler;
use super::sampler::SamplerConfig;
use super::state::{State, Weights};
use super::tokenizer::{chat_format, Tokenizer};
use super::weights;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::sync::Mutex;

pub const DEVICE_AUTO: i32 = 0;
pub const DEVICE_CPU: i32 = 1;
pub const DEVICE_GPU: i32 = 2;

// ---------------------------------------------------------------------------
// C structs (layouts match aot_llm.h)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelParams {
    pub n_threads: i32,
    pub device: i32,
    pub prefault: i32,
    pub weights_path: *const c_char,
    pub kernels: *const c_char,
}

impl ModelParams {
    pub fn default_params() -> ModelParams {
        ModelParams { n_threads: 0, device: DEVICE_AUTO, prefault: 1, weights_path: std::ptr::null(), kernels: std::ptr::null() }
    }
}

#[repr(C)]
pub struct ModelInfo {
    pub name: *const c_char,
    pub arch: *const c_char,
    pub file_type: *const c_char,
    pub backend: *const c_char,
    pub compiler: *const c_char,
    pub n_params: u64,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_layer: u32,
    pub dim: u32,
    pub n_prefix_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u64,
}

impl SamplerParams {
    pub fn default_params() -> SamplerParams {
        let d = SamplerConfig::default();
        SamplerParams { temperature: d.temperature, top_k: d.top_k as i32, top_p: d.top_p, repeat_penalty: d.repeat_penalty, repeat_last_n: d.repeat_last_n as i32, seed: d.seed }
    }

    fn config(&self) -> SamplerConfig {
        SamplerConfig { temperature: self.temperature, top_k: self.top_k.max(0) as usize, top_p: self.top_p, repeat_penalty: self.repeat_penalty, repeat_last_n: self.repeat_last_n.max(0) as usize, seed: self.seed }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GenerateParams {
    pub chat: i32,
    pub system: *const c_char,
    pub max_tokens: i32,
    pub add_bos: i32,
    pub sampler: SamplerParams,
    /// Stop strings (`n_stop` NUL-terminated strings).
    pub stop: *const *const c_char,
    pub n_stop: usize,
    /// Allowed answers: the output is exactly one of these strings.
    pub choices: *const *const c_char,
    pub n_choices: usize,
}

impl GenerateParams {
    pub fn default_params() -> GenerateParams {
        GenerateParams { chat: 0, system: std::ptr::null(), max_tokens: 128, add_bos: -1, sampler: SamplerParams::default_params(), stop: std::ptr::null(), n_stop: 0, choices: std::ptr::null(), n_choices: 0 }
    }
}

/// Collect `n` C strings into owned Rust strings (invalid UTF-8 skipped).
unsafe fn strings(p: *const *const c_char, n: usize) -> Vec<String> {
    if p.is_null() || n == 0 {
        return Vec::new();
    }
    std::slice::from_raw_parts(p, n).iter().filter_map(|&s| cstr(s).map(str::to_string)).collect()
}

/// `int (*)(const char *piece, size_t len, uint32_t token, void *user)`.
pub type TokenCb = Option<unsafe extern "C" fn(*const c_char, usize, u32, *mut std::ffi::c_void) -> i32>;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub fn set_error(msg: impl Into<String>) {
    let s = CString::new(msg.into().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = s);
}

/// Pointer to this thread's last error message (valid until the next error
/// on the same thread).
pub fn last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller passes a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Bytes of a C string argument: `len` bytes, or NUL-terminated when `len == 0`.
unsafe fn bytes<'a>(p: *const c_char, len: usize) -> Option<&'a [u8]> {
    if p.is_null() {
        return None;
    }
    Some(if len == 0 { CStr::from_ptr(p).to_bytes() } else { std::slice::from_raw_parts(p as *const u8, len) })
}

/// Copy `src` into a caller buffer, returning the full length.
unsafe fn copy_out(src: &[u8], out: *mut c_char, cap: usize) -> i32 {
    if !out.is_null() && cap > 0 {
        let n = src.len().min(cap);
        std::ptr::copy_nonoverlapping(src.as_ptr(), out as *mut u8, n);
    }
    src.len() as i32
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Model {
    rt: Runtime,
    tok: Tokenizer,
    wbytes: &'static [u8],
    weights: Weights,
    pool: Pool,
    threads: usize,
    device: i32,
    /// Serialises evaluations: the pool runs one job at a time.
    lock: Mutex<()>,
    name: CString,
    arch: CString,
    file_type: CString,
    compiler: CString,
    backend: Mutex<CString>,
    n_prefix_tokens: u32,
}

/// The process-wide model handle (a raw pointer, kept behind a mutex).
struct ModelPtr(*mut Model);
// SAFETY: the pointer is only handed out, never dereferenced through this slot.
unsafe impl Send for ModelPtr {}

static MODEL: Mutex<Option<ModelPtr>> = Mutex::new(None);

/// Open the model (once per process); later calls return the same handle.
///
/// # Safety
/// `params` is null or points to a valid `ModelParams` whose strings are
/// NUL-terminated.
pub unsafe fn open(rt: Runtime, params: *const ModelParams) -> *mut Model {
    let mut slot = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = &*slot {
        return m.0;
    }
    let p = if params.is_null() { ModelParams::default_params() } else { *params };
    match Model::new(rt, &p) {
        Ok(m) => {
            let ptr = Box::into_raw(Box::new(m));
            *slot = Some(ModelPtr(ptr));
            ptr
        }
        Err(e) => {
            set_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Drop the model. Contexts created from it must have been freed.
///
/// # Safety
/// `m` came from [`open`] and is not used afterwards.
pub unsafe fn close(m: *mut Model) {
    let mut slot = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().is_some_and(|cur| cur.0 == m) {
        *slot = None;
        drop(Box::from_raw(m));
    }
}

impl Model {
    fn new(rt: Runtime, p: &ModelParams) -> Result<Model, String> {
        let tok = Tokenizer::new(rt.tokenizer_blob)?;
        let path = cstr(p.weights_path).map(Path::new);
        let wbytes = (rt.weights)(path).map_err(|e| format!("cannot open weights: {e}"))?;
        if wbytes.len() < rt.info.weights_len {
            return Err(format!("weights blob is {} bytes, expected {}", wbytes.len(), rt.info.weights_len));
        }
        if let Some(name) = cstr(p.kernels) {
            match matvec::kernels_by_name(name) {
                Some(k) => {
                    matvec::set_kernels(k);
                }
                None => return Err(format!("unknown kernel set {name}")),
            }
        }
        let threads = if p.n_threads > 0 { p.n_threads as usize } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8) };
        if p.prefault != 0 {
            weights::prefault_background(&wbytes[..rt.info.weights_len], threads);
        }
        let device = match p.device {
            DEVICE_AUTO | DEVICE_CPU | DEVICE_GPU => p.device,
            other => return Err(format!("unknown device {other}")),
        };
        if device == DEVICE_GPU && rt.forward_gpu.is_none() {
            return Err("this library was built without GPU support (aot-llm compile --gpu metal)".into());
        }
        let n_prefix_tokens = Prefix::parse(rt.prefix_kv, rt.dims).map(|p| p.len() as u32).unwrap_or(0);
        let backend = CString::new(matvec::active_kernels().name).unwrap_or_default();
        Ok(Model {
            tok,
            wbytes,
            weights: Weights::new(wbytes),
            pool: Pool::new(threads),
            threads,
            device,
            lock: Mutex::new(()),
            name: CString::new(rt.info.name).unwrap_or_default(),
            arch: CString::new(rt.info.arch).unwrap_or_default(),
            file_type: CString::new(rt.info.file_type).unwrap_or_default(),
            compiler: CString::new(rt.info.compiler_version).unwrap_or_default(),
            backend: Mutex::new(backend),
            n_prefix_tokens,
            rt,
        })
    }

    pub fn info(&self, out: &mut ModelInfo) {
        let d = self.rt.dims;
        *out = ModelInfo {
            name: self.name.as_ptr(),
            arch: self.arch.as_ptr(),
            file_type: self.file_type.as_ptr(),
            backend: self.backend.lock().map(|b| b.as_ptr()).unwrap_or(std::ptr::null()),
            compiler: self.compiler.as_ptr(),
            n_params: self.rt.info.n_params,
            n_vocab: d.vocab as u32,
            n_ctx_train: d.n_ctx_train as u32,
            n_layer: d.n_layer as u32,
            dim: d.dim as u32,
            n_prefix_tokens: self.n_prefix_tokens,
        };
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Chat template to use: Qwen3 answers without thinking by default.
    fn chat_fmt(&self) -> u32 {
        self.tok.chat_format()
    }

    pub fn chat_prompt(&self, system: Option<&str>, user: &str) -> Option<String> {
        chat::format(self.chat_fmt(), system, user)
    }

    /// The text that continues a conversation with another user message.
    pub fn chat_continuation(&self, user: &str) -> Option<String> {
        chat::continuation(self.chat_fmt(), user)
    }

    /// Stop token ids (EOS / EOT / ...), for the answer constraints.
    fn stop_ids(&self) -> Vec<u32> {
        (0..self.rt.dims.vocab as u32).filter(|&t| self.tok.is_stop(t)).collect()
    }

    /// `aot_tokenize`.
    pub unsafe fn tokenize(&self, text: *const c_char, len: usize, add_bos: i32, parse_special: i32, out: *mut u32, cap: usize) -> i32 {
        let Some(b) = bytes(text, len) else {
            set_error("text is NULL");
            return -1;
        };
        let Ok(s) = std::str::from_utf8(b) else {
            set_error("text is not valid UTF-8");
            return -1;
        };
        let add_bos = if add_bos < 0 { self.tok.add_bos() } else { add_bos != 0 };
        let ids = self.tok.encode(s, add_bos, parse_special != 0);
        if !out.is_null() && cap > 0 {
            let n = ids.len().min(cap);
            std::ptr::copy_nonoverlapping(ids.as_ptr(), out, n);
        }
        ids.len() as i32
    }

    /// `aot_detokenize`.
    pub unsafe fn detokenize(&self, ids: *const u32, n: usize, out: *mut c_char, cap: usize) -> i32 {
        if ids.is_null() && n > 0 {
            set_error("ids is NULL");
            return -1;
        }
        let ids = if n == 0 { &[][..] } else { std::slice::from_raw_parts(ids, n) };
        let mut buf = Vec::new();
        for &id in ids {
            if id as usize >= self.tok.n_tokens() {
                set_error(format!("token id {id} out of range"));
                return -1;
            }
            self.tok.decode_piece(id, &mut buf, false);
        }
        copy_out(&buf, out, cap)
    }

    pub unsafe fn token_piece(&self, id: u32, out: *mut c_char, cap: usize) -> i32 {
        if id as usize >= self.tok.n_tokens() {
            set_error(format!("token id {id} out of range"));
            return -1;
        }
        let mut buf = Vec::new();
        self.tok.decode_piece(id, &mut buf, false);
        copy_out(&buf, out, cap)
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

pub struct Ctx {
    model: &'static Model,
    state: State,
    gpu: Option<Gpu>,
    /// Tokens evaluated so far (the repeat penalty looks at them).
    history: Vec<u32>,
    batch: usize,
    /// Decoded bytes waiting for a complete UTF-8 sequence (`generate`).
    pending: Vec<u8>,
}

#[cfg(target_os = "macos")]
fn gpu_new(weights: &'static [u8], state: &State, d: &super::state::Dims) -> Result<Gpu, String> {
    Gpu::new(weights, state, d)
}

#[cfg(not(target_os = "macos"))]
fn gpu_new(_weights: &'static [u8], _state: &State, _d: &super::state::Dims) -> Result<Gpu, String> {
    Err("no GPU backend on this platform".into())
}

/// `aot_ctx_new`.
///
/// # Safety
/// `m` is null or a live handle from [`open`]; the model must outlive the
/// context (documented in the header).
pub unsafe fn ctx_new(m: *mut Model, n_ctx: u32, n_batch: u32) -> *mut Ctx {
    if m.is_null() {
        set_error("model is NULL");
        return std::ptr::null_mut();
    }
    let model: &'static Model = &*m;
    let d = model.rt.dims;
    let ctx = if n_ctx == 0 { d.n_ctx_train } else { n_ctx as usize };
    let batch = if n_batch == 0 { DEFAULT_BATCH } else { n_batch as usize };
    let rope_ff = model.rt.rope_freqs.as_ref().map(|t| model.weights.f32s(t));
    let state = State::new(d, ctx, rope_ff, model.threads, batch);
    let mut gpu = None;
    if model.device != DEVICE_CPU && model.rt.forward_gpu.is_some() {
        match gpu_new(model.wbytes, &state, d) {
            Ok(g) => {
                if let Ok(mut b) = model.backend.lock() {
                    *b = CString::new(format!("metal ({})", g.name)).unwrap_or_default();
                }
                gpu = Some(g);
            }
            Err(e) if model.device == DEVICE_AUTO => {
                let _ = e;
            }
            Err(e) => {
                set_error(e);
                return std::ptr::null_mut();
            }
        }
    }
    Box::into_raw(Box::new(Ctx { model, state, gpu, history: Vec::new(), batch, pending: Vec::new() }))
}

/// # Safety
/// `c` is null or came from [`ctx_new`] and is not used afterwards.
pub unsafe fn ctx_free(c: *mut Ctx) {
    if !c.is_null() {
        drop(Box::from_raw(c));
    }
}

impl Ctx {
    pub fn reset(&mut self) {
        self.history.clear();
        self.pending.clear();
    }

    pub fn n_past(&self) -> u32 {
        self.history.len() as u32
    }

    pub fn n_ctx(&self) -> u32 {
        self.state.ctx as u32
    }

    pub fn logits(&self) -> *const f32 {
        self.state.logits.as_ptr()
    }

    fn step(&mut self, token: u32, pos: usize, want_logits: bool) {
        let rt = &self.model.rt;
        match &mut self.gpu {
            Some(g) => (rt.forward_gpu.expect("gpu forward"))(&mut self.state, &self.model.weights, g, token, pos, want_logits),
            None => (rt.forward)(&mut self.state, &self.model.weights, &self.model.pool, token, pos, want_logits),
        }
    }

    fn step_batch(&mut self, tokens: &[u32], pos0: usize, want_logits: bool) {
        let rt = &self.model.rt;
        match &mut self.gpu {
            Some(g) => (rt.forward_batch_gpu.expect("gpu batched forward"))(&mut self.state, &self.model.weights, g, tokens, pos0, want_logits),
            None => (rt.forward_batch)(&mut self.state, &self.model.weights, &self.model.pool, tokens, pos0, want_logits),
        }
    }

    /// Append `tokens` to the sequence and evaluate them.
    pub fn eval(&mut self, tokens: &[u32], want_logits: bool) -> Result<(), String> {
        let d = self.model.rt.dims;
        let n_past = self.history.len();
        if tokens.is_empty() {
            return Ok(());
        }
        if n_past + tokens.len() > self.state.ctx {
            return Err(format!("context full: {} + {} tokens > {}", n_past, tokens.len(), self.state.ctx));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= d.vocab) {
            return Err(format!("token id {t} out of range"));
        }
        let _guard = self.model.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut done = 0usize;
        if n_past == 0 {
            if let Some(pre) = Prefix::parse(self.model.rt.prefix_kv, d) {
                if tokens.len() > pre.len() && tokens.starts_with(&pre.tokens) {
                    pre.load_into(&mut self.state);
                    done = pre.len();
                }
            }
        }
        let batch = self.batch.max(1);
        while done < tokens.len() {
            let end = (done + batch).min(tokens.len());
            let last = end == tokens.len() && want_logits;
            if end - done == 1 {
                self.step(tokens[done], n_past + done, last);
            } else {
                let chunk = tokens[done..end].to_vec();
                self.step_batch(&chunk, n_past + done, last);
            }
            done = end;
        }
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    /// Sample the next token from the current logits.
    pub fn sample(&mut self, sampler: &mut Sampler) -> u32 {
        sampler.sample(&mut self.state.logits, &self.history)
    }

    /// `aot_ctx_save`: the tokens of the sequence and their KV rows. The
    /// last token is stored without its row so that `load` re-evaluates it
    /// and the logits are available right after loading.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let n = self.history.len();
        let blob = super::prefix::serialize(&self.state, self.model.rt.dims, &self.history, n.saturating_sub(1));
        std::fs::write(path, &blob).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// `aot_ctx_load`: restore a saved sequence into an empty context.
    pub fn load(&mut self, path: &Path) -> Result<(), String> {
        if !self.history.is_empty() {
            return Err("the context is not empty (aot_ctx_reset first)".into());
        }
        let blob = weights::mmap_file(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let Some(pre) = Prefix::parse(blob, self.model.rt.dims) else {
            return Err(format!("{} is not a KV cache of this model", path.display()));
        };
        if pre.len() >= self.state.ctx {
            return Err(format!("saved sequence has {} tokens, context is {}", pre.len(), self.state.ctx));
        }
        let _guard = self.model.lock.lock().unwrap_or_else(|e| e.into_inner());
        pre.load_into(&mut self.state);
        let cached = pre.cached();
        self.history.extend_from_slice(&pre.tokens[..cached]);
        let tail = pre.tokens[cached..].to_vec();
        drop(_guard);
        if !tail.is_empty() {
            self.eval(&tail, true)?;
        }
        Ok(())
    }

    /// `aot_generate`. Appends to the context: a second call on the same
    /// context continues the conversation (`chat`) or the text.
    pub unsafe fn generate(&mut self, prompt: *const c_char, prompt_len: usize, params: *const GenerateParams, cb: TokenCb, user: *mut std::ffi::c_void) -> i32 {
        let p = if params.is_null() { GenerateParams::default_params() } else { *params };
        let Some(raw) = bytes(prompt, prompt_len) else {
            set_error("prompt is NULL");
            return -1;
        };
        let Ok(raw) = std::str::from_utf8(raw) else {
            set_error("prompt is not valid UTF-8");
            return -1;
        };
        let tok = &self.model.tok;
        let continuing = !self.history.is_empty();
        let text = if p.chat != 0 {
            let built = if continuing { self.model.chat_continuation(raw) } else { self.model.chat_prompt(cstr(p.system), raw) };
            match built {
                Some(t) => t,
                None => {
                    set_error("this model has no known chat template");
                    return -1;
                }
            }
        } else {
            raw.to_string()
        };
        let add_bos = (if p.add_bos < 0 { tok.add_bos() } else { p.add_bos != 0 }) && !continuing;
        let ids = tok.encode(&text, add_bos, true);
        if ids.is_empty() {
            set_error("empty prompt");
            return -1;
        }
        self.pending.clear();
        if let Err(e) = self.eval(&ids, true) {
            set_error(e);
            return -1;
        }
        let stops = strings(p.stop, p.n_stop);
        let choice_texts = strings(p.choices, p.n_choices);
        let mut stopper = StopMatcher::new(&stops);
        let mut choices = if choice_texts.is_empty() { None } else { Some(Choices::new(tok, &choice_texts)) };
        let stop_ids = if choices.is_some() { self.model.stop_ids() } else { Vec::new() };
        let max_tokens = if p.max_tokens <= 0 { 128 } else { p.max_tokens as usize };
        let mut sampler = Sampler::new(p.sampler.config());
        let mut generated = 0usize;
        // Bytes ready for the callback (complete UTF-8 only).
        let mut outbuf: Vec<u8> = Vec::new();
        let emit = |outbuf: &mut Vec<u8>, token: u32, force: bool| -> bool {
            let Some(f) = cb else {
                outbuf.clear();
                return false;
            };
            let valid = match std::str::from_utf8(outbuf) {
                Ok(_) => outbuf.len(),
                Err(e) => e.valid_up_to(),
            };
            let n = if force || outbuf.len() > 4 { outbuf.len() } else { valid };
            if n == 0 {
                return false;
            }
            let piece: Vec<u8> = outbuf.drain(..n).collect();
            let piece = if std::str::from_utf8(&piece).is_ok() { piece } else { String::from_utf8_lossy(&piece).into_owned().into_bytes() };
            f(piece.as_ptr() as *const c_char, piece.len(), token, user) != 0
        };
        while generated < max_tokens && self.history.len() < self.state.ctx {
            if let Some(ch) = &mut choices {
                ch.mask(&mut self.state.logits, &stop_ids);
            }
            let next = self.sample(&mut sampler);
            if tok.is_stop(next) {
                break;
            }
            let choice_done = match &mut choices {
                Some(ch) => {
                    if !ch.advance(next) {
                        break;
                    }
                    ch.done()
                }
                None => false,
            };
            generated += 1;
            let mut piece = Vec::new();
            tok.decode_piece(next, &mut piece, false);
            if choices.is_some() {
                // Collected; the chosen string is delivered once at the end.
                outbuf.extend(piece);
            } else {
                outbuf.extend(stopper.feed(&piece));
                if emit(&mut outbuf, next, false) {
                    break;
                }
            }
            if stopper.matched.is_some() || choice_done {
                break;
            }
            if generated >= max_tokens || self.history.len() >= self.state.ctx {
                break;
            }
            if let Err(e) = self.eval(&[next], true) {
                set_error(e);
                return -1;
            }
        }
        if stopper.matched.is_none() {
            outbuf.extend(stopper.finish());
        }
        if let Some(t) = choices.as_ref().and_then(|c| c.selected()) {
            outbuf.clear();
            outbuf.extend_from_slice(t.as_bytes());
        }
        let last = self.history.last().copied().unwrap_or(0);
        emit(&mut outbuf, last, true);
        self.pending.clear();
        generated as i32
    }
}

// ---------------------------------------------------------------------------
// Sampler handles
// ---------------------------------------------------------------------------

/// # Safety
/// `params` is null or points to a valid `SamplerParams`.
pub unsafe fn sampler_new(params: *const SamplerParams) -> *mut Sampler {
    let p = if params.is_null() { SamplerParams::default_params() } else { *params };
    Box::into_raw(Box::new(Sampler::new(p.config())))
}

/// # Safety
/// `s` is null or came from [`sampler_new`] and is not used afterwards.
pub unsafe fn sampler_free(s: *mut Sampler) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// Keep the Qwen3 constant referenced so the import is meaningful on every
/// platform (the chat module decides the template behaviour).
const _: u32 = chat_format::QWEN3;

// ---------------------------------------------------------------------------
// The exported functions
// ---------------------------------------------------------------------------

/// Expand to the `extern "C"` functions of `aot_llm.h`. `$capi` is the path
/// of this module in the generated crate (`crate::rt::capi`), `$runtime` an
/// expression producing the generated [`Runtime`], `$prefix` a string put in
/// front of every exported symbol (`""` for the plain `aot_*` names; see
/// `aot-llm compile --symbol-prefix`).
#[macro_export]
macro_rules! export_c_api {
    ($capi:path, $runtime:expr, $prefix:literal) => {
        mod __aot_c_api {
            use std::ffi::{c_char, c_void};
            use $capi as capi;

            #[export_name = concat!($prefix, "aot_model_default_params")]
            pub extern "C" fn aot_model_default_params() -> capi::ModelParams {
                capi::ModelParams::default_params()
            }

            #[export_name = concat!($prefix, "aot_open")]
            pub unsafe extern "C" fn aot_open(params: *const capi::ModelParams) -> *mut capi::Model {
                capi::open($runtime, params)
            }

            #[export_name = concat!($prefix, "aot_close")]
            pub unsafe extern "C" fn aot_close(m: *mut capi::Model) {
                capi::close(m)
            }

            #[export_name = concat!($prefix, "aot_last_error")]
            pub extern "C" fn aot_last_error() -> *const c_char {
                capi::last_error()
            }

            #[export_name = concat!($prefix, "aot_model_info_get")]
            pub unsafe extern "C" fn aot_model_info_get(m: *const capi::Model, out: *mut capi::ModelInfo) {
                if !m.is_null() && !out.is_null() {
                    (*m).info(&mut *out);
                }
            }

            #[export_name = concat!($prefix, "aot_tokenize")]
            pub unsafe extern "C" fn aot_tokenize(m: *const capi::Model, text: *const c_char, len: usize, add_bos: i32, parse_special: i32, out: *mut u32, cap: usize) -> i32 {
                if m.is_null() {
                    capi::set_error("model is NULL");
                    return -1;
                }
                (*m).tokenize(text, len, add_bos, parse_special, out, cap)
            }

            #[export_name = concat!($prefix, "aot_detokenize")]
            pub unsafe extern "C" fn aot_detokenize(m: *const capi::Model, ids: *const u32, n: usize, out: *mut c_char, cap: usize) -> i32 {
                if m.is_null() {
                    capi::set_error("model is NULL");
                    return -1;
                }
                (*m).detokenize(ids, n, out, cap)
            }

            #[export_name = concat!($prefix, "aot_token_piece")]
            pub unsafe extern "C" fn aot_token_piece(m: *const capi::Model, id: u32, out: *mut c_char, cap: usize) -> i32 {
                if m.is_null() {
                    capi::set_error("model is NULL");
                    return -1;
                }
                (*m).token_piece(id, out, cap)
            }

            #[export_name = concat!($prefix, "aot_token_is_stop")]
            pub unsafe extern "C" fn aot_token_is_stop(m: *const capi::Model, id: u32) -> i32 {
                if m.is_null() {
                    return 0;
                }
                (*m).tokenizer().is_stop(id) as i32
            }

            #[export_name = concat!($prefix, "aot_chat_prompt")]
            pub unsafe extern "C" fn aot_chat_prompt(m: *const capi::Model, system: *const c_char, user: *const c_char, out: *mut c_char, cap: usize) -> i32 {
                if m.is_null() || user.is_null() {
                    capi::set_error("model or user is NULL");
                    return -1;
                }
                let sys = if system.is_null() { None } else { std::ffi::CStr::from_ptr(system).to_str().ok() };
                let Ok(user) = std::ffi::CStr::from_ptr(user).to_str() else {
                    capi::set_error("user is not valid UTF-8");
                    return -1;
                };
                match (*m).chat_prompt(sys, user) {
                    Some(t) => {
                        let b = t.as_bytes();
                        if !out.is_null() && cap > 0 {
                            let n = b.len().min(cap);
                            std::ptr::copy_nonoverlapping(b.as_ptr(), out as *mut u8, n);
                        }
                        b.len() as i32
                    }
                    None => {
                        capi::set_error("this model has no known chat template");
                        -1
                    }
                }
            }

            #[export_name = concat!($prefix, "aot_ctx_new")]
            pub unsafe extern "C" fn aot_ctx_new(m: *mut capi::Model, n_ctx: u32, n_batch: u32) -> *mut capi::Ctx {
                capi::ctx_new(m, n_ctx, n_batch)
            }

            #[export_name = concat!($prefix, "aot_ctx_free")]
            pub unsafe extern "C" fn aot_ctx_free(c: *mut capi::Ctx) {
                capi::ctx_free(c)
            }

            #[export_name = concat!($prefix, "aot_ctx_reset")]
            pub unsafe extern "C" fn aot_ctx_reset(c: *mut capi::Ctx) {
                if !c.is_null() {
                    (*c).reset();
                }
            }

            #[export_name = concat!($prefix, "aot_ctx_n_past")]
            pub unsafe extern "C" fn aot_ctx_n_past(c: *const capi::Ctx) -> u32 {
                if c.is_null() {
                    0
                } else {
                    (*c).n_past()
                }
            }

            #[export_name = concat!($prefix, "aot_ctx_n_ctx")]
            pub unsafe extern "C" fn aot_ctx_n_ctx(c: *const capi::Ctx) -> u32 {
                if c.is_null() {
                    0
                } else {
                    (*c).n_ctx()
                }
            }

            #[export_name = concat!($prefix, "aot_eval")]
            pub unsafe extern "C" fn aot_eval(c: *mut capi::Ctx, tokens: *const u32, n: usize, want_logits: i32) -> i32 {
                if c.is_null() || (tokens.is_null() && n > 0) {
                    capi::set_error("context or tokens is NULL");
                    return -1;
                }
                let toks = if n == 0 { &[][..] } else { std::slice::from_raw_parts(tokens, n) };
                match (*c).eval(toks, want_logits != 0) {
                    Ok(()) => 0,
                    Err(e) => {
                        capi::set_error(e);
                        -1
                    }
                }
            }

            #[export_name = concat!($prefix, "aot_logits")]
            pub unsafe extern "C" fn aot_logits(c: *const capi::Ctx) -> *const f32 {
                if c.is_null() {
                    std::ptr::null()
                } else {
                    (*c).logits()
                }
            }

            #[export_name = concat!($prefix, "aot_sampler_default_params")]
            pub extern "C" fn aot_sampler_default_params() -> capi::SamplerParams {
                capi::SamplerParams::default_params()
            }

            #[export_name = concat!($prefix, "aot_sampler_new")]
            pub unsafe extern "C" fn aot_sampler_new(params: *const capi::SamplerParams) -> *mut capi::Sampler {
                capi::sampler_new(params)
            }

            #[export_name = concat!($prefix, "aot_sampler_free")]
            pub unsafe extern "C" fn aot_sampler_free(s: *mut capi::Sampler) {
                capi::sampler_free(s)
            }

            #[export_name = concat!($prefix, "aot_sample")]
            pub unsafe extern "C" fn aot_sample(s: *mut capi::Sampler, c: *mut capi::Ctx) -> u32 {
                if s.is_null() || c.is_null() {
                    capi::set_error("sampler or context is NULL");
                    return 0;
                }
                (*c).sample(&mut *s)
            }

            #[export_name = concat!($prefix, "aot_chat_continuation")]
            pub unsafe extern "C" fn aot_chat_continuation(m: *const capi::Model, user: *const c_char, out: *mut c_char, cap: usize) -> i32 {
                if m.is_null() || user.is_null() {
                    capi::set_error("model or user is NULL");
                    return -1;
                }
                let Ok(user) = std::ffi::CStr::from_ptr(user).to_str() else {
                    capi::set_error("user is not valid UTF-8");
                    return -1;
                };
                match (*m).chat_continuation(user) {
                    Some(t) => {
                        let b = t.as_bytes();
                        if !out.is_null() && cap > 0 {
                            let n = b.len().min(cap);
                            std::ptr::copy_nonoverlapping(b.as_ptr(), out as *mut u8, n);
                        }
                        b.len() as i32
                    }
                    None => {
                        capi::set_error("this model has no known chat template");
                        -1
                    }
                }
            }

            #[export_name = concat!($prefix, "aot_ctx_save")]
            pub unsafe extern "C" fn aot_ctx_save(c: *const capi::Ctx, path: *const c_char) -> i32 {
                if c.is_null() || path.is_null() {
                    capi::set_error("context or path is NULL");
                    return -1;
                }
                let Ok(path) = std::ffi::CStr::from_ptr(path).to_str() else {
                    capi::set_error("path is not valid UTF-8");
                    return -1;
                };
                match (*c).save(std::path::Path::new(path)) {
                    Ok(()) => 0,
                    Err(e) => {
                        capi::set_error(e);
                        -1
                    }
                }
            }

            #[export_name = concat!($prefix, "aot_ctx_load")]
            pub unsafe extern "C" fn aot_ctx_load(c: *mut capi::Ctx, path: *const c_char) -> i32 {
                if c.is_null() || path.is_null() {
                    capi::set_error("context or path is NULL");
                    return -1;
                }
                let Ok(path) = std::ffi::CStr::from_ptr(path).to_str() else {
                    capi::set_error("path is not valid UTF-8");
                    return -1;
                };
                match (*c).load(std::path::Path::new(path)) {
                    Ok(()) => 0,
                    Err(e) => {
                        capi::set_error(e);
                        -1
                    }
                }
            }

            #[export_name = concat!($prefix, "aot_generate_default_params")]
            pub extern "C" fn aot_generate_default_params() -> capi::GenerateParams {
                capi::GenerateParams::default_params()
            }

            #[export_name = concat!($prefix, "aot_generate")]
            pub unsafe extern "C" fn aot_generate(c: *mut capi::Ctx, prompt: *const c_char, prompt_len: usize, params: *const capi::GenerateParams, cb: capi::TokenCb, user: *mut c_void) -> i32 {
                if c.is_null() {
                    capi::set_error("context is NULL");
                    return -1;
                }
                (*c).generate(prompt, prompt_len, params, cb, user)
            }
        }
    };
}
