//! Access to the weights blob: embedded in the executable or memory-mapped
//! from a sidecar file. Both paths are zero-copy: no byte of weight data is
//! read until a kernel touches it, and the OS pages it in on demand.

use std::path::Path;

#[cfg(unix)]
mod sys {
    use std::os::raw::{c_int, c_void};
    pub const PROT_READ: c_int = 1;
    pub const MAP_PRIVATE: c_int = 2;
    pub const MADV_WILLNEED: c_int = 3;
    extern "C" {
        pub fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
        pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }
}

/// Map `path` read-only for the rest of the process lifetime.
#[cfg(unix)]
pub fn mmap_file(path: &Path) -> std::io::Result<&'static [u8]> {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path)?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        return Ok(&[]);
    }
    // SAFETY: standard read-only private mapping of a regular file; the fd
    // may be closed afterwards, the mapping stays valid. The returned slice
    // is never unmapped, hence 'static.
    let p = unsafe { sys::mmap(std::ptr::null_mut(), len, sys::PROT_READ, sys::MAP_PRIVATE, f.as_raw_fd(), 0) };
    if p as usize == usize::MAX {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { std::slice::from_raw_parts(p as *const u8, len) })
}

/// Fallback for non-unix targets: read the file into a leaked buffer.
#[cfg(not(unix))]
pub fn mmap_file(path: &Path) -> std::io::Result<&'static [u8]> {
    let v = std::fs::read(path)?;
    Ok(Box::leak(v.into_boxed_slice()))
}

/// Ask the kernel to start paging in `data` (used by `--prefetch`).
pub fn prefetch(data: &[u8]) {
    #[cfg(unix)]
    {
        if data.is_empty() {
            return;
        }
        let page = 16384usize;
        let start = data.as_ptr() as usize & !(page - 1);
        let end = data.as_ptr() as usize + data.len();
        // SAFETY: advisory call on a range inside an existing mapping.
        unsafe { sys::madvise(start as *mut _, end - start, sys::MADV_WILLNEED) };
    }
    #[cfg(not(unix))]
    {
        let _ = data;
    }
}

/// Page in `data` from background threads while the caller keeps working.
///
/// A cold start otherwise pays one page fault per 16 KiB page, serially, on
/// the thread that first touches each weight (and on macOS each page of a
/// signed executable is also hash-verified on first fault). Touching the
/// pages from `n_threads` detached threads overlaps that work with
/// tokenization and lets the first forward pass find most pages resident.
/// The threads exit when they are done; nothing waits for them.
pub fn prefault_background(data: &'static [u8], n_threads: usize) {
    if data.is_empty() {
        return;
    }
    let n = n_threads.max(1);
    let chunk = data.len().div_ceil(n);
    for t in 0..n {
        let start = (t * chunk).min(data.len());
        let end = ((t + 1) * chunk).min(data.len());
        if start >= end {
            break;
        }
        let part = &data[start..end];
        let _ = std::thread::Builder::new().name(format!("aot-prefault-{t}")).spawn(move || {
            // MADV_WILLNEED is synchronous on macOS (it reads the range before
            // returning), so issue it per chunk from the worker, not the caller.
            prefetch(part);
            let mut acc = 0u8;
            let mut i = 0;
            while i < part.len() {
                // Volatile read so the loop is not optimised away.
                // SAFETY: `i` is within `part`.
                acc ^= unsafe { std::ptr::read_volatile(part.as_ptr().add(i)) };
                i += 4096;
            }
            std::hint::black_box(acc);
        });
    }
}

/// Peak resident set size in bytes, if the platform exposes it.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        #[repr(C)]
        struct Timeval {
            tv_sec: i64,
            tv_usec: i64,
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: Timeval,
            ru_stime: Timeval,
            ru_maxrss: i64,
            _rest: [i64; 13],
        }
        extern "C" {
            fn getrusage(who: std::os::raw::c_int, usage: *mut Rusage) -> std::os::raw::c_int;
        }
        let mut r = Rusage { ru_utime: Timeval { tv_sec: 0, tv_usec: 0 }, ru_stime: Timeval { tv_sec: 0, tv_usec: 0 }, ru_maxrss: 0, _rest: [0; 13] };
        // SAFETY: getrusage fills the struct; RUSAGE_SELF = 0.
        if unsafe { getrusage(0, &mut r) } != 0 {
            return None;
        }
        // macOS reports bytes, Linux reports kilobytes.
        let v = r.ru_maxrss as u64;
        Some(if cfg!(target_os = "macos") { v } else { v * 1024 })
    }
    #[cfg(not(unix))]
    {
        None
    }
}
