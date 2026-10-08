//! Minimal fork-join thread pool.
//!
//! Decoding a token issues a few hundred small parallel regions (one per
//! matrix-vector product). Spawning OS threads per region costs tens of
//! microseconds each, which would dominate the compute, so the pool keeps a
//! fixed set of workers that spin briefly on an epoch counter and fall back
//! to yielding / sleeping when idle.
//!
//! [`Pool::run`] blocks until every worker has finished the job, which is
//! what makes it sound to hand workers a reference with an erased lifetime:
//! the closure cannot be dropped while any worker may still call it.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

type Job<'a> = dyn Fn(usize, usize) + Sync + 'a;

struct Shared {
    n_threads: usize,
    /// Incremented once per job; workers wake up when it changes.
    epoch: AtomicUsize,
    /// Number of workers that still have to finish the current job.
    pending: AtomicUsize,
    stop: AtomicBool,
    /// Fat pointer to the current job, published before `epoch` (Release)
    /// and read after observing the new epoch (Acquire).
    job: std::cell::UnsafeCell<*const Job<'static>>,
}

// SAFETY: `job` is only written by the main thread between jobs (while no
// worker is running) and only read by workers after an Acquire load of
// `epoch` that follows the Release store, so accesses are ordered.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// Fixed-size pool of worker threads for data-parallel loops.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Create a pool that will run jobs on `n_threads` threads in total
    /// (the calling thread participates as thread 0).
    pub fn new(n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            n_threads,
            epoch: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(std::ptr::null::<fn(usize, usize)>() as *const Job<'static>),
        });
        let workers = (1..n_threads)
            .map(|tid| {
                let s = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("aot-worker-{tid}"))
                    .spawn(move || worker(s, tid))
                    .expect("failed to spawn worker thread")
            })
            .collect();
        Pool { shared, workers }
    }

    /// Number of threads including the caller.
    pub fn n_threads(&self) -> usize {
        self.shared.n_threads
    }

    /// Run `f(thread_index, n_threads)` on every thread and wait for all of
    /// them. `f` must partition its work using the thread index.
    pub fn run(&self, f: &Job<'_>) {
        let n = self.shared.n_threads;
        if n == 1 {
            f(0, 1);
            return;
        }
        // SAFETY: we erase the lifetime of `f` but do not return before every
        // worker has reported completion (pending == 0), so the reference
        // never outlives the borrow.
        let erased: *const Job<'static> = unsafe { std::mem::transmute::<*const Job<'_>, *const Job<'static>>(f) };
        unsafe { *self.shared.job.get() = erased };
        self.shared.pending.store(n - 1, Ordering::Relaxed);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        f(0, n);
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: Arc<Shared>, tid: usize) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < 50_000 {
                std::hint::spin_loop();
            } else if spins < 60_000 {
                thread::yield_now();
            } else {
                // Idle (no job for a while): sleep to free the core.
                thread::sleep(Duration::from_micros(200));
            }
        }
        // SAFETY: see `Shared::job`; the pointer is valid until `pending`
        // reaches zero, which happens only after this call returns.
        let job = unsafe { &**shared.job.get() };
        job(tid, shared.n_threads);
        shared.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Split `n` items into `nth` contiguous chunks and return the range owned
/// by thread `tid`.
#[inline]
pub fn split(n: usize, tid: usize, nth: usize) -> std::ops::Range<usize> {
    let chunk = n.div_ceil(nth);
    let start = (tid * chunk).min(n);
    let end = ((tid + 1) * chunk).min(n);
    start..end
}

/// Raw pointer wrapper that lets a parallel region write disjoint parts of
/// one buffer. Callers must guarantee the index ranges touched by different
/// threads do not overlap.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
// SAFETY: the wrapped pointer is only dereferenced on disjoint index ranges
// (enforced by `split`), so concurrent use cannot alias.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// # Safety
    /// `i` must be within the original buffer and not concurrently accessed.
    #[inline]
    pub unsafe fn write(&self, i: usize, v: T) {
        *self.0.add(i) = v;
    }

    /// # Safety
    /// `range` must lie within the original buffer and be disjoint from the
    /// ranges used by other threads.
    #[inline]
    pub unsafe fn slice_mut(&self, range: std::ops::Range<usize>) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(range.start), range.end - range.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn sums_disjoint_ranges() {
        let pool = Pool::new(4);
        let n = 1003usize;
        let mut out = vec![0u64; n];
        let o = SendPtr(out.as_mut_ptr());
        for round in 0..50u64 {
            pool.run(&|tid, nth| {
                for i in split(n, tid, nth) {
                    unsafe { o.write(i, i as u64 * round) };
                }
            });
            for (i, &v) in out.iter().enumerate() {
                assert_eq!(v, i as u64 * round);
            }
        }
    }

    #[test]
    fn every_thread_runs_once() {
        let pool = Pool::new(3);
        let hits = AtomicU64::new(0);
        pool.run(&|tid, nth| {
            assert_eq!(nth, 3);
            hits.fetch_add(1 << (tid * 8), Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 0x010101);
    }

    #[test]
    fn single_thread() {
        let pool = Pool::new(1);
        let mut v = 0;
        let p = SendPtr(&mut v as *mut i32);
        pool.run(&|_, _| unsafe { p.write(0, 7) });
        assert_eq!(v, 7);
    }

    #[test]
    fn split_covers_everything() {
        for n in [0usize, 1, 5, 8, 17, 1000] {
            for nth in 1..9 {
                let mut total = 0;
                let mut last_end = 0;
                for t in 0..nth {
                    let r = split(n, t, nth);
                    assert_eq!(r.start, last_end.max(r.start));
                    last_end = r.end;
                    total += r.len();
                }
                assert_eq!(total, n);
            }
        }
    }
}
