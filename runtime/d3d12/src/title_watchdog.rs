//! Pause ProjectScarlett's hang watchdog while TuxBox blocks a title thread.
//!
//! ProjectScarlett 1.0.2.1 runs a watchdog thread (RVA 0x2215A60) that wakes
//! every second. It crashes the title on purpose, with a store to address 0,
//! when the title's heartbeat has not moved for 30 checks or a registered
//! thread has not checked in for 300 seconds. On the console, pipeline
//! creation never takes that long. Under TuxBox a title thread can wait for a
//! shader translation or a driver compile for minutes, and the watchdog then
//! ends the run (the crash at RVA 0x2215C35 after "presented N frames in 40s").
//!
//! The watchdog skips both checks, and restarts its count, while bit 0 of its
//! pause flag (RVA 0x71B6330) is set. A [`Pause`] sets that bit while a title
//! thread waits on TuxBox and clears it again once no wait remains, after a
//! grace period that lets the released threads check in. A flag the title set
//! itself is never cleared. Nothing is written unless the watchdog's code
//! matches this build byte for byte.
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

const FLAG_RVA: usize = 0x71B_6330;
/// `mov cl, [rip+disp]` / `test cl, 1` / `je` in both watchdog loops, and the
/// deliberate crash after each.
const CHECKS: [(usize, &[u8]); 4] = [
    (0x221_5B50, &[0x8A, 0x0D, 0xDA, 0x07, 0xFA, 0x04, 0xF6, 0xC1, 0x01, 0x74, 0x11]),
    (0x221_5CF4, &[0x8A, 0x0D, 0x36, 0x06, 0xFA, 0x04, 0xF6, 0xC1, 0x01, 0x74, 0x11]),
    (0x221_5C35, &[0xC7, 0x04, 0x25, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0x221_5DA9, &[0xC7, 0x04, 0x25, 0, 0, 0, 0, 0, 0, 0, 0]),
];
/// Released threads check in well within this time.
const GRACE: Duration = Duration::from_secs(5);

/// Whether `code` holds the watchdog of this build at `base`, read through
/// `read(rva, len)`.
fn matches(read: impl Fn(usize, usize) -> Option<Vec<u8>>) -> bool {
    CHECKS.iter().all(|(rva, bytes)| read(*rva, bytes.len()).is_some_and(|found| found == *bytes))
}

fn flag() -> Option<&'static AtomicU8> {
    static FLAG: LazyLock<Option<usize>> = LazyLock::new(|| {
        if std::env::var_os("TUXBOX_KEEP_TITLE_WATCHDOG").is_some_and(|v| v == "1") {
            return None;
        }
        let (start, end) = crate::survey::title_image()?;
        let read = |rva: usize, len: usize| {
            let at = start.checked_add(rva)?;
            if at.checked_add(len)? > end {
                return None;
            }
            let (bytes, avail) = crate::clwb::read_code_bytes(at, at + len);
            (avail == len).then(|| bytes[..len].to_vec())
        };
        let flag = start.checked_add(FLAG_RVA).filter(|&at| at < end)?;
        if !matches(read) {
            return None;
        }
        crate::diagnostics::log(&format!(
            "pausing the title's hang watchdog (RVA {FLAG_RVA:#x}) while shaders compile"
        ));
        Some(flag)
    });
    // SAFETY: the address lies in the title's writable data section, verified
    // through the watchdog code that reads it, and stays mapped for the process.
    FLAG.map(|at| unsafe { AtomicU8::from_ptr(at as *mut u8) })
}

struct State {
    waiting: usize,
    /// Whether TuxBox set the bit, so that only TuxBox clears it.
    set: bool,
    released: Option<Instant>,
}

static STATE: Mutex<State> = Mutex::new(State { waiting: 0, set: false, released: None });

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|error| error.into_inner())
}

/// Held while a title thread waits on TuxBox.
pub(crate) struct Pause(());

pub(crate) fn pause() -> Option<Pause> {
    let flag = flag()?;
    let mut state = state();
    state.waiting += 1;
    state.released = None;
    if state.waiting == 1 && !state.set && flag.fetch_or(1, Ordering::SeqCst) & 1 == 0 {
        state.set = true;
        static RESUMER: std::sync::Once = std::sync::Once::new();
        RESUMER.call_once(|| {
            let _ = std::thread::Builder::new().name("tuxbox-watchdog".into()).spawn(resume);
        });
    }
    Some(Pause(()))
}

impl Drop for Pause {
    fn drop(&mut self) {
        let mut state = state();
        state.waiting -= 1;
        if state.waiting == 0 {
            state.released = Some(Instant::now());
        }
    }
}

/// Clear the bit once no title thread has waited for the grace period.
fn resume() {
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let mut state = state();
        if state.set && state.waiting == 0 && state.released.is_some_and(|at| at.elapsed() >= GRACE) {
            if let Some(flag) = flag() {
                flag.fetch_and(!1, Ordering::SeqCst);
            }
            state.set = false;
            state.released = None;
        }
    }
}

#[cfg(test)]
#[path = "tests/title_watchdog.rs"]
mod tests;
