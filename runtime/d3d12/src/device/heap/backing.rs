//! The host memory behind a range of XMem physical pages.
//!
//! amdgpu imports only private anonymous memory as a host pointer (#191).
//! XMem can give each physical page one private backing and leave its other
//! addresses as GPU-only aliases. A GPU import of an alias must then use the
//! backing, so that every alias and the CPU see the same memory.
//!
//! The backing must also stay mapped for as long as an import of it exists.
//! RADV passes every live buffer with each submission, so one import whose
//! host pages are gone makes the kernel refuse every later submission with
//! `EFAULT`, and the device is lost. The importer therefore holds the backing
//! until its native heap is released.
//!
//! XMem provides both through two exports:
//!
//! ```text
//! BOOL TuxBoxAcquirePhysicalBacking(UINT64 address, UINT64 bytes, UINT64* backing, UINT64* hold);
//! void TuxBoxReleasePhysicalBacking(UINT64 hold);
//! ```
//!
//! For a 64 KiB-aligned range, acquire writes the start of the contiguous host
//! range that holds the same bytes, which may equal `address`, and a hold. While
//! the hold lives, XMem keeps that host range mapped to private memory, even if
//! the title unmaps, remaps or frees the pages. What it contains after a remap
//! does not matter: the queue retires imports of remapped pages. Acquire fails
//! with `ERROR_NOT_FOUND` when the range holds no XMem physical page; the
//! importer then uses the address itself, as it does without the exports. Any
//! other failure, such as pages whose backings are not contiguous, fails the
//! import.
use std::sync::OnceLock;
use windows::Win32::Foundation::{E_INVALIDARG, ERROR_NOT_FOUND, GetLastError};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::core::{HRESULT, Result, s, w};

type Acquire = unsafe extern "system" fn(u64, u64, *mut u64, *mut u64) -> i32;
type Release = unsafe extern "system" fn(u64);

#[derive(Clone, Copy)]
pub(super) struct Exports {
    acquire: Acquire,
    release: Release,
}

impl Exports {
    #[cfg(test)]
    pub(super) fn new(acquire: Acquire, release: Release) -> Self {
        Self { acquire, release }
    }
}

fn lookup() -> Option<Exports> {
    // SAFETY: both names are XMem exports with the signatures above.
    unsafe {
        let module = GetModuleHandleW(w!("xmem.dll")).ok()?;
        let acquire = GetProcAddress(module, s!("TuxBoxAcquirePhysicalBacking"))?;
        let release = GetProcAddress(module, s!("TuxBoxReleasePhysicalBacking"))?;
        Some(Exports {
            acquire: core::mem::transmute::<unsafe extern "system" fn() -> isize, Acquire>(acquire),
            release: core::mem::transmute::<unsafe extern "system" fn() -> isize, Release>(release),
        })
    }
}

#[cfg(test)]
pub(super) static TEST_EXPORTS: std::sync::Mutex<Option<Exports>> = std::sync::Mutex::new(None);

fn exports() -> Option<Exports> {
    #[cfg(test)]
    if let Some(exports) = *TEST_EXPORTS.lock().unwrap_or_else(|e| e.into_inner()) {
        return Some(exports);
    }
    // The title links XMem, so it is loaded before the first device exists.
    static EXPORTS: OnceLock<Option<Exports>> = OnceLock::new();
    *EXPORTS.get_or_init(lookup)
}

/// Keeps an XMem backing mapped. Drop it only after the native heap.
pub(in super::super) struct Hold {
    release: Release,
    cookie: u64,
}

impl Drop for Hold {
    fn drop(&mut self) {
        // SAFETY: the cookie came from the matching acquire and is released once.
        unsafe { (self.release)(self.cookie) };
    }
}

/// Where to import a guest range from.
pub(super) struct Backing {
    pub(super) address: u64,
    pub(super) hold: Option<Hold>,
}

/// Resolve `base..base + size` to the host memory the GPU must import.
pub(super) fn acquire(base: u64, size: u64) -> Result<Backing> {
    let identity = || Backing { address: base, hold: None };
    let Some(exports) = exports() else { return Ok(identity()) };
    let (mut address, mut cookie) = (0u64, 0u64);
    // SAFETY: both outputs are writable locals.
    if unsafe { (exports.acquire)(base, size, &mut address, &mut cookie) } == 0 {
        let error = unsafe { GetLastError() };
        if error == ERROR_NOT_FOUND {
            return Ok(identity());
        }
        return Err(HRESULT::from_win32(error.0).into());
    }
    let hold = Hold { release: exports.release, cookie };
    if address == 0 || !address.is_multiple_of(65536) {
        return Err(E_INVALIDARG.into());
    }
    if address != base {
        static REDIRECTED: std::sync::Once = std::sync::Once::new();
        crate::diagnostics::log_once(&REDIRECTED, || {
            "game memory is imported through its XMem backing".into()
        });
    }
    Ok(Backing { address, hold: Some(hold) })
}

#[cfg(test)]
#[path = "../../tests/device/heap_backing.rs"]
mod tests;
