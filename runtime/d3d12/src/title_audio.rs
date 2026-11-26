//! ProjectScarlett 1.0.2.1: release a lock the title's audio pump leaves held.
//!
//! The pump at RVA `0x5267720` takes the audio output's lock and then, when the
//! output has no streams yet, returns without releasing it. The presenter
//! thread needs that lock for its next frame, so it stops for good: nothing is
//! presented, the render command buffers are never consumed, and the main
//! thread asserts at RVA `0x4dcbc9d` once they fill. The title reaches this
//! when it presents before audio start-up has finished, for example while the
//! host is slow to open an audio stream.
//!
//! The early return's branch is pointed at the `int3` padding in front of the
//! function. The exception handler there calls the title's own unlock with the
//! same lock and resumes at the function's epilogue.
use windows::Win32::System::Diagnostics::Debug::CONTEXT;
use windows::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS, VirtualProtect};

/// `int3` padding in front of the pump.
const PADDING: usize = 0x526771f;
/// `cmp dword [rdi+0x52c], 0; je <epilogue>`, after the lock call.
const CHECK: usize = 0x5267740;
const EPILOGUE: usize = 0x5267cfb;
/// The title's lock release, taking the lock in `rcx`.
const UNLOCK: usize = 0x4d9dab0;

const CHECK_BYTES: [u8; 13] = [0x83, 0xbf, 0x2c, 0x05, 0x00, 0x00, 0x00, 0x0f, 0x84, 0xae, 0x05, 0x00, 0x00];
/// `add rsp, 0x468`, the start of the epilogue.
const EPILOGUE_BYTES: [u8; 7] = [0x48, 0x81, 0xc4, 0x68, 0x04, 0x00, 0x00];
const UNLOCK_BYTES: [u8; 8] = [0x48, 0x8b, 0x01, 0xff, 0x49, 0x10, 0x75, 0x2a];
/// Offset of the branch's 32-bit displacement within the check.
const DISPLACEMENT: usize = 9;

/// Whether the image holds the 1.0.2.1 code this patch relies on.
fn matches(image: &[u8]) -> bool {
    let at = |rva: usize, bytes: &[u8]| image.get(rva..rva + bytes.len()) == Some(bytes);
    at(PADDING, &[0xcc]) && at(CHECK, &CHECK_BYTES) && at(EPILOGUE, &EPILOGUE_BYTES) && at(UNLOCK, &UNLOCK_BYTES)
}

/// The displacement that sends the branch to the padding byte.
fn redirected() -> [u8; 4] {
    let next = (CHECK + DISPLACEMENT + 4) as i64;
    ((PADDING as i64 - next) as i32).to_le_bytes()
}

/// Redirect the early return. Called before title code runs.
pub(crate) fn patch(start: usize, end: usize) {
    // SAFETY: the mapped title image spans start..end.
    let image = unsafe { core::slice::from_raw_parts(start as *const u8, end - start) };
    if !matches(image) {
        return;
    }
    let at = (start + CHECK + DISPLACEMENT) as *mut [u8; 4];
    // SAFETY: four bytes of the matched instruction; no title thread runs yet.
    unsafe {
        let mut old = PAGE_PROTECTION_FLAGS::default();
        if VirtualProtect(at.cast(), 4, PAGE_EXECUTE_READWRITE, &mut old).is_err() {
            crate::diagnostics::log("could not patch the title's audio lock release");
            return;
        }
        at.write_unaligned(redirected());
        let _ = VirtualProtect(at.cast(), 4, old, &mut old);
    }
    crate::diagnostics::log(&format!("patched the title's audio pump (RVA {CHECK:#x}) to release its lock"));
}

/// Handle the redirected early return at `int3`.
pub(crate) fn handle(context: &mut CONTEXT, start: usize, int3: usize) -> bool {
    if int3 != start + PADDING {
        return false;
    }
    // Call the unlock as the function body would: the lock is in rdi, and the
    // return address is the epilogue the branch originally went to.
    context.Rcx = context.Rdi;
    context.Rsp -= 8;
    // SAFETY: the faulting thread's stack, below the frame of the pump.
    unsafe { *(context.Rsp as *mut u64) = (start + EPILOGUE) as u64 };
    context.Rip = (start + UNLOCK) as u64;
    static LOGGED: std::sync::Once = std::sync::Once::new();
    crate::diagnostics::log_once(&LOGGED, || {
        "the title's audio output has no streams yet; released the lock its pump leaves held".into()
    });
    true
}

#[cfg(test)]
#[path = "tests/title_audio.rs"]
mod tests;
