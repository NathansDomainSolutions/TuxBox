//! Fallback for `clzero` on host CPUs without the CLZERO instruction.
//!
//! Xbox Series X|S (Zen 2) implements AMD's `clzero` (`0F 01 FC`, CPUID
//! `0x8000_0008` EBX bit 0). It zeroes the 64-byte cache line that contains the
//! address in RAX and leaves RAX unchanged. Intel CPUs do not implement it and
//! raise `EXCEPTION_ILLEGAL_INSTRUCTION`.
//!
//! No other three-byte instruction zeroes a line, so the site cannot be
//! rewritten in place the way `clwb` is. Instead the handler performs the store
//! and resumes after the instruction.
//!
//! ProjectScarlett 1.0.2.1 clears object fields with back-to-back loops of the
//! form `clzero; add rax, 0x40; cmp rax, reg; jb clzero`. When that exact loop
//! follows the faulting instruction, the handler clears every line the loop
//! would clear, leaves RAX at its exit value and resumes at the `cmp`, so the
//! CPU evaluates the loop condition itself. Each loop then costs one trap
//! instead of one per line.
use core::ffi::c_void;
use std::sync::{LazyLock, Mutex};
use windows::Win32::System::Diagnostics::Debug::CONTEXT;
use windows::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};

const CLZERO: [u8; 3] = [0x0F, 0x01, 0xFC];
const LINE: u64 = 64;
/// `add rax, 0x40`
const ADD_RAX_LINE: [u8; 4] = [0x48, 0x83, 0xC0, 0x40];
/// Bytes from the `clzero` to the `cmp` of a recognized loop.
const LOOP_CMP_OFFSET: u64 = (CLZERO.len() + ADD_RAX_LINE.len()) as u64;
const MEM_COMMIT_STATE: u32 = 0x1000;
const PAGE_GUARD_FLAG: u32 = 0x100;
/// Writable page protection mask (READWRITE, WRITECOPY, EXECUTE_READWRITE,
/// EXECUTE_WRITECOPY).
const PAGE_WRITABLE_MASK: u32 = 0xCC;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    /// A `clzero` that is not part of a recognized loop.
    Single,
    /// `clzero; add rax, 0x40; cmp rax, <end_reg>; jb clzero`.
    Loop { end_reg: u8 },
}

/// Lines a fallback clears, and where execution resumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) first: u64,
    pub(crate) bytes: u64,
    pub(crate) rax: u64,
    pub(crate) resume: u64,
}

fn host_has_clzero() -> bool {
    static CLZERO_SUPPORTED: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            // Rust 1.95 still marks `__cpuid` unsafe; newer toolchains make it safe.
            #[allow(unused_unsafe)]
            unsafe {
                let max_extended = core::arch::x86_64::__cpuid(0x8000_0000).eax;
                max_extended >= 0x8000_0008
                    && (core::arch::x86_64::__cpuid(0x8000_0008).ebx & 1) != 0
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    });
    *CLZERO_SUPPORTED
}

/// Decode a prefix-free `clzero` at the start of `bytes`, and the loop that
/// follows it when it matches exactly.
pub(crate) fn decode(bytes: &[u8]) -> Option<Site> {
    if !bytes.starts_with(&CLZERO) {
        return None;
    }
    let rest = &bytes[CLZERO.len()..];
    // add rax, 0x40 ; cmp rax, r64 (REX.W [REX.R] 39 /r, mod 11, rm rax) ;
    // jb rel8 -12, back to the clzero.
    if let [a, b, c, d, rex, 0x39, modrm, 0x72, 0xF4, ..] = *rest {
        if [a, b, c, d] == ADD_RAX_LINE
            && (rex == 0x48 || rex == 0x4C)
            && modrm >> 6 == 3
            && modrm & 7 == 0
        {
            let end_reg = ((modrm >> 3) & 7) | if rex == 0x4C { 8 } else { 0 };
            if end_reg != 0 {
                return Some(Site::Loop { end_reg });
            }
        }
    }
    Some(Site::Single)
}

/// The lines one `clzero` at `rip` clears. RAX is unchanged.
pub(crate) fn plan_single(rip: u64, rax: u64) -> Plan {
    Plan { first: rax & !(LINE - 1), bytes: LINE, rax, resume: rip + CLZERO.len() as u64 }
}

/// The lines the recognized loop clears, starting from `rax`, until RAX is no
/// longer below `end`. Returns `None` if the arithmetic would wrap.
pub(crate) fn plan_loop(rip: u64, rax: u64, end: u64) -> Option<Plan> {
    let first = rax & !(LINE - 1);
    let after_first = rax.checked_add(LINE)?;
    let extra = if after_first < end { (end - after_first).div_ceil(LINE) } else { 0 };
    let bytes = extra.checked_add(1)?.checked_mul(LINE)?;
    first.checked_add(bytes)?;
    Some(Plan { first, bytes, rax: rax.checked_add(bytes)?, resume: rip + LOOP_CMP_OFFSET })
}

fn register(context: &CONTEXT, index: u8) -> u64 {
    match index {
        0 => context.Rax,
        1 => context.Rcx,
        2 => context.Rdx,
        3 => context.Rbx,
        4 => context.Rsp,
        5 => context.Rbp,
        6 => context.Rsi,
        7 => context.Rdi,
        8 => context.R8,
        9 => context.R9,
        10 => context.R10,
        11 => context.R11,
        12 => context.R12,
        13 => context.R13,
        14 => context.R14,
        _ => context.R15,
    }
}

/// Whether every byte of `[start, start + len)` is committed, writable and not
/// a guard page.
fn writable(start: u64, len: u64) -> bool {
    let (Ok(start), Ok(len)) = (usize::try_from(start), usize::try_from(len)) else {
        return false;
    };
    let Some(end) = start.checked_add(len) else { return false };
    let mut addr = start;
    while addr < end {
        let mut region = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: `VirtualQuery` inspects the caller's own address space.
        let queried = unsafe {
            VirtualQuery(Some(addr as *const c_void), &mut region, size_of_val(&region))
        };
        if queried == 0
            || region.State.0 != MEM_COMMIT_STATE
            || (region.Protect.0 & PAGE_GUARD_FLAG) != 0
            || (region.Protect.0 & PAGE_WRITABLE_MASK) == 0
        {
            return false;
        }
        let region_end = (region.BaseAddress as usize).saturating_add(region.RegionSize);
        if region_end <= addr {
            return false;
        }
        addr = region_end;
    }
    true
}

/// Apply the fallback without checking the host CPU. Returns whether execution
/// should resume, and the plan that was applied.
fn handle_unchecked(context: &mut CONTEXT, start: usize, end: usize) -> Option<(Site, Plan)> {
    let rip = context.Rip as usize;
    if !(start..end).contains(&rip) {
        return None;
    }
    let (buf, avail) = crate::clwb::read_code_bytes(rip, end);
    let site = decode(&buf[..avail])?;
    let rip = rip as u64;
    let single = plan_single(rip, context.Rax);
    let (site, plan) = match site {
        Site::Loop { end_reg } => match plan_loop(rip, context.Rax, register(context, end_reg)) {
            Some(plan) if writable(plan.first, plan.bytes) => (site, plan),
            // Clear one line at a time; the line that real hardware would fault
            // on is then left to fault here as well.
            _ => (Site::Single, single),
        },
        Site::Single => (site, single),
    };
    if !writable(plan.first, plan.bytes) {
        return None;
    }
    // SAFETY: `writable` verified the range is committed and writable.
    unsafe { core::ptr::write_bytes(plan.first as usize as *mut u8, 0, plan.bytes as usize) };
    context.Rax = plan.rax;
    context.Rip = plan.resume;
    Some((site, plan))
}

/// Handle `EXCEPTION_ILLEGAL_INSTRUCTION` inside `[start, end)` when caused by
/// `clzero` on a host CPU without CLZERO. Returns `true` if execution should
/// resume.
pub(crate) fn handle_illegal_instruction(context: &mut CONTEXT, start: usize, end: usize) -> bool {
    if host_has_clzero() {
        return false;
    }
    let rip = context.Rip as usize;
    let Some((site, _)) = handle_unchecked(context, start, end) else {
        return false;
    };
    static LOGGED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if !logged.contains(&rip) {
        logged.push(rip);
        drop(logged);
        let form = if site == Site::Single { "clzero" } else { "clzero loop" };
        crate::diagnostics::log(&format!(
            "emulated unsupported {form} at RVA {:#x} (host lacks CLZERO)",
            rip - start
        ));
    }
    true
}

// ProjectScarlett 1.0.2.1 calls one clearing routine (RVA 0x4E8AA10) a few
// hundred times per second. Its fast path is five `clzero` loops, so even with
// the loop fallback above every call raised five exceptions: about 4 million
// in half an hour on an Intel host, each one logged by Wine.
//
// The five loops step 64 bytes from `rcx + 0x40` and each one starts where
// the previous one ended. Every length is a whole number of lines, so together
// they clear exactly `[align_down(rcx + 0x40, 64), + 0x7400)`. The
// replacement below clears that range with AVX stores and jumps to the
// routine's own tail at +0xEC. The tail zeroes `xmm0` and runs `vzeroupper`
// before it returns, so the `ymm0` clobber is invisible to the caller, as are
// the different final values of the volatile `rdx` and `r8`-`r10`. Nothing
// else branches into the replaced bytes; the overflow slow path at +0x115
// keeps its own `clzero` loops and the trap fallback.
const SCARLETT_CLEAR_RVA: usize = 0x4E8_AA10;
const SCARLETT_CLEAR_FAST_PATH: usize = 0x60;
const SCARLETT_CLEAR: [u8; 433] = [
    0x48, 0x8D, 0x41, 0x40, 0x4C, 0x8D, 0x91, 0xC0, 0x39, 0x00, 0x00, 0x4C,
    0x8D, 0x89, 0x40, 0x55, 0x00, 0x00, 0x4C, 0x8D, 0x81, 0x00, 0x62, 0x00,
    0x00, 0x48, 0x8D, 0x91, 0xC0, 0x70, 0x00, 0x00, 0x48, 0x3D, 0x7F, 0xC6,
    0xFF, 0xFF, 0x0F, 0x87, 0xE9, 0x00, 0x00, 0x00, 0x49, 0x81, 0xFA, 0x7F,
    0xE4, 0xFF, 0xFF, 0x0F, 0x87, 0xDC, 0x00, 0x00, 0x00, 0x49, 0x81, 0xF9,
    0x3F, 0xF3, 0xFF, 0xFF, 0x0F, 0x87, 0xCF, 0x00, 0x00, 0x00, 0x49, 0x81,
    0xF8, 0x3F, 0xF1, 0xFF, 0xFF, 0x0F, 0x87, 0xC2, 0x00, 0x00, 0x00, 0x48,
    0x81, 0xFA, 0x7F, 0xFC, 0xFF, 0xFF, 0x0F, 0x87, 0xB5, 0x00, 0x00, 0x00,
    0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4,
    0x4C, 0x89, 0xD0, 0x49, 0x81, 0xC2, 0x80, 0x1B, 0x00, 0x00, 0x66, 0x2E,
    0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x01, 0xFC, 0x48,
    0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4, 0x4C, 0x89, 0xC8, 0x49,
    0x81, 0xC1, 0xC0, 0x0C, 0x00, 0x00, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C,
    0x39, 0xC8, 0x72, 0xF4, 0x4C, 0x89, 0xC0, 0x49, 0x81, 0xC0, 0xC0, 0x0E,
    0x00, 0x00, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xC0, 0x72, 0xF4,
    0x48, 0x89, 0xD0, 0x48, 0x81, 0xC2, 0x80, 0x03, 0x00, 0x00, 0x66, 0x2E,
    0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x01, 0xFC, 0x48,
    0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4, 0xC5, 0xF8, 0x57, 0xC0,
    0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xC5, 0xFC,
    0x29, 0x81, 0x40, 0xDA, 0x00, 0x00, 0xC5, 0xFC, 0x29, 0x81, 0xC0, 0xDA,
    0x00, 0x00, 0x0F, 0xAE, 0xF8, 0x48, 0x89, 0x41, 0x20, 0xC5, 0xF8, 0x77,
    0xC3, 0x48, 0x3D, 0x7F, 0xC6, 0xFF, 0xFF, 0x77, 0x0C, 0x0F, 0x01, 0xFC,
    0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4, 0x49, 0x81, 0xFA,
    0x7F, 0xE4, 0xFF, 0xFF, 0x77, 0x19, 0x4D, 0x89, 0xD3, 0x49, 0x81, 0xC3,
    0x80, 0x1B, 0x00, 0x00, 0x4C, 0x89, 0xD0, 0x49, 0x83, 0xC2, 0x40, 0x0F,
    0x01, 0xFC, 0x4D, 0x39, 0xDA, 0x72, 0xF1, 0x49, 0x81, 0xF9, 0x3F, 0xF3,
    0xFF, 0xFF, 0x77, 0x16, 0x4D, 0x8D, 0x91, 0xC0, 0x0C, 0x00, 0x00, 0x4C,
    0x89, 0xC8, 0x49, 0x83, 0xC1, 0x40, 0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xD1,
    0x72, 0xF1, 0x49, 0x81, 0xF8, 0x3F, 0xF1, 0xFF, 0xFF, 0x77, 0x16, 0x4D,
    0x8D, 0x88, 0xC0, 0x0E, 0x00, 0x00, 0x4C, 0x89, 0xC0, 0x49, 0x83, 0xC0,
    0x40, 0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xC8, 0x72, 0xF1, 0x48, 0x81, 0xFA,
    0x7F, 0xFC, 0xFF, 0xFF, 0x0F, 0x87, 0x56, 0xFF, 0xFF, 0xFF, 0x4C, 0x8D,
    0x82, 0x80, 0x03, 0x00, 0x00, 0x48, 0x89, 0xD0, 0x48, 0x83, 0xC2, 0x40,
    0x0F, 0x01, 0xFC, 0x4C, 0x39, 0xC2, 0x72, 0xF1, 0xE9, 0x3B, 0xFF, 0xFF,
    0xFF,
];
const SCARLETT_CLEAR_AVX: [u8; 42] = [
    0x48, 0x8D, 0x41, 0x40, 0x48, 0x83, 0xE0, 0xC0, 0x48, 0x8D, 0x90, 0x00,
    0x74, 0x00, 0x00, 0xC5, 0xFD, 0xEF, 0xC0, 0xC5, 0xFE, 0x7F, 0x00, 0xC5,
    0xFE, 0x7F, 0x40, 0x20, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72,
    0xEE, 0xE9, 0x62, 0x00, 0x00, 0x00,
];

/// Rewrite the routine at `at` when its bytes match exactly.
///
/// # Safety
/// `at..at + SCARLETT_CLEAR.len()` must be mapped, and no thread may be
/// executing the replaced bytes.
unsafe fn patch_scarlett_clear(at: usize) -> Result<(), &'static str> {
    use windows::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows::Win32::System::Memory::{
        PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS, VirtualProtect,
    };
    use windows::Win32::System::Threading::GetCurrentProcess;

    let (buf, avail) = read_bytes(at, SCARLETT_CLEAR.len());
    if buf[..avail] != SCARLETT_CLEAR {
        return Err("routine bytes differ");
    }
    let target = at + SCARLETT_CLEAR_FAST_PATH;
    let len = SCARLETT_CLEAR_AVX.len();
    let mut old = PAGE_PROTECTION_FLAGS(0);
    // SAFETY: the range is mapped (just read) and no thread executes it.
    unsafe {
        VirtualProtect(target as *const c_void, len, PAGE_EXECUTE_READWRITE, &mut old)
            .map_err(|_| "page protection refused")?;
        core::ptr::copy_nonoverlapping(SCARLETT_CLEAR_AVX.as_ptr(), target as *mut u8, len);
        let _ = FlushInstructionCache(GetCurrentProcess(), Some(target as *const c_void), len);
        let mut ignored = PAGE_PROTECTION_FLAGS(0);
        let _ = VirtualProtect(target as *const c_void, len, old, &mut ignored);
    }
    Ok(())
}

/// Copy up to `len` committed, readable bytes at `at`, as the trap path does.
fn read_bytes(at: usize, len: usize) -> ([u8; SCARLETT_CLEAR.len()], usize) {
    let mut buf = [0u8; SCARLETT_CLEAR.len()];
    let mut copied = 0;
    while copied < len.min(buf.len()) {
        let (chunk, avail) = crate::clwb::read_code_bytes(at + copied, at + len);
        if avail == 0 {
            break;
        }
        let take = avail.min(buf.len() - copied);
        buf[copied..copied + take].copy_from_slice(&chunk[..take]);
        copied += take;
    }
    (buf, copied)
}

/// Replace known hot `clzero` routines in the title image `[start, end)` so
/// they never trap. Runs when `d3d12_xs.dll` attaches, before title code.
pub(crate) fn patch_known_routines(start: usize, end: usize) {
    if host_has_clzero() || !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let Some(at) = start.checked_add(SCARLETT_CLEAR_RVA) else { return };
    if at.checked_add(SCARLETT_CLEAR.len()).is_none_or(|routine_end| routine_end > end) {
        return;
    }
    // SAFETY: inside the title image, and the title has not started running.
    match unsafe { patch_scarlett_clear(at) } {
        Ok(()) => crate::diagnostics::log(&format!(
            "replaced clzero loops at RVA {SCARLETT_CLEAR_RVA:#x} with AVX stores (host lacks CLZERO)"
        )),
        Err("routine bytes differ") => {}
        Err(reason) => crate::diagnostics::log(&format!(
            "clzero routine at RVA {SCARLETT_CLEAR_RVA:#x} left to the trap fallback: {reason}"
        )),
    }
}

#[cfg(test)]
#[path = "tests/clzero.rs"]
mod tests;
