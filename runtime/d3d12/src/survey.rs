//! Survey mode (`TUXBOX_SURVEY=1`): keep a title running past recoverable
//! failures so one launch reports many of them instead of stopping at the first.
//!
//! - A title `int3` assert is stepped over. The shipping code follows each such
//!   break with its own recovery path (typically returning null), so the title
//!   continues as it would after a debugger resumed it.
//! - An unmapped D3D12.x vtable slot returns `E_NOTIMPL` instead of failing fast.
//!
//! Each distinct site is logged once, with a running hit count at powers of two.
//! This is a diagnostic mode: rendering may be wrong after the first failure.
use std::sync::{LazyLock, Mutex, Once};
use windows::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_BREAKPOINT, HMODULE};
use windows::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_POINTERS, RtlCaptureStackBackTrace, RtlPcToFileHeader,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;

const CONTINUE_EXECUTION: i32 = -1;
const CONTINUE_SEARCH: i32 = 0;

pub(crate) fn enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("TUXBOX_SURVEY").is_some_and(|v| v == "1"));
    *ENABLED
}

/// Install the assert handler once, when the title creates its first device.
/// Outside survey mode it only steps over the asserts in `RECOVERABLE`.
pub(crate) fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        relax_hang_watchdog();
        let handle = unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
        if enabled() {
            log(format!("survey mode active (assert handler installed: {})", !handle.is_null()));
        }
    });
}

/// LOCAL: ProjectScarlett 1.0.2.1 RAGE hang watchdog (thread at RVA 0x2215a60).
/// Two branches reach its fatal handler: a registered thread missing its
/// heartbeat threshold (jae at 0x2215bdc) and the main-loop counter at
/// 0x71b6338 not advancing for 30 checks (je at 0x2215c70). Under translation
/// these stalls last longer than on console, so both branches become NOPs.
/// `TUXBOX_WATCHDOG=1` keeps the watchdog.
fn relax_hang_watchdog() {
    use windows::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS, VirtualProtect};
    // (RVA of the matched bytes, expected bytes, offset of the 2-byte branch)
    const SITES: [(usize, [u8; 6], usize); 2] = [
        (0x2215bd8, [0xc5, 0x78, 0x2e, 0xc7, 0x73, 0x2e], 4),
        (0x2215c6e, [0xff, 0xcd, 0x74, 0x04, 0x89, 0xeb], 2),
    ];
    if std::env::var_os("TUXBOX_WATCHDOG").is_some_and(|v| v == "1") { return; }
    let Some((start, end)) = title_image() else { return };
    for (rva, expected, branch) in SITES {
        let at = start + rva;
        if at + expected.len() > end { continue; }
        // SAFETY: the range lies within the mapped title image.
        if unsafe { core::slice::from_raw_parts(at as *const u8, expected.len()) } != expected { continue; }
        let target = (at + branch) as *mut u8;
        let mut old = PAGE_PROTECTION_FLAGS::default();
        // SAFETY: two bytes of the title's own code, made writable only for this store.
        unsafe {
            if VirtualProtect(target.cast(), 2, PAGE_EXECUTE_READWRITE, &mut old).is_ok() {
                target.write(0x90);
                target.add(1).write(0x90);
                let _ = VirtualProtect(target.cast(), 2, old, &mut old);
                crate::diagnostics::log_once(&Once::new(), || format!("relaxed the title hang watchdog at RVA {:#x}", rva + branch));
            }
        }
    }
}

/// Title asserts whose own continue path is correct for TuxBox, matched by
/// RVA and by the exact bytes there, so another build cannot match.
///
/// ProjectScarlett 1.0.2.1, RVA 0x4ffd879: the tile-heap mapping loop at
/// 0x4ffd520 asserts when a request is shorter than a whole flagged page run
/// (an 8 KiB request against a 16 KiB run about 60 s after player selection).
/// The continue path maps min(run, remaining), which is what was requested.
///
/// LOCAL: the same loop's other consistency asserts (0x4ffd859 request past the
/// run table's sentinel, 0x4ffd85f request before the run, 0x4ffd868) fire in
/// the open world once the tile heap's run table is shorter than its requests.
/// Root cause (a short table) is not found yet; continuing may map wrong pages.
const RECOVERABLE: [(usize, [u8; 9]); 4] = [
    (0x4ffd879, [0xcc, 0x48, 0x8b, 0x1f, 0xe9, 0xf0, 0xfd, 0xff, 0xff]),
    (0x4ffd859, [0xcc, 0xe9, 0xa5, 0xfd, 0xff, 0xff, 0xcc, 0x48, 0x8b]),
    (0x4ffd85f, [0xcc, 0x48, 0x8b, 0x1f, 0xe9, 0xc1, 0xfd, 0xff, 0xff]),
    (0x4ffd868, [0xcc, 0x48, 0x8b, 0x1f, 0x89, 0xda, 0x81, 0xe2, 0x00]),
];

fn recoverable(start: usize, end: usize, at: usize) -> bool {
    RECOVERABLE.iter().any(|&(rva, bytes)| {
        at == start + rva && at + bytes.len() <= end
            // SAFETY: the range lies within the mapped title image.
            && unsafe { core::slice::from_raw_parts(at as *const u8, bytes.len()) } == bytes
    })
}

/// ProjectScarlett 1.0.2.1, RVA 0x51d251e: the tile-run counter at 0x51d2470
/// asserts when the bytes still to cover outlast the heap's page-run table.
/// Its own continue path reads past the table, so instead count the remainder
/// as one more run and return, as the loop would for a final partial run.
const RUN_COUNT: (usize, usize, [u8; 9]) =
    (0x51d251e, 0x51d250e, [0xcc, 0xeb, 0xc5, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc]);

fn run_count_overflow(start: usize, end: usize, at: usize, context: &mut windows::Win32::System::Diagnostics::Debug::CONTEXT) -> bool {
    let (rva, ret, bytes) = RUN_COUNT;
    if at != start + rva || at + bytes.len() > end
        // SAFETY: the range lies within the mapped title image.
        || unsafe { core::slice::from_raw_parts(at as *const u8, bytes.len()) } != bytes
        || unsafe { *((start + ret) as *const u8) } != 0xc3
    {
        return false;
    }
    context.Rax = u64::from((context.Rax as u32).wrapping_add(1));
    context.Rip = (start + ret) as u64;
    true
}

fn log(line: String) {
    crate::diagnostics::log_once(&Once::new(), || line);
}

/// Bounds of the title image (the process executable).
fn title_image() -> Option<(usize, usize)> {
    static IMAGE: LazyLock<Option<(usize, usize)>> = LazyLock::new(|| {
        let HMODULE(base) = unsafe { GetModuleHandleW(None) }.ok()?;
        let base = base as usize;
        // SAFETY: the loaded executable starts with valid DOS and PE32+ headers.
        unsafe {
            let nt = base + (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
            // Signature (4) + file header (20) + SizeOfImage offset in the optional header (56).
            let size = (nt as *const u8).add(4 + 20 + 56).cast::<u32>().read_unaligned() as usize;
            Some((base, base + size))
        }
    });
    *IMAGE
}

/// Count a hit for `site`. Returns the new count.
fn record(table: &Mutex<Vec<(usize, u64)>>, site: usize) -> u64 {
    let mut sites = table.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = sites.iter_mut().find(|(at, _)| *at == site) {
        entry.1 += 1;
        return entry.1;
    }
    sites.push((site, 1));
    1
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    static ASSERTS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    let Some(info) = (unsafe { info.as_mut() }) else { return CONTINUE_SEARCH };
    let (Some(record_), Some(context)) = (unsafe { info.ExceptionRecord.as_ref() }, unsafe { info.ContextRecord.as_mut() })
    else {
        return CONTINUE_SEARCH;
    };
    if record_.ExceptionCode == EXCEPTION_ACCESS_VIOLATION {
        if enabled() {
            describe_fault(context);
        }
        return CONTINUE_SEARCH;
    }
    if record_.ExceptionCode != EXCEPTION_BREAKPOINT {
        return CONTINUE_SEARCH;
    }
    let Some((start, end)) = title_image() else { return CONTINUE_SEARCH };
    let rip = context.Rip as usize;
    if !(start..end).contains(&rip) {
        return CONTINUE_SEARCH;
    }
    if run_count_overflow(start, end, rip, context) {
        let hits = record(&ASSERTS, rip - start);
        if hits.is_power_of_two() {
            crate::diagnostics::log_once(&Once::new(), || {
                format!("closed tile-run count past the page table at RVA {:#x} (hit {hits})", rip - start)
            });
        }
        return CONTINUE_EXECUTION;
    }
    if !enabled() {
        if !recoverable(start, end, rip) {
            return CONTINUE_SEARCH;
        }
        context.Rip += 1;
        let hits = record(&ASSERTS, rip - start);
        if hits.is_power_of_two() {
            crate::diagnostics::log_once(&Once::new(), || {
                format!("continued past recoverable title assert RVA {:#x} (hit {hits})", rip - start)
            });
        }
        return CONTINUE_EXECUTION;
    }
    // Windows reports an int3 at its own address; resume after it.
    // SAFETY: rip lies within the mapped title image.
    let at = if unsafe { *(rip as *const u8) } == 0xcc {
        context.Rip += 1;
        rip
    } else if rip > start && unsafe { *((rip - 1) as *const u8) } == 0xcc {
        rip - 1
    } else {
        return CONTINUE_SEARCH;
    };
    let hits = record(&ASSERTS, at - start);
    if hits.is_power_of_two() {
        log(format!("survey: stepped over title assert RVA {:#x} (hit {hits})", at - start));
    }
    CONTINUE_EXECUTION
}

/// Return address of the first title frame on this thread, as an RVA.
pub(crate) fn title_rva() -> Option<usize> {
    let mut frames = [core::ptr::null_mut(); 32];
    let count = unsafe { RtlCaptureStackBackTrace(0, &mut frames, None) } as usize;
    let (start, _) = title_image()?;
    frames[..count].iter().find_map(|&pc| {
        let mut base = core::ptr::null_mut();
        unsafe { RtlPcToFileHeader(pc, &mut base) };
        (base as usize == start).then(|| pc as usize - start)
    })
}

/// Log a call to an unmapped vtable slot (once per slot and caller).
pub(crate) fn unsupported_slot(slot: usize) {
    static SLOTS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    let rva = title_rva().unwrap_or(0);
    let hits = record(&SLOTS, (slot << 40) | rva);
    if hits.is_power_of_two() {
        log(format!("survey: unsupported D3D12.x vtable slot {slot} from title RVA {rva:#x} (hit {hits})"));
    }
}

/// Log the memory region behind each register of a title access violation.
fn describe_fault(context: &windows::Win32::System::Diagnostics::Debug::CONTEXT) {
    use windows::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static FAULTS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    let Some((start, end)) = title_image() else { return };
    let rip = context.Rip as usize;
    if !(start..end).contains(&rip) || record(&FAULTS, rip - start) != 1 {
        return;
    }
    let registers = [("rax", context.Rax), ("rbx", context.Rbx), ("rcx", context.Rcx), ("rdx", context.Rdx),
        ("rsi", context.Rsi), ("rdi", context.Rdi), ("r8", context.R8), ("r9", context.R9),
        ("r12", context.R12), ("r13", context.R13), ("r14", context.R14), ("r15", context.R15)];
    let mut line = format!("survey: title access violation at RVA {:#x};", rip - start);
    for (name, value) in registers {
        let mut region = MEMORY_BASIC_INFORMATION::default();
        if value < 0x10000 || unsafe { VirtualQuery(Some(value as _), &mut region, size_of_val(&region)) } == 0
            || region.State.0 != 0x1000 {
            continue;
        }
        line += &format!(" {name}={value:#x} in allocation {:p} (region {:p}+{:#x} protect={:#x} type={:#x});",
            region.AllocationBase, region.BaseAddress, region.RegionSize, region.Protect.0, region.Type.0);
    }
    log(line);
    // Extent of a repeated 16-byte pattern at rcx (a fill over title memory).
    'fill: {
        let at = (context.Rcx as usize) & !0xf;
        let mut region = MEMORY_BASIC_INFORMATION::default();
        if unsafe { VirtualQuery(Some(at as _), &mut region, size_of_val(&region)) } == 0
            || region.State.0 != 0x1000 || region.Protect.0 & 0x66 == 0 {
            break 'fill;
        }
        let (low, high) = (region.BaseAddress as usize, region.BaseAddress as usize + region.RegionSize);
        // SAFETY: every read below stays inside the committed readable region.
        let read = |a: usize| unsafe { (a as *const u128).read_unaligned() };
        let pattern = read(at);
        let (mut start, mut end) = (at, at + 16);
        while start >= low + 16 && read(start - 16) == pattern { start -= 16; }
        while end + 16 <= high && read(end) == pattern { end += 16; }
        log(format!("survey: fill {pattern:#034x} covers {start:#x}..{end:#x} ({:#x} bytes; before {:#034x} after {:#034x})",
            end - start, if start >= low + 16 { read(start - 16) } else { 0 }, if end + 16 <= high { read(end) } else { 0 }));
    }
    // Contents around each pointer register that lies in readable memory.
    for (name, value) in [("rax", context.Rax), ("rcx", context.Rcx), ("rsi", context.Rsi)] {
        let base = (value as usize).saturating_sub(0x80) & !0xf;
        let mut region = MEMORY_BASIC_INFORMATION::default();
        if unsafe { VirtualQuery(Some(base as _), &mut region, size_of_val(&region)) } == 0
            || region.State.0 != 0x1000 || region.Protect.0 & 0x66 == 0
            || base + 0x180 > region.BaseAddress as usize + region.RegionSize {
            continue;
        }
        for row in (0..0x180).step_by(32) {
            // SAFETY: the committed readable region covers these bytes.
            let bytes = unsafe { core::slice::from_raw_parts((base + row) as *const u8, 32) };
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
            log(format!("survey: {name} {:#x}: {hex}", base + row));
        }
    }
}
