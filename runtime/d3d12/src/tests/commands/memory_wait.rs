use super::*;
use std::sync::atomic::AtomicBool;
use std::thread;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIAdapter1, IDXGIFactory1,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{PCSTR, s, w};

#[test]
fn all_eight_comparisons_evaluate_exact_u64_predicates() {
    assert_eq!(Comparison::from_raw(0), Err(E_INVALIDARG.into()));
    assert_eq!(Comparison::from_raw(9), Err(E_INVALIDARG.into()));
    assert_eq!(Comparison::from_raw(u32::MAX), Err(E_INVALIDARG.into()));

    let ops = [
        (1, Comparison::Never, false, "NEVER"),
        (2, Comparison::Less, false, "<"),
        (3, Comparison::Equal, false, "=="),
        (4, Comparison::LessEqual, false, "<="),
        (5, Comparison::Greater, true, ">"),
        (6, Comparison::NotEqual, false, "!="),
        (7, Comparison::GreaterEqual, true, ">="),
        (8, Comparison::Always, false, "ALWAYS"),
    ];
    for (raw, expected, monotone, sym) in ops {
        let parsed = Comparison::from_raw(raw).expect("valid D3D12 comparison");
        assert_eq!(parsed, expected);
        assert_eq!(parsed.is_monotone(), monotone);
        assert_eq!(parsed.symbol(), sym);
    }

    let cases = [
        (0u64, 0u64),
        (0, 1),
        (1, 0),
        (42, 42),
        (41, 42),
        (43, 42),
        (0x1_0000_0000, 0x1_0000_0001),
        (u64::MAX - 1, u64::MAX),
        (u64::MAX, u64::MAX),
    ];
    for (cur, tgt) in cases {
        assert!(!Comparison::Never.matches(cur, tgt));
        assert_eq!(Comparison::Less.matches(cur, tgt), cur < tgt);
        assert_eq!(Comparison::Equal.matches(cur, tgt), cur == tgt);
        assert_eq!(Comparison::LessEqual.matches(cur, tgt), cur <= tgt);
        assert_eq!(Comparison::Greater.matches(cur, tgt), cur > tgt);
        assert_eq!(Comparison::NotEqual.matches(cur, tgt), cur != tgt);
        assert_eq!(Comparison::GreaterEqual.matches(cur, tgt), cur >= tgt);
        assert!(Comparison::Always.matches(cur, tgt));
    }

    assert!(Comparison::Always.is_always_satisfied(0));
    assert!(Comparison::Always.is_always_satisfied(u64::MAX));
    assert!(Comparison::GreaterEqual.is_always_satisfied(0));
    assert!(!Comparison::GreaterEqual.is_always_satisfied(1));
    assert!(Comparison::LessEqual.is_always_satisfied(u64::MAX));
    assert!(!Comparison::LessEqual.is_always_satisfied(0));
    assert!(!Comparison::Equal.is_always_satisfied(0));
    assert!(!Comparison::Less.is_always_satisfied(0));
    assert!(!Comparison::Greater.is_always_satisfied(0));
    assert!(!Comparison::NotEqual.is_always_satisfied(0));
    assert!(!Comparison::Never.is_always_satisfied(0));
}

#[test]
fn elision_matrix_forbids_unsafe_cases() {
    let all_ops = [
        Comparison::Never,
        Comparison::Less,
        Comparison::Equal,
        Comparison::LessEqual,
        Comparison::Greater,
        Comparison::NotEqual,
        Comparison::GreaterEqual,
        Comparison::Always,
    ];
    for has_prefix in [false, true] {
        for op in all_ops {
            for prior_write in [false, true] {
                for reached in [false, true] {
                    let expected = !has_prefix && op.is_monotone() && !prior_write && reached;
                    assert_eq!(
                        can_elide_wait(has_prefix, op, prior_write, reached),
                        expected,
                        "has_prefix={has_prefix} op={op:?} prior_write={prior_write} reached={reached}"
                    );
                }
            }
        }
    }

    let mut writes = WrittenRanges::default();
    let word = 0x1_0000_1000u64;
    writes.record(word - 4, 4);
    writes.record(word + 8, 8);
    assert!(!writes.overlaps_word(word));

    writes.record(word - 4, 5);
    assert!(writes.overlaps_word(word));
    writes.clear();
    assert!(!writes.overlaps_word(word));

    writes.record(word + 7, 1);
    assert!(writes.overlaps_word(word));
    writes.clear();

    writes.record_unknown();
    assert!(writes.overlaps_word(word));
    assert!(writes.overlaps_word(0x9999_0000));
    writes.clear();
    assert!(!writes.overlaps_word(word));

    for i in 0..=(WrittenRanges::MAX_RANGES as u64) {
        writes.record(0x10_0000 + i * 0x100, 8);
    }
    assert!(writes.overlaps_word(0x9999_0000));
}

#[test]
fn monotonic_fence_pairs_are_strictly_increasing_and_overflow_safe() {
    let counter = AtomicU64::new(0);
    assert_eq!(next_monotonic_pair(&counter).unwrap(), (1, 2));
    assert_eq!(next_monotonic_pair(&counter).unwrap(), (3, 4));
    assert_eq!(next_monotonic_pair(&counter).unwrap(), (5, 6));

    let near_max = AtomicU64::new(u64::MAX - 4);
    assert_eq!(
        next_monotonic_pair(&near_max).unwrap(),
        (u64::MAX - 3, u64::MAX - 2)
    );
    assert!(next_monotonic_pair(&near_max).is_err());
}

#[test]
fn wake_notifier_wakes_immediately_and_avoids_lost_wakeups() {
    let notifier = Arc::new(Notifier::new());
    let flag = Arc::new(AtomicBool::new(false));

    let epoch = notifier.epoch();
    notifier.notify_all();
    let start = Instant::now();
    notifier.wait_timeout(epoch, Duration::from_millis(250));
    assert!(
        start.elapsed() < Duration::from_millis(50),
        "missed wakeup before wait_timeout"
    );

    let n2 = notifier.clone();
    let f2 = flag.clone();
    let notify_at = Arc::new(Mutex::new(None::<Instant>));
    let notify_at_bg = notify_at.clone();
    let epoch = notifier.epoch();
    let handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(5));
        f2.store(true, Ordering::Release);
        *notify_at_bg.lock().unwrap() = Some(Instant::now());
        n2.notify_all();
    });
    while !flag.load(Ordering::Acquire) {
        notifier.wait_timeout(epoch, Duration::from_millis(500));
    }
    let wake_elapsed = notify_at
        .lock()
        .unwrap()
        .expect("notifier recorded timestamp")
        .elapsed();
    handle.join().unwrap();
    assert!(
        wake_elapsed < Duration::from_millis(50),
        "wakeup after notify took too long: {wake_elapsed:?}"
    );
}

// --- Hardware D3D12 integration harness (exercises the real Xbox proxy ABI) ---

static GPU_LOCK: Mutex<()> = Mutex::new(());

struct ComObj(*mut c_void);

impl Drop for ComObj {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let release: unsafe extern "system" fn(*mut c_void) -> u32 = vslot(self.0, 2);
                release(self.0);
            }
        }
    }
}

unsafe fn vslot<F: Copy>(obj: *mut c_void, slot: usize) -> F {
    unsafe {
        let vtable = *(obj as *const *const *const c_void);
        core::mem::transmute_copy(&*vtable.add(slot))
    }
}

struct GuestMem {
    base: *mut u8,
}

impl GuestMem {
    fn new(bytes: usize) -> Self {
        let base = unsafe { VirtualAlloc(None, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        assert!(!base.is_null(), "VirtualAlloc failed");
        Self { base: base.cast() }
    }

    fn addr(&self, offset: usize) -> u64 {
        self.base.addr() as u64 + offset as u64
    }

    fn write64(&self, offset: usize, val: u64) {
        unsafe {
            AtomicU64::from_ptr(self.base.add(offset).cast::<u64>()).store(val, Ordering::Release);
        }
    }

    fn read64(&self, offset: usize) -> u64 {
        unsafe { AtomicU64::from_ptr(self.base.add(offset).cast::<u64>()).load(Ordering::Acquire) }
    }

    fn write32(&self, offset: usize, val: u32) {
        unsafe {
            std::sync::atomic::AtomicU32::from_ptr(self.base.add(offset).cast::<u32>())
                .store(val, Ordering::Release);
        }
    }

    fn read32(&self, offset: usize) -> u32 {
        unsafe {
            std::sync::atomic::AtomicU32::from_ptr(self.base.add(offset).cast::<u32>())
                .load(Ordering::Acquire)
        }
    }
}

impl Drop for GuestMem {
    fn drop(&mut self) {
        unsafe {
            let _ = VirtualFree(self.base.cast(), 0, MEM_RELEASE);
        }
    }
}

fn create_hardware_device() -> Option<ComObj> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1().ok()? };
    let mut fallback: Option<IDXGIAdapter1> = None;
    let mut chosen: Option<IDXGIAdapter1> = None;
    let mut idx = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(idx) } {
        idx += 1;
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        if desc.VendorId == 0x10de && desc.DeviceId == 0x2783 {
            chosen = Some(adapter);
            break;
        }
        if fallback.is_none() {
            fallback = Some(adapter);
        }
    }
    let adapter = chosen.or(fallback)?;
    let profile: [u32; 18] = [
        0x20004, 0, u32::MAX, 0, 0, 0, u32::MAX, u32::MAX, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mut raw = ptr::null_mut();
    let hr = unsafe {
        crate::create_device(
            adapter.as_raw(),
            profile.as_ptr().cast(),
            &ID3D12Device11::IID,
            &mut raw,
        )
    };
    assert!(hr.is_ok(), "D3D12XboxCreateDevice failed: {hr:?}");
    Some(ComObj(raw))
}

struct GuestList {
    allocator: ComObj,
    list: ComObj,
}

impl GuestList {
    fn new(device: &ComObj, kind: D3D12_COMMAND_LIST_TYPE) -> Self {
        let mut alloc = ptr::null_mut();
        let create_alloc: unsafe extern "system" fn(
            *mut c_void,
            D3D12_COMMAND_LIST_TYPE,
            *const windows::core::GUID,
            *mut *mut c_void,
        ) -> HRESULT = unsafe { vslot(device.0, 9) };
        let hr = unsafe { create_alloc(device.0, kind, &ID3D12CommandAllocator::IID, &mut alloc) };
        assert!(hr.is_ok(), "CreateCommandAllocator failed: {hr:?}");

        let desc = [kind.0 as u32, 0, 1, 0];
        let mut list = ptr::null_mut();
        let create_list: unsafe extern "system" fn(
            *mut c_void,
            *const u32,
            *mut c_void,
            *mut c_void,
            *const windows::core::GUID,
            *mut *mut c_void,
        ) -> HRESULT = unsafe { vslot(device.0, 73) };
        let hr = unsafe {
            create_list(
                device.0,
                desc.as_ptr(),
                alloc,
                ptr::null_mut(),
                &ID3D12GraphicsCommandList::IID,
                &mut list,
            )
        };
        assert!(hr.is_ok(), "CreateCommandListX failed: {hr:?}");
        let gl = Self {
            allocator: ComObj(alloc),
            list: ComObj(list),
        };
        gl.reset_open();
        gl
    }

    fn reset_open(&self) {
        let reset: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> HRESULT =
            unsafe { vslot(self.list.0, 10) };
        let hr = unsafe { reset(self.list.0, self.allocator.0, ptr::null_mut()) };
        assert!(hr.is_ok(), "List::Reset failed: {hr:?}");
    }

    fn reopen(&self) {
        let reset_alloc: unsafe extern "system" fn(*mut c_void) -> HRESULT =
            unsafe { vslot(self.allocator.0, 8) };
        let hr = unsafe { reset_alloc(self.allocator.0) };
        assert!(hr.is_ok(), "Allocator::Reset failed: {hr:?}");
        self.reset_open();
    }

    fn close(&self) -> HRESULT {
        let close: unsafe extern "system" fn(*mut c_void) -> HRESULT =
            unsafe { vslot(self.list.0, 9) };
        unsafe { close(self.list.0) }
    }

    fn write64(&self, addr: u64, value: u64) {
        let f: unsafe extern "system" fn(*mut c_void, u64, u64, u32, u32) =
            unsafe { vslot(self.list.0, 72) };
        unsafe { f(self.list.0, addr, value, 0, 0) };
    }

    fn wait64(&self, addr: u64, comparison: u32, target: u64) {
        let f: unsafe extern "system" fn(*mut c_void, u64, u32, u64, u32) =
            unsafe { vslot(self.list.0, 74) };
        unsafe { f(self.list.0, addr, comparison, target, 0) };
    }

    fn copy_memory(&self, dst: u64, src: u64, bytes: u64) {
        let f: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u32) =
            unsafe { vslot(self.list.0, 62) };
        unsafe { f(self.list.0, dst, src, bytes, 0) };
    }

    fn set_compute_root(&self, root: &ComObj) {
        let f: unsafe extern "system" fn(*mut c_void, *mut c_void) =
            unsafe { vslot(self.list.0, 33) };
        unsafe { f(self.list.0, root.0) };
    }

    fn set_pipeline(&self, pso: &ComObj) {
        let f: unsafe extern "system" fn(*mut c_void, *mut c_void) =
            unsafe { vslot(self.list.0, 25) };
        unsafe { f(self.list.0, pso.0) };
    }

    fn set_compute_const(&self, slot: u32, value: u32, offset: u32) {
        let f: unsafe extern "system" fn(*mut c_void, u32, u32, u32) =
            unsafe { vslot(self.list.0, 37) };
        unsafe { f(self.list.0, slot, value, offset) };
    }

    fn set_compute_srv(&self, slot: u32, addr: u64) {
        let f: unsafe extern "system" fn(*mut c_void, u32, u64) =
            unsafe { vslot(self.list.0, 43) };
        unsafe { f(self.list.0, slot, addr) };
    }

    fn set_compute_uav(&self, slot: u32, addr: u64) {
        let f: unsafe extern "system" fn(*mut c_void, u32, u64) =
            unsafe { vslot(self.list.0, 45) };
        unsafe { f(self.list.0, slot, addr) };
    }

    fn dispatch(&self, x: u32, y: u32, z: u32) {
        let f: unsafe extern "system" fn(*mut c_void, u32, u32, u32) =
            unsafe { vslot(self.list.0, 14) };
        unsafe { f(self.list.0, x, y, z) };
    }

    fn set_direct_state_probe(&self) {
        let topo: unsafe extern "system" fn(*mut c_void, windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY) =
            unsafe { vslot(self.list.0, 20) };
        let vp: unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT) =
            unsafe { vslot(self.list.0, 21) };
        let sc: unsafe extern "system" fn(*mut c_void, u32, *const RECT) =
            unsafe { vslot(self.list.0, 22) };
        let bf: unsafe extern "system" fn(*mut c_void, *const f32) =
            unsafe { vslot(self.list.0, 23) };
        let sr: unsafe extern "system" fn(*mut c_void, u32) =
            unsafe { vslot(self.list.0, 24) };
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: 64.0,
            Height: 64.0,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let rect = RECT { left: 0, top: 0, right: 64, bottom: 64 };
        let blend = [0.25f32, 0.5, 0.75, 1.0];
        unsafe {
            topo(self.list.0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            vp(self.list.0, 1, &viewport);
            sc(self.list.0, 1, &rect);
            bf(self.list.0, blend.as_ptr());
            sr(self.list.0, 0x7);
        }
    }
}

struct GuestQueue {
    queue: ComObj,
    fence: ComObj,
    submitted: u64,
}

impl GuestQueue {
    fn new(device: &ComObj, kind: D3D12_COMMAND_LIST_TYPE) -> Self {
        let desc = D3D12_COMMAND_QUEUE_DESC {
            Type: kind,
            ..Default::default()
        };
        let mut queue = ptr::null_mut();
        let create_q: unsafe extern "system" fn(
            *mut c_void,
            *const D3D12_COMMAND_QUEUE_DESC,
            *const windows::core::GUID,
            *mut *mut c_void,
        ) -> HRESULT = unsafe { vslot(device.0, 8) };
        let hr = unsafe { create_q(device.0, &desc, &ID3D12CommandQueue::IID, &mut queue) };
        assert!(hr.is_ok(), "CreateCommandQueue failed: {hr:?}");

        let mut fence = ptr::null_mut();
        let create_f: unsafe extern "system" fn(
            *mut c_void,
            u64,
            D3D12_FENCE_FLAGS,
            *const windows::core::GUID,
            *mut *mut c_void,
        ) -> HRESULT = unsafe { vslot(device.0, 36) };
        let hr = unsafe {
            create_f(
                device.0,
                0,
                D3D12_FENCE_FLAG_NONE,
                &ID3D12Fence::IID,
                &mut fence,
            )
        };
        assert!(hr.is_ok(), "CreateFence failed: {hr:?}");
        Self {
            queue: ComObj(queue),
            fence: ComObj(fence),
            submitted: 0,
        }
    }

    fn execute(&mut self, lists: &[&GuestList]) -> u64 {
        let raws: Vec<*mut c_void> = lists.iter().map(|l| l.list.0).collect();
        let exec: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
            unsafe { vslot(self.queue.0, 10) };
        unsafe { exec(self.queue.0, raws.len() as u32, raws.as_ptr()) };
        self.submitted += 1;
        // Device11 queue Signal is slot 16.
        let sig: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT =
            unsafe { vslot(self.queue.0, 16) };
        let hr = unsafe { sig(self.queue.0, self.fence.0, self.submitted) };
        assert!(hr.is_ok(), "Queue::Signal failed: {hr:?}");
        self.submitted
    }

    fn completed(&self) -> u64 {
        let f: unsafe extern "system" fn(*mut c_void) -> u64 = unsafe { vslot(self.fence.0, 8) };
        unsafe { f(self.fence.0) }
    }

    fn wait_done(&self, target: u64, timeout_ms: u32) -> bool {
        if self.completed() >= target {
            return true;
        }
        let event = unsafe { CreateEventW(None, false, false, None).expect("CreateEventW") };
        let set_ev: unsafe extern "system" fn(*mut c_void, u64, HANDLE) -> HRESULT =
            unsafe { vslot(self.fence.0, 9) };
        let hr = unsafe { set_ev(self.fence.0, target, event) };
        assert!(hr.is_ok(), "SetEventOnCompletion failed: {hr:?}");
        let res = unsafe { WaitForSingleObject(event, timeout_ms) };
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(event);
        }
        res.0 == 0 && self.completed() >= target
    }
}

fn compile_compute_shader(hlsl: &str) -> Vec<u8> {
    type D3DCompileFn = unsafe extern "system" fn(
        *const c_void,
        usize,
        PCSTR,
        *const c_void,
        *mut c_void,
        PCSTR,
        PCSTR,
        u32,
        u32,
        *mut Option<ID3DBlob>,
        *mut Option<ID3DBlob>,
    ) -> HRESULT;
    let module = unsafe { LoadLibraryW(w!("d3dcompiler_47.dll")).expect("load d3dcompiler_47.dll") };
    let proc = unsafe { GetProcAddress(module, s!("D3DCompile")).expect("D3DCompile export") };
    let compile: D3DCompileFn = unsafe { core::mem::transmute(proc) };
    let mut code: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    let hr = unsafe {
        compile(
            hlsl.as_ptr().cast(),
            hlsl.len(),
            PCSTR::null(),
            ptr::null(),
            ptr::null_mut(),
            s!("main"),
            s!("cs_5_0"),
            0,
            0,
            &mut code,
            &mut err,
        )
    };
    if hr.is_err() {
        let msg = err
            .map(|b| unsafe {
                std::str::from_utf8(slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .unwrap_or("")
                .to_owned()
            })
            .unwrap_or_default();
        panic!("D3DCompile failed ({hr:?}): {msg}");
    }
    let blob = code.expect("compiled shader blob");
    unsafe {
        slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec()
    }
}

fn create_compute_pipeline(device: &ComObj) -> (ComObj, ComObj) {
    let hlsl = "
        cbuffer Params : register(b0) { uint mul; };
        StructuredBuffer<uint> inp : register(t0);
        RWStructuredBuffer<uint> outp : register(u0);
        [numthreads(1, 1, 1)]
        void main() { outp[0] = inp[0] * mul + 1; }
    ";
    let bytecode = compile_compute_shader(hlsl);

    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let sig_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    let mut blob: Option<ID3DBlob> = None;
    unsafe {
        D3D12SerializeRootSignature(&sig_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
            .expect("serialize root signature");
    }
    let blob = blob.expect("root signature blob");

    let mut root_raw = ptr::null_mut();
    let create_root: unsafe extern "system" fn(
        *mut c_void,
        u32,
        *const c_void,
        usize,
        *const windows::core::GUID,
        *mut *mut c_void,
    ) -> HRESULT = unsafe { vslot(device.0, 16) };
    let hr = unsafe {
        create_root(
            device.0,
            0,
            blob.GetBufferPointer(),
            blob.GetBufferSize(),
            &ID3D12RootSignature::IID,
            &mut root_raw,
        )
    };
    assert!(hr.is_ok(), "CreateRootSignature failed: {hr:?}");
    let root = ComObj(root_raw);

    let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: unsafe {
            std::mem::ManuallyDrop::new(Some(ID3D12RootSignature::from_raw(root.0)))
        },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytecode.as_ptr().cast(),
            BytecodeLength: bytecode.len(),
        },
        ..Default::default()
    };
    let mut pso_raw = ptr::null_mut();
    let create_pso: unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_COMPUTE_PIPELINE_STATE_DESC,
        u32,
        *const c_void,
        *const windows::core::GUID,
        *mut *mut c_void,
    ) -> HRESULT = unsafe { vslot(device.0, 62) };
    let hr = unsafe {
        create_pso(
            device.0,
            &pso_desc,
            0,
            ptr::null(),
            &ID3D12PipelineState::IID,
            &mut pso_raw,
        )
    };
    assert!(hr.is_ok(), "CreateComputePipelineStateX failed: {hr:?}");
    (root, ComObj(pso_raw))
}

#[test]
fn gpu_all_comparison_ops_and_non_poisoning() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(device) = create_hardware_device() else {
        return;
    };
    let mem = GuestMem::new(65536);
    let mut producer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let mut consumer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let prod_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let cons_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);

    // 1. NEVER (1) must record and close cleanly without poisoning the list or dropping writes.
    mem.write64(0, 0);
    cons_list.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_NEVER.0 as u32, 1);
    cons_list.write64(mem.addr(8), 0xCAFE);
    assert_eq!(cons_list.close(), S_OK, "NEVER must not fail Close()");

    // Invalid raw comparison (9) must fail Close() with E_INVALIDARG.
    cons_list.reopen();
    cons_list.wait64(mem.addr(0), 9, 1);
    assert_eq!(cons_list.close(), E_INVALIDARG);

    // Unsupported flags (> 1) must fail Close() with E_NOTIMPL.
    cons_list.reopen();
    let raw_wait: unsafe extern "system" fn(*mut c_void, u64, u32, u64, u32) =
        unsafe { vslot(cons_list.list.0, 74) };
    unsafe {
        raw_wait(
            cons_list.list.0,
            mem.addr(0),
            D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32,
            1,
            2,
        );
    }
    assert_eq!(cons_list.close(), E_NOTIMPL);

    // 2. ALWAYS (8) runs immediately with zero split and zero wait.
    cons_list.reopen();
    mem.write64(0, 0);
    mem.write64(8, 0);
    cons_list.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_ALWAYS.0 as u32, 999);
    cons_list.write64(mem.addr(8), 0xA105);
    assert_eq!(cons_list.close(), S_OK);
    let t = consumer.execute(&[&cons_list]);
    assert!(consumer.wait_done(t, 2000), "ALWAYS wait timed out");
    assert_eq!(mem.read64(8), 0xA105);

    // 3. Test all 6 conditional ops (LESS, EQUAL, LESS_EQUAL, GREATER, NOT_EQUAL, GREATER_EQUAL),
    // including EQUAL(0) so target == 0 is proven not to bypass non-GE predicates.
    let cases: &[(D3D12_COMPARISON_FUNC, u64, u64, u64, u64)] = &[
        (D3D12_COMPARISON_FUNC_LESS, 50, 20, 19, 0x2001),
        (D3D12_COMPARISON_FUNC_EQUAL, 100, 42, 42, 0x3001),
        (D3D12_COMPARISON_FUNC_EQUAL, 5, 0, 0, 0x3002),
        (D3D12_COMPARISON_FUNC_LESS_EQUAL, 50, 20, 20, 0x4001),
        (D3D12_COMPARISON_FUNC_GREATER, 10, 10, 11, 0x5001),
        (D3D12_COMPARISON_FUNC_NOT_EQUAL, 77, 77, 78, 0x6001),
        (D3D12_COMPARISON_FUNC_GREATER_EQUAL, 0, 10, 10, 0x7001),
    ];
    for (idx, &(func, initial, target, release_val, marker)) in cases.iter().enumerate() {
        cons_list.reopen();
        if idx > 0 {
            prod_list.reopen();
        }
        mem.write64(0, initial);
        mem.write64(8, 0);

        cons_list.wait64(mem.addr(0), func.0 as u32, target);
        cons_list.write64(mem.addr(8), marker);
        assert_eq!(cons_list.close(), S_OK);
        let ct = consumer.execute(&[&cons_list]);

        // Negative check: must still be held before producer writes release_val.
        thread::sleep(Duration::from_millis(15));
        assert_eq!(
            mem.read64(8),
            0,
            "consumer bypassed unsatisfied comparison {func:?} (init={initial}, target={target})"
        );

        prod_list.write64(mem.addr(0), release_val);
        assert_eq!(prod_list.close(), S_OK);
        let pt = producer.execute(&[&prod_list]);
        assert!(producer.wait_done(pt, 2000), "producer timed out for {func:?}");
        assert!(consumer.wait_done(ct, 2000), "consumer timed out for {func:?}");
        assert_eq!(mem.read64(8), marker, "post-wait write missing for {func:?}");
    }
}

#[test]
fn gpu_copy_queue_mid_list_wait_ordering() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(device) = create_hardware_device() else {
        return;
    };
    let mem = GuestMem::new(65536);
    // Layout: [0]=flag, [8]=stage1_done, [16]=stage2_done,
    // [256]=shared_src, [512]=pre_dst, [768]=post_dst, [1024]=updated_src
    mem.write64(0, 0);
    mem.write64(8, 0);
    mem.write64(16, 0);
    mem.write64(256, 0x1111_2222_3333_4444);
    mem.write64(512, 0);
    mem.write64(768, 0);
    mem.write64(1024, 0xAAAA_BBBB_CCCC_DDDD);

    let mut producer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let mut consumer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_COPY);
    let prod_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let cons_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COPY);

    let splits_before = MID_LIST_SPLITS.load(Ordering::Relaxed);

    // Single COPY list: Copy1 -> Signal stage1 -> MID-LIST Wait64 -> Copy2 -> Signal stage2
    cons_list.copy_memory(mem.addr(512), mem.addr(256), 64);
    cons_list.write64(mem.addr(8), 1);
    cons_list.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_EQUAL.0 as u32, 0x2222);
    cons_list.copy_memory(mem.addr(768), mem.addr(256), 64);
    cons_list.write64(mem.addr(16), 1);
    assert_eq!(cons_list.close(), S_OK);
    assert!(
        MID_LIST_SPLITS.load(Ordering::Relaxed) > splits_before,
        "COPY mid-list wait did not split"
    );

    let ct = consumer.execute(&[&cons_list]);

    // Pre-wait segment MUST complete before producer signals!
    let start = Instant::now();
    while mem.read64(8) != 1 {
        assert!(start.elapsed() < Duration::from_secs(2), "pre-wait COPY segment stalled");
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(mem.read64(512), 0x1111_2222_3333_4444);

    // Negative control: post-wait segment MUST remain held until producer writes flag.
    thread::sleep(Duration::from_millis(40));
    assert_eq!(mem.read64(16), 0, "COPY mid-list wait released prematurely");
    assert_eq!(mem.read64(768), 0, "post-wait COPY ran before producer signal");

    // Producer updates shared_src and releases consumer's mid-list wait.
    prod_list.copy_memory(mem.addr(256), mem.addr(1024), 64);
    prod_list.write64(mem.addr(0), 0x2222);
    assert_eq!(prod_list.close(), S_OK);
    let pt = producer.execute(&[&prod_list]);

    assert!(producer.wait_done(pt, 2000), "producer timed out");
    assert!(consumer.wait_done(ct, 2000), "COPY consumer timed out");
    assert_eq!(mem.read64(16), 1);
    assert_eq!(
        mem.read64(768),
        0xAAAA_BBBB_CCCC_DDDD,
        "post-wait COPY did not observe producer's updated buffer"
    );
    assert_eq!(mem.read64(512), 0x1111_2222_3333_4444);
}

#[test]
fn gpu_compute_and_direct_mid_list_wait_replays_bound_state() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(device) = create_hardware_device() else {
        return;
    };
    let (root, pso) = create_compute_pipeline(&device);

    for queue_kind in [D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_DIRECT] {
        let mem = GuestMem::new(65536);
        // [0]=flag, [8]=stage1_done, [16]=stage2_done
        // [256]=inp_buf (u32), [512]=out1_buf (u32), [768]=out2_buf (u32), [1024]=new_inp (u32)
        mem.write64(0, 0);
        mem.write64(8, 0);
        mem.write64(16, 0);
        mem.write32(256, 3);
        mem.write32(512, 0);
        mem.write32(768, 0);
        mem.write32(1024, 7);

        let mut producer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
        let mut consumer = GuestQueue::new(&device, queue_kind);
        let prod_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
        let cons_list = GuestList::new(&device, queue_kind);

        // Bind state ONCE before mid-list wait; do NOT rebind root/PSO/constants/SRV after wait!
        if queue_kind == D3D12_COMMAND_LIST_TYPE_DIRECT {
            cons_list.set_direct_state_probe();
        }
        cons_list.set_compute_root(&root);
        cons_list.set_pipeline(&pso);
        cons_list.set_compute_const(0, 10, 0); // mul = 10
        cons_list.set_compute_srv(1, mem.addr(256));
        cons_list.set_compute_uav(2, mem.addr(512));
        cons_list.dispatch(1, 1, 1); // out1 = 3 * 10 + 1 = 31
        cons_list.write64(mem.addr(8), 1);

        // Mid-list wait splits segment 0 -> segment 1 and replays root/PSO/constants/SRV/UAV.
        cons_list.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 99);
        // Only change UAV to out2; root sig, PSO, constant (mul=10), and SRV (inp_buf) rely on replay!
        cons_list.set_compute_uav(2, mem.addr(768));
        cons_list.dispatch(1, 1, 1); // out2 = 7 * 10 + 1 = 71
        cons_list.write64(mem.addr(16), 1);
        assert_eq!(cons_list.close(), S_OK);

        let ct = consumer.execute(&[&cons_list]);

        let start = Instant::now();
        while mem.read64(8) != 1 {
            assert!(start.elapsed() < Duration::from_secs(2), "pre-wait dispatch stalled ({queue_kind:?})");
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(mem.read32(512), 31, "pre-wait dispatch result mismatch ({queue_kind:?})");

        thread::sleep(Duration::from_millis(40));
        assert_eq!(mem.read64(16), 0, "mid-list wait released early ({queue_kind:?})");
        assert_eq!(mem.read32(768), 0, "post-wait dispatch ran early ({queue_kind:?})");

        // Producer updates inp_buf from 3 -> 7, then signals flag = 99.
        prod_list.copy_memory(mem.addr(256), mem.addr(1024), 4);
        prod_list.write64(mem.addr(0), 99);
        assert_eq!(prod_list.close(), S_OK);
        let pt = producer.execute(&[&prod_list]);

        assert!(producer.wait_done(pt, 2000), "producer timed out ({queue_kind:?})");
        assert!(consumer.wait_done(ct, 2000), "consumer timed out ({queue_kind:?})");
        assert_eq!(mem.read64(16), 1);
        assert_eq!(
            mem.read32(768),
            71,
            "post-wait compute dispatch failed to replay state or read updated input ({queue_kind:?})"
        );
    }
}

#[test]
fn gpu_negative_control_and_in_batch_write_prevents_unsafe_elision() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(device) = create_hardware_device() else {
        return;
    };
    let mem = GuestMem::new(65536);
    let mut producer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let mut consumer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let list0 = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let list1 = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let prod_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);

    // Part A: Strict EQUAL negative control — wrong values must NOT release wait.
    mem.write64(0, 0);
    mem.write64(8, 0);
    list0.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_EQUAL.0 as u32, 0x5555);
    list0.write64(mem.addr(8), 0xBEEF);
    assert_eq!(list0.close(), S_OK);
    let ct = consumer.execute(&[&list0]);

    for wrong in [0x5554u64, 0x5556u64, u64::MAX] {
        mem.write64(0, wrong);
        notify_waiters();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(mem.read64(8), 0, "EQUAL wait falsely released on {wrong:#x}");
    }
    let release_start = Instant::now();
    prod_list.write64(mem.addr(0), 0x5555);
    assert_eq!(prod_list.close(), S_OK);
    let pt = producer.execute(&[&prod_list]);
    assert!(producer.wait_done(pt, 2000));
    assert!(consumer.wait_done(ct, 2000));
    assert_eq!(mem.read64(8), 0xBEEF);
    assert!(
        release_start.elapsed() < Duration::from_millis(250),
        "wakeup after exact match was too slow: {:?}",
        release_start.elapsed()
    );

    // Part B: In-batch prior write resets flag from 5 -> 0 before list1's wait >= 5.
    // Even though flag == 5 at submit time, list1 MUST NOT be elided!
    list0.reopen();
    prod_list.reopen();
    mem.write64(0, 5);
    mem.write64(8, 0);
    mem.write64(16, 0);

    list0.write64(mem.addr(0), 0); // resets flag to 0
    list0.write64(mem.addr(8), 1);
    assert_eq!(list0.close(), S_OK);

    list1.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 5);
    list1.write64(mem.addr(16), 1);
    assert_eq!(list1.close(), S_OK);

    let batch_ticket = consumer.execute(&[&list0, &list1]);
    let start = Instant::now();
    while mem.read64(8) != 1 {
        assert!(start.elapsed() < Duration::from_secs(2), "list0 did not finish");
        thread::sleep(Duration::from_millis(1));
    }
    thread::sleep(Duration::from_millis(40));
    assert_eq!(
        mem.read64(16),
        0,
        "list1 wait was unsafely elided despite prior in-batch write resetting flag"
    );

    prod_list.write64(mem.addr(0), 5);
    assert_eq!(prod_list.close(), S_OK);
    let pt = producer.execute(&[&prod_list]);
    assert!(producer.wait_done(pt, 2000));
    assert!(consumer.wait_done(batch_ticket, 2000));
    assert_eq!(mem.read64(16), 1);
}

#[test]
fn gpu_persistent_fence_and_1000_presatisfied_wait_elision() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(device) = create_hardware_device() else {
        return;
    };
    let mem = GuestMem::new(65536);

    // 1. 1,000 pre-satisfied monotone waits on a fresh queue: 0 fences created, 1,000 elided!
    let mut fast_q = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let fast_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    mem.write64(0, 1000);
    fast_list.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, 1000);
    assert_eq!(fast_list.close(), S_OK);

    let fences_before = FENCES_CREATED.load(Ordering::Relaxed);
    let elided_before = WAITS_ELIDED.load(Ordering::Relaxed);
    let t0 = Instant::now();
    let mut last_ticket = 0;
    for _ in 0..1000 {
        last_ticket = fast_q.execute(&[&fast_list]);
    }
    assert!(fast_q.wait_done(last_ticket, 5000));
    let elided_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let fences_delta_elided = FENCES_CREATED.load(Ordering::Relaxed) - fences_before;
    let elided_delta = WAITS_ELIDED.load(Ordering::Relaxed) - elided_before;
    assert_eq!(fences_delta_elided, 0, "pre-satisfied waits allocated a fence");
    assert_eq!(elided_delta, 1000, "expected all 1,000 pre-satisfied waits elided");

    // 2. 100 blocking cross-queue waits on one queue: strictly ONE persistent fence allocated!
    let mut producer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let mut consumer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    let prod_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let cons_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    mem.write64(64, 0);
    mem.write64(72, 0);

    let fences_before_blocking = FENCES_CREATED.load(Ordering::Relaxed);
    let submitted_before = WAITS_SUBMITTED.load(Ordering::Relaxed);
    let mut latencies_us = Vec::with_capacity(100);
    for step in 1..=100u64 {
        if step > 1 {
            cons_list.reopen();
            prod_list.reopen();
        }
        cons_list.wait64(mem.addr(64), D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, step);
        cons_list.write64(mem.addr(72), step);
        assert_eq!(cons_list.close(), S_OK);
        prod_list.write64(mem.addr(64), step);
        assert_eq!(prod_list.close(), S_OK);

        let iter_start = Instant::now();
        let ct = consumer.execute(&[&cons_list]);
        let pt = producer.execute(&[&prod_list]);
        assert!(producer.wait_done(pt, 2000));
        assert!(consumer.wait_done(ct, 2000));
        latencies_us.push(iter_start.elapsed().as_secs_f64() * 1e6);
        assert_eq!(mem.read64(72), step);
    }
    let fences_delta_blocking = FENCES_CREATED.load(Ordering::Relaxed) - fences_before_blocking;
    let submitted_delta = WAITS_SUBMITTED.load(Ordering::Relaxed) - submitted_before;
    assert_eq!(
        fences_delta_blocking, 1,
        "100 blocking waits on one queue must allocate exactly 1 persistent fence"
    );
    assert_eq!(submitted_delta, 100);
    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let median_us = latencies_us[latencies_us.len() / 2];
    eprintln!(
        "gpu_memory_wait_bench: 1000_presatisfied={elided_ms:.2}ms fences_elided={fences_delta_elided} 100_blocking_fences={fences_delta_blocking} handoff_median_us={median_us:.1}"
    );
}

#[test]
fn gpu_multiple_mid_list_splits_back_to_back_and_reset_reuse() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(device) = create_hardware_device() else {
        return;
    };
    let mem = GuestMem::new(65536);
    let mut producer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let mut consumer = GuestQueue::new(&device, D3D12_COMMAND_LIST_TYPE_COPY);
    let prod_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    let cons_list = GuestList::new(&device, D3D12_COMMAND_LIST_TYPE_COPY);

    // Repeat across 3 frames reusing the exact same allocator and split command list.
    for frame in 0..3u64 {
        if frame > 0 {
            cons_list.reopen();
            prod_list.reopen();
        }
        let base = frame * 100;
        // [0]=flag1, [8]=flag2, [16]=flag3, [24]=stage1, [32]=stage2, [40]=stage3
        // [256]=src, [512]=dst1, [768]=dst2, [1024]=val1, [1280]=val2
        mem.write64(0, 0);
        mem.write64(8, 0);
        mem.write64(16, 0);
        mem.write64(24, 0);
        mem.write64(32, 0);
        mem.write64(40, 0);
        mem.write64(256, base + 1);
        mem.write64(512, 0);
        mem.write64(768, 0);
        mem.write64(1024, base + 2);
        mem.write64(1280, base + 3);

        // Single list with 3 waits: Work0 -> Wait1 -> Wait2 (back-to-back) -> Work1 -> Wait3 -> Work2
        cons_list.copy_memory(mem.addr(512), mem.addr(256), 8);
        cons_list.write64(mem.addr(24), 1);
        cons_list.wait64(mem.addr(0), D3D12_COMPARISON_FUNC_EQUAL.0 as u32, base + 10);
        cons_list.wait64(mem.addr(8), D3D12_COMPARISON_FUNC_GREATER_EQUAL.0 as u32, base + 20);
        cons_list.copy_memory(mem.addr(768), mem.addr(256), 8);
        cons_list.write64(mem.addr(32), 1);
        cons_list.wait64(mem.addr(16), D3D12_COMPARISON_FUNC_NOT_EQUAL.0 as u32, 0);
        cons_list.copy_memory(mem.addr(512), mem.addr(256), 8);
        cons_list.write64(mem.addr(40), 1);
        assert_eq!(cons_list.close(), S_OK);

        let ct = consumer.execute(&[&cons_list]);

        // Stage 1 completes immediately; stages 2 and 3 remain held.
        let t0 = Instant::now();
        while mem.read64(24) != 1 {
            assert!(t0.elapsed() < Duration::from_secs(2), "frame {frame}: stage1 stalled");
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(mem.read64(512), base + 1);
        assert_eq!(mem.read64(32), 0);

        // Satisfy only flag1 — back-to-back flag2 must STILL hold stage 2!
        prod_list.write64(mem.addr(0), base + 10);
        assert_eq!(prod_list.close(), S_OK);
        let pt1 = producer.execute(&[&prod_list]);
        assert!(producer.wait_done(pt1, 2000));
        thread::sleep(Duration::from_millis(20));
        assert_eq!(mem.read64(32), 0, "frame {frame}: back-to-back wait2 bypassed");

        // Update src -> base+2 and satisfy flag2 -> stage 2 completes, stage 3 still held!
        prod_list.reopen();
        prod_list.copy_memory(mem.addr(256), mem.addr(1024), 8);
        prod_list.write64(mem.addr(8), base + 20);
        assert_eq!(prod_list.close(), S_OK);
        let pt2 = producer.execute(&[&prod_list]);
        assert!(producer.wait_done(pt2, 2000));
        let t1 = Instant::now();
        while mem.read64(32) != 1 {
            assert!(t1.elapsed() < Duration::from_secs(2), "frame {frame}: stage2 stalled");
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(mem.read64(768), base + 2);
        assert_eq!(mem.read64(40), 0);

        // Update src -> base+3 and satisfy flag3 -> stage 3 completes!
        prod_list.reopen();
        prod_list.copy_memory(mem.addr(256), mem.addr(1280), 8);
        prod_list.write64(mem.addr(16), 1);
        assert_eq!(prod_list.close(), S_OK);
        let pt3 = producer.execute(&[&prod_list]);
        assert!(producer.wait_done(pt3, 2000));
        assert!(consumer.wait_done(ct, 2000), "frame {frame}: consumer did not finish");
        assert_eq!(mem.read64(40), 1);
        assert_eq!(mem.read64(512), base + 3);
    }
}

#[path = "memory_wait_regressions.rs"]
mod regressions;
