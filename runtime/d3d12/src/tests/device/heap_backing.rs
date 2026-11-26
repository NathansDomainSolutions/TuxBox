use super::*;
use super::super::PagePools;
use crate::commands::page_mapping::GPU_APERTURE;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use windows::Win32::Foundation::{
    ERROR_NOT_FOUND, ERROR_NOT_SUPPORTED, SetLastError, WAIT_OBJECT_0,
};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

const PAGE: u64 = 65536;

// The fake XMem below is process-wide state.
static SERIAL: Mutex<()> = Mutex::new(());
// (guest page, backing) pairs the fake resolves; anything else is not XMem's.
static MAP: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static FAIL_WITH: AtomicU64 = AtomicU64::new(0);
static ACQUIRED: AtomicU64 = AtomicU64::new(0);
static RELEASED: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn fake_acquire(address: u64, bytes: u64, backing: *mut u64, hold: *mut u64) -> i32 {
    let fail = FAIL_WITH.load(Ordering::Acquire);
    if fail != 0 {
        unsafe { SetLastError(windows::Win32::Foundation::WIN32_ERROR(fail as u32)) };
        return 0;
    }
    let map = MAP.lock().unwrap();
    let Some(&(_, start)) = map.iter().find(|(guest, _)| *guest == address) else {
        unsafe { SetLastError(ERROR_NOT_FOUND) };
        return 0;
    };
    assert_eq!(bytes % PAGE, 0);
    let cookie = ACQUIRED.fetch_add(1, Ordering::AcqRel) + 1;
    unsafe {
        backing.write(start);
        hold.write(cookie);
    }
    1
}

unsafe extern "system" fn fake_release(_hold: u64) {
    RELEASED.fetch_add(1, Ordering::AcqRel);
}

struct FakeXmem(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl FakeXmem {
    fn install(map: &[(u64, u64)]) -> Self {
        let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        *MAP.lock().unwrap() = map.to_vec();
        FAIL_WITH.store(0, Ordering::Release);
        ACQUIRED.store(0, Ordering::Release);
        RELEASED.store(0, Ordering::Release);
        *TEST_EXPORTS.lock().unwrap() = Some(Exports::new(fake_acquire, fake_release));
        Self(guard)
    }
}

impl Drop for FakeXmem {
    fn drop(&mut self) {
        *TEST_EXPORTS.lock().unwrap() = None;
        MAP.lock().unwrap().clear();
    }
}

#[test]
fn without_backing_exports_the_address_is_imported() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // The test process has no xmem.dll.
    let backing = acquire(0x7_0000_0000, PAGE).unwrap();
    assert_eq!(backing.address, 0x7_0000_0000);
    assert!(backing.hold.is_none());
}

#[test]
fn memory_outside_xmem_is_imported_at_its_address() {
    let _xmem = FakeXmem::install(&[]);
    let backing = acquire(0x7_0001_0000, PAGE).unwrap();
    assert_eq!(backing.address, 0x7_0001_0000);
    assert!(backing.hold.is_none());
    assert_eq!(ACQUIRED.load(Ordering::Acquire), 0);
}

#[test]
fn an_unusable_backing_fails_the_import() {
    let _xmem = FakeXmem::install(&[]);
    FAIL_WITH.store(ERROR_NOT_SUPPORTED.0 as u64, Ordering::Release);
    let error = acquire(0x7_0002_0000, 2 * PAGE).err().expect("scattered backing must fail");
    assert_eq!(error.code(), windows::core::HRESULT::from_win32(ERROR_NOT_SUPPORTED.0));
}

#[test]
fn a_backing_is_held_until_dropped_and_released_once() {
    let _xmem = FakeXmem::install(&[(0x7_0003_0000, 0x7_0100_0000)]);
    let backing = acquire(0x7_0003_0000, PAGE).unwrap();
    assert_eq!(backing.address, 0x7_0100_0000);
    assert!(backing.hold.is_some());
    assert_eq!((ACQUIRED.load(Ordering::Acquire), RELEASED.load(Ordering::Acquire)), (1, 0));
    drop(backing);
    assert_eq!(RELEASED.load(Ordering::Acquire), 1);
}

struct Region(*mut core::ffi::c_void);

impl Region {
    fn private(bytes: usize) -> Self {
        let p = unsafe { VirtualAlloc(None, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!p.is_null());
        Self(p)
    }

    // A GPU-only alias: an address with no CPU memory behind it.
    fn reserved(bytes: usize) -> Self {
        let p = unsafe { VirtualAlloc(None, bytes, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!p.is_null());
        Self(p)
    }

    fn address(&self) -> u64 {
        self.0 as u64
    }

    fn words(&self, count: usize) -> &mut [u32] {
        unsafe { core::slice::from_raw_parts_mut(self.0.cast(), count) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe { VirtualFree(self.0, 0, MEM_RELEASE).unwrap() };
    }
}

fn hardware_device() -> Option<ID3D12Device13> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
    let mut index = 0;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(index) } {
        index += 1;
        if unsafe { adapter.GetDesc1() }.is_ok_and(|desc| desc.Flags & 2 != 0) {
            continue;
        }
        let mut device: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_ok()
            && let Some(device) = device.and_then(|d| d.cast::<ID3D12Device13>().ok())
        {
            return Some(device);
        }
    }
    None
}

fn buffer_desc(width: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
        ..Default::default()
    }
}

fn placed(device: &ID3D12Device13, heap: &ID3D12Heap) -> ID3D12Resource {
    let mut resource = None;
    unsafe {
        device.CreatePlacedResource(heap, 0, &buffer_desc(PAGE), D3D12_RESOURCE_STATE_COMMON, None, &mut resource)
    }
    .unwrap();
    resource.unwrap()
}

fn committed(device: &ID3D12Device13, kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_NONE, ..buffer_desc(PAGE) };
    let mut resource = None;
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut resource) }
        .unwrap();
    resource.unwrap()
}

/// Run `record` on a direct queue and wait for it.
fn submit(device: &ID3D12Device13, record: impl FnOnce(&ID3D12GraphicsCommandList)) {
    let desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
    let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&desc) }.unwrap();
    let allocator: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap();
    let list: ID3D12GraphicsCommandList =
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None::<&ID3D12PipelineState>) }.unwrap();
    record(&list);
    unsafe { list.Close() }.unwrap();
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    let event = unsafe { CreateEventW(None, false, false, None) }.unwrap();
    unsafe {
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&fence, 1).unwrap();
        fence.SetEventOnCompletion(1, event).unwrap();
        assert_eq!(WaitForSingleObject(event, 10_000), WAIT_OBJECT_0, "GPU copy did not finish");
        let _ = windows::Win32::Foundation::CloseHandle(event);
    }
}

// Two GPU aliases of one private page, as XMem would hand them out: imports of
// either address must reach the same memory, and the CPU sees GPU writes.
#[test]
fn aliases_import_their_private_backing() {
    let Some(device) = hardware_device() else { return };
    let home = Region::private(PAGE as usize);
    let (first, second) = (Region::reserved(PAGE as usize), Region::reserved(PAGE as usize));
    let xmem = FakeXmem::install(&[(first.address(), home.address()), (second.address(), home.address())]);
    for (i, word) in home.words(16).iter_mut().enumerate() {
        *word = 0x5ec7_0000 + i as u32;
    }

    let pools = PagePools::default();
    let import = |guest: &Region| {
        let backing = acquire(guest.address(), PAGE).unwrap();
        assert_eq!(backing.address, home.address());
        let pin = GPU_APERTURE.pin(guest.address(), PAGE).unwrap();
        pools.import(&device, guest.address(), backing, PAGE, pin, "test", false).unwrap()
    };
    let heap_first = import(&first);
    // The same guest page with the same backing reuses its live import.
    assert!(Arc::ptr_eq(&heap_first, &import(&first)));
    let heap_second = import(&second);
    assert!(!Arc::ptr_eq(&heap_first, &heap_second));

    let via_first = placed(&device, &heap_first.native);
    let via_second = placed(&device, &heap_second.native);
    let upload = committed(&device, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    let readback = committed(&device, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut mapped = core::ptr::null_mut();
    unsafe { upload.Map(0, None, Some(&mut mapped)) }.unwrap();
    let gpu_words: [u32; 4] = [0x00c0_ffee, 0x0bad_f00d, 0x7e57_c0de, 0x0ddb_a115];
    unsafe { core::ptr::copy_nonoverlapping(gpu_words.as_ptr(), mapped.cast::<u32>().add(8), 4) };

    // The GPU reads the CPU's bytes through the first alias, writes through
    // it, and reads that write back through the second alias.
    submit(&device, |list| unsafe {
        list.CopyBufferRegion(&readback, 0, &via_first, 0, 32);
        list.CopyBufferRegion(&via_first, 32, &upload, 32, 16);
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            ..Default::default()
        };
        list.ResourceBarrier(&[barrier]);
        list.CopyBufferRegion(&readback, 32, &via_second, 32, 16);
    });
    let mut out = core::ptr::null_mut();
    unsafe { readback.Map(0, None, Some(&mut out)) }.unwrap();
    let seen = unsafe { core::slice::from_raw_parts(out.cast::<u32>(), 12) };
    let expected: Vec<u32> = (0..8).map(|i| 0x5ec7_0000 + i).collect();
    assert_eq!(&seen[..8], &expected[..], "GPU did not read the backing through an alias");
    assert_eq!(&seen[8..12], &gpu_words, "a GPU write through one alias is not visible through the other");
    assert_eq!(&home.words(12)[8..12], &gpu_words, "the CPU does not see the GPU write");

    // Holds outlive every native heap and are released once each.
    assert_eq!(ACQUIRED.load(Ordering::Acquire), 3);
    assert_eq!(RELEASED.load(Ordering::Acquire), 1, "only the reused import's extra hold is gone");
    drop((via_first, via_second));
    drop(heap_first);
    assert_eq!(RELEASED.load(Ordering::Acquire), 2);
    drop(heap_second);
    assert_eq!(RELEASED.load(Ordering::Acquire), 3);
    drop(xmem);
}

// A guest page whose backing moved must not reuse the import of the old one.
#[test]
fn a_moved_backing_is_imported_again() {
    let Some(device) = hardware_device() else { return };
    let (old, new) = (Region::private(PAGE as usize), Region::private(PAGE as usize));
    let guest = Region::reserved(PAGE as usize);
    let xmem = FakeXmem::install(&[(guest.address(), old.address())]);
    let pools = PagePools::default();
    let import = || {
        let backing = acquire(guest.address(), PAGE).unwrap();
        let pin = GPU_APERTURE.pin(guest.address(), PAGE).unwrap();
        pools.import(&device, guest.address(), backing, PAGE, pin, "test", false).unwrap()
    };
    let before = import();
    *MAP.lock().unwrap() = vec![(guest.address(), new.address())];
    let after = import();
    assert!(!Arc::ptr_eq(&before, &after));
    drop((before, after));
    assert_eq!(RELEASED.load(Ordering::Acquire), ACQUIRED.load(Ordering::Acquire));
    drop(xmem);
}
