#[path = "allocation_regressions.rs"]
mod regressions;

use super::{
    copy_footprint_desc, initial_barrier_layout, packed_uav_formats, packed_uav_raw_format,
    supports_typed_uav,
};
use core::ffi::c_void;
use core::ptr;
use windows::Win32::Foundation::{CloseHandle, E_INVALIDARG, HANDLE};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{GUID, HRESULT, IUnknown, Interface, PCSTR, s};

#[test]
fn packed_uav_formats_and_raw_views_cover_rgb9e5_and_bgra5551() {
    let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    assert_eq!(packed_uav_raw_format(DXGI_FORMAT_R9G9B9E5_SHAREDEXP), Some(DXGI_FORMAT_R32_UINT));
    assert_eq!(packed_uav_raw_format(DXGI_FORMAT_B5G5R5A1_UNORM), Some(DXGI_FORMAT_R16_UINT));
    assert_eq!(packed_uav_raw_format(DXGI_FORMAT_R8G8B8A8_UNORM), None);
    assert_eq!(packed_uav_raw_format(DXGI_FORMAT_R32_UINT), None);

    assert_eq!(
        packed_uav_formats(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, uav),
        &[DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_UINT]
    );
    assert_eq!(
        packed_uav_formats(DXGI_FORMAT_B5G5R5A1_UNORM, uav),
        &[DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_R16_UINT]
    );
    assert!(packed_uav_formats(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_NONE).is_empty());
    assert!(packed_uav_formats(DXGI_FORMAT_B5G5R5A1_UNORM, D3D12_RESOURCE_FLAG_NONE).is_empty());
    assert!(packed_uav_formats(DXGI_FORMAT_R16G16B16A16_FLOAT, uav).is_empty());
}

#[test]
fn copy_footprint_desc_strips_uav_and_console_flags_for_packed_uav() {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1920,
        Height: 1080,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAGS(0x00c0_0004),
        ..Default::default()
    };
    let cleaned = copy_footprint_desc(&desc).expect("packed UAV footprint desc");
    assert_eq!(cleaned.Format, DXGI_FORMAT_R9G9B9E5_SHAREDEXP);
    assert_eq!(cleaned.Flags, D3D12_RESOURCE_FLAG_NONE);

    let regular = D3D12_RESOURCE_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..desc
    };
    assert!(copy_footprint_desc(&regular).is_none());
}

#[test]
fn initial_barrier_layout_maps_supported_legacy_states_and_rejects_unknown() {
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_COMMON),
        Some(D3D12_BARRIER_LAYOUT_COMMON)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        Some(D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        Some(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
        Some(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE),
        Some(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE)
    );
    assert_eq!(
        initial_barrier_layout(
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE
        ),
        Some(D3D12_BARRIER_LAYOUT_GENERIC_READ)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_COPY_DEST),
        Some(D3D12_BARRIER_LAYOUT_COPY_DEST)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_COPY_SOURCE),
        Some(D3D12_BARRIER_LAYOUT_COPY_SOURCE)
    );
    assert_eq!(
        initial_barrier_layout(D3D12_RESOURCE_STATE_GENERIC_READ),
        Some(D3D12_BARRIER_LAYOUT_GENERIC_READ)
    );
    assert_eq!(
        initial_barrier_layout(crate::device::host_state(D3D12_RESOURCE_STATE_GENERIC_READ, true)),
        Some(D3D12_BARRIER_LAYOUT_GENERIC_READ)
    );
    assert_eq!(initial_barrier_layout(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT), None);
    assert_eq!(initial_barrier_layout(D3D12_RESOURCE_STATE_DEPTH_WRITE), None);
}

#[repr(C, align(8))]
struct ExtendedResourceDesc([u8; 64]);

impl ExtendedResourceDesc {
    fn new(desc: D3D12_RESOURCE_DESC, dcc: u32, extension: u64) -> Self {
        let mut raw = [0u8; 64];
        unsafe {
            ptr::copy_nonoverlapping(
                ptr::from_ref(&desc).cast::<u8>(),
                raw.as_mut_ptr(),
                size_of::<D3D12_RESOURCE_DESC>(),
            );
            raw.as_mut_ptr().add(52).cast::<u32>().write_unaligned(dcc);
            raw.as_mut_ptr().add(56).cast::<u64>().write_unaligned(extension);
        }
        Self(raw)
    }

    fn as_desc_ptr(&self) -> *const D3D12_RESOURCE_DESC {
        self.0.as_ptr().cast()
    }
}

unsafe fn vtable_slot<F: Copy>(object: *mut c_void, slot: usize) -> F {
    let table = unsafe { *object.cast::<*const *const c_void>() };
    let func = unsafe { *table.add(slot) };
    unsafe { core::mem::transmute_copy(&func) }
}

fn compile_hlsl(source: &str, entry: PCSTR, target: PCSTR) -> Vec<u8> {
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
    let module = unsafe { LoadLibraryA(s!("d3dcompiler_47.dll")) }.expect("load d3dcompiler_47");
    let sym = unsafe { GetProcAddress(module, s!("D3DCompile")) }.expect("D3DCompile symbol");
    let compile: D3DCompileFn = unsafe { core::mem::transmute(sym) };
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let hr = unsafe {
        compile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            ptr::null(),
            ptr::null_mut(),
            entry,
            target,
            0,
            0,
            &mut code,
            &mut errors,
        )
    };
    if hr.is_err() {
        let msg = errors
            .map(|b| unsafe {
                core::str::from_utf8_unchecked(core::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .to_owned()
            })
            .unwrap_or_default();
        panic!("D3DCompile failed {hr:?}: {msg}");
    }
    let blob = code.expect("compiled shader blob");
    unsafe {
        core::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    }
}

fn assert_no_debug_errors(native_device: &ID3D12Device, stage: &str) {
    let Ok(iq) = native_device.cast::<ID3D12InfoQueue>() else {
        return;
    };
    let count = unsafe { iq.GetNumStoredMessages() };
    let mut failures = Vec::new();
    for i in 0..count {
        let mut len = 0usize;
        let _ = unsafe { iq.GetMessage(i, None, &mut len) };
        if len == 0 {
            continue;
        }
        let mut buf = vec![0u8; len];
        let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
        if unsafe { iq.GetMessage(i, Some(msg), &mut len) }.is_ok() {
            let sev = unsafe { (*msg).Severity };
            if sev == D3D12_MESSAGE_SEVERITY_CORRUPTION
                || sev == D3D12_MESSAGE_SEVERITY_ERROR
                || sev == D3D12_MESSAGE_SEVERITY_WARNING
            {
                let text = unsafe { core::ffi::CStr::from_ptr((*msg).pDescription.cast()) }
                    .to_string_lossy()
                    .into_owned();
                failures.push(format!("[sev={} id={}] {text}", sev.0, unsafe { (*msg).ID.0 }));
            }
        }
    }
    unsafe { iq.ClearStoredMessages() };
    assert!(failures.is_empty(), "D3D12 debug messages at {stage}: {failures:#?}");
}

fn find_non_typed_rgb9e5_adapter() -> Option<(IDXGIAdapter1, ID3D12Device)> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
    let mut index = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(index) } {
        index += 1;
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        if (desc.Flags & 2) != 0 {
            continue; // Skip software adapter when looking for hardware.
        }
        let mut native: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut native) }.is_ok()
            && let Some(dev) = native
            && !supports_typed_uav(&dev, DXGI_FORMAT_R9G9B9E5_SHAREDEXP)
        {
            return Some((adapter, dev));
        }
    }
    None
}

fn enable_desktop_debug_layer() {
    type GetDebugFn = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> HRESULT;
    if let Ok(module) = unsafe { LoadLibraryA(s!("d3d12.dll")) }
        && let Some(sym) = unsafe { GetProcAddress(module, s!("D3D12GetDebugInterface")) }
    {
        let get_debug: GetDebugFn = unsafe { core::mem::transmute(sym) };
        let mut raw = ptr::null_mut();
        if unsafe { get_debug(&ID3D12Debug::IID, &mut raw) }.is_ok() && !raw.is_null() {
            let dbg: ID3D12Debug = unsafe { Interface::from_raw(raw) };
            unsafe { dbg.EnableDebugLayer() };
        }
    }
}

#[test]
fn r9g9b9e5_uav_gpu_roundtrip_and_negative_control() {
    enable_desktop_debug_layer();

    let Some((adapter, raw_device)) = find_non_typed_rgb9e5_adapter() else {
        return; // No adapter lacking native R9G9B9E5 typed UAV available on this host.
    };

    let heap_default = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let tex_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 16,
        Height: 16,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };

    // 1. Negative control: raw desktop CreateCommittedResource MUST fail on this GPU.
    let mut raw_res: Option<ID3D12Resource> = None;
    let raw_err = unsafe {
        raw_device.CreateCommittedResource(
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            &tex_desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut raw_res,
        )
    }
    .expect_err("raw CreateCommittedResource(R9G9B9E5, UAV) must fail on this GPU");
    assert_eq!(raw_err.code(), E_INVALIDARG);
    if let Ok(iq) = raw_device.cast::<ID3D12InfoQueue>() {
        unsafe { iq.ClearStoredMessages() };
    }
    drop(raw_device);

    // 2. Create TuxBox proxy device on the exact same adapter.
    let mut proxy_ptr = ptr::null_mut();
    let hr = unsafe {
        crate::device::create(adapter.as_raw(), ID3D12Device::IID, &mut proxy_ptr)
    };
    assert!(hr.is_ok(), "proxy device::create failed: {hr:?}");
    let proxy_device = unsafe { IUnknown::from_raw(proxy_ptr) };
    let dev_native = unsafe { &(*proxy_ptr.cast::<super::Device>()).native }.clone();
    assert_no_debug_errors(&dev_native, "device create");

    // 3. Test slot 25 (GetResourceAllocationInfo), slot 108 (info2), slot 38 (footprints).
    type AllocInfoFn = unsafe extern "system" fn(
        *mut D3D12_RESOURCE_ALLOCATION_INFO,
        *mut c_void,
        u32,
        u32,
        *const D3D12_RESOURCE_DESC,
    ) -> *mut D3D12_RESOURCE_ALLOCATION_INFO;
    let mut info = D3D12_RESOURCE_ALLOCATION_INFO::default();
    unsafe {
        let f: AllocInfoFn = vtable_slot(proxy_ptr, 25);
        f(&mut info, proxy_ptr, 0, 1, &tex_desc);
    }
    assert_ne!(info.SizeInBytes, u64::MAX);
    assert!(info.SizeInBytes > 0 && info.Alignment > 0);

    type AllocInfo2Fn = unsafe extern "system" fn(
        *mut D3D12_RESOURCE_ALLOCATION_INFO,
        *mut c_void,
        u32,
        u32,
        *const D3D12_RESOURCE_DESC1,
        *mut D3D12_RESOURCE_ALLOCATION_INFO1,
    ) -> *mut D3D12_RESOURCE_ALLOCATION_INFO;
    let tex_desc1 = super::resource_desc1(&tex_desc);
    let mut info2 = D3D12_RESOURCE_ALLOCATION_INFO::default();
    unsafe {
        let f: AllocInfo2Fn = vtable_slot(proxy_ptr, 108);
        f(&mut info2, proxy_ptr, 0, 1, &tex_desc1, ptr::null_mut());
    }
    assert_eq!(info2.SizeInBytes, info.SizeInBytes);

    type FootprintsFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_RESOURCE_DESC,
        u32,
        u32,
        u64,
        *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        *mut u32,
        *mut u64,
        *mut u64,
    );
    let (mut fp, mut rows, mut row_bytes, mut total) =
        (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u32, 0u64, 0u64);
    unsafe {
        let f: FootprintsFn = vtable_slot(proxy_ptr, 38);
        f(proxy_ptr, &tex_desc, 0, 1, 0, &mut fp, &mut rows, &mut row_bytes, &mut total);
    }
    assert_eq!(rows, 16);
    assert_eq!(row_bytes, 64);
    assert_eq!(total, 15 * u64::from(fp.Footprint.RowPitch) + row_bytes);
    assert_no_debug_errors(&dev_native, "allocation info and footprints");

    // 4. Test slot 76 (CreateCommittedResourceX, 1920x1080 console layout 0x101 + DCC 3),
    //    slot 29 (CreatePlacedResource), slot 30 (CreateReservedResource), slot 127 (committed_layout).
    type CommittedFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_FLAGS,
        *const D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_STATES,
        *const D3D12_CLEAR_VALUE,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    type ResDescFn =
        unsafe extern "system" fn(*mut c_void, *mut D3D12_RESOURCE_DESC) -> *mut D3D12_RESOURCE_DESC;
    let ext_desc = ExtendedResourceDesc::new(
        D3D12_RESOURCE_DESC {
            Width: 1920,
            Height: 1080,
            Layout: D3D12_TEXTURE_LAYOUT(0x101),
            Flags: D3D12_RESOURCE_FLAGS(0x00c0_0004),
            ..tex_desc
        },
        3,
        0,
    );
    let mut slot76_ptr = ptr::null_mut();
    let hr = unsafe {
        let f: CommittedFn = vtable_slot(proxy_ptr, 76);
        f(
            proxy_ptr,
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            ext_desc.as_desc_ptr(),
            D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut slot76_ptr,
        )
    };
    assert!(hr.is_ok(), "slot 76 R9G9B9E5 UAV failed: {hr:?}");
    let _slot76 = unsafe { IUnknown::from_raw(slot76_ptr) };
    let mut got_desc = D3D12_RESOURCE_DESC::default();
    unsafe {
        let f: ResDescFn = vtable_slot(slot76_ptr, 10);
        f(slot76_ptr, &mut got_desc);
    }
    assert_eq!(got_desc.Format, DXGI_FORMAT_R9G9B9E5_SHAREDEXP);
    assert_eq!(got_desc.Layout, D3D12_TEXTURE_LAYOUT(0x101));
    assert_eq!(got_desc.Flags, D3D12_RESOURCE_FLAGS(0x00c0_0004));

    // Slot 29 (CreatePlacedResource) on default heap.
    type CreateHeapFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_HEAP_DESC,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    type PlaceFn = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u64,
        *const D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_STATES,
        *const D3D12_CLEAR_VALUE,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    let heap_desc = D3D12_HEAP_DESC {
        SizeInBytes: info.SizeInBytes.max(65536),
        Properties: heap_default,
        Alignment: info.Alignment.max(65536),
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    };
    let mut heap_ptr = ptr::null_mut();
    let hr = unsafe {
        let f: CreateHeapFn = vtable_slot(proxy_ptr, 28);
        f(proxy_ptr, &heap_desc, &ID3D12Heap::IID, &mut heap_ptr)
    };
    assert!(hr.is_ok(), "CreateHeap failed: {hr:?}");
    let _heap = unsafe { IUnknown::from_raw(heap_ptr) };
    let mut placed_ptr = ptr::null_mut();
    let hr = unsafe {
        let f: PlaceFn = vtable_slot(proxy_ptr, 29);
        f(
            proxy_ptr,
            heap_ptr,
            0,
            &tex_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut placed_ptr,
        )
    };
    assert!(hr.is_ok(), "slot 29 CreatePlacedResource(R9G9B9E5, UAV) failed: {hr:?}");
    let _placed = unsafe { IUnknown::from_raw(placed_ptr) };

    // Slot 30 (CreateReservedResource).
    type ReservedFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_STATES,
        *const D3D12_CLEAR_VALUE,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    let reserved_desc = D3D12_RESOURCE_DESC {
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        ..tex_desc
    };
    let mut reserved_ptr = ptr::null_mut();
    let hr = unsafe {
        let f: ReservedFn = vtable_slot(proxy_ptr, 30);
        f(
            proxy_ptr,
            &reserved_desc,
            D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut reserved_ptr,
        )
    };
    assert!(hr.is_ok(), "slot 30 CreateReservedResource(R9G9B9E5, UAV) failed: {hr:?}");
    let _reserved = unsafe { IUnknown::from_raw(reserved_ptr) };

    // Slot 127 (committed_layout) with count=0 and count=1 {R9G9B9E5}.
    type CommittedLayoutFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_FLAGS,
        *const D3D12_RESOURCE_DESC,
        D3D12_BARRIER_LAYOUT,
        *const D3D12_CLEAR_VALUE,
        *mut c_void,
        u32,
        *const DXGI_FORMAT,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    let only_rgb9e5 = [DXGI_FORMAT_R9G9B9E5_SHAREDEXP];
    for (count, list) in [(0u32, ptr::null()), (1u32, only_rgb9e5.as_ptr())] {
        let mut p = ptr::null_mut();
        let hr = unsafe {
            let f: CommittedLayoutFn = vtable_slot(proxy_ptr, 127);
            f(
                proxy_ptr,
                &heap_default,
                D3D12_HEAP_FLAG_NONE,
                &tex_desc,
                D3D12_BARRIER_LAYOUT_COMMON,
                ptr::null(),
                ptr::null_mut(),
                count,
                list,
                &ID3D12Resource::IID,
                &mut p,
            )
        };
        assert!(hr.is_ok(), "slot 127 committed_layout(count={count}) failed: {hr:?}");
        drop(unsafe { IUnknown::from_raw(p) });
    }
    assert_no_debug_errors(&dev_native, "resource creation slots");

    // 5. Full GPU compute write -> barrier -> SRV sample -> exact HDR float4 readback
    //    exercising BOTH initial=COMMON and initial=UNORDERED_ACCESS (enhanced barrier interop)!
    let write_hlsl = r#"
        RWTexture2D<uint> dst : register(u0);
        [numthreads(4, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {
            static const uint patterns[4] = {
                0xC0100401u,
                0x81010100u,
                0xD00007F4u,
                0x00000000u
            };
            dst[uint2(tid.x, 0)] = patterns[tid.x];
        }
    "#;
    let sample_hlsl = r#"
        Texture2D<float4> src : register(t0);
        SamplerState samp : register(s0);
        RWStructuredBuffer<float4> out_buf : register(u0);
        [numthreads(5, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {
            float2 uv = float2((float(tid.x) + 0.5) / 16.0, 0.5 / 16.0);
            out_buf[tid.x] = src.SampleLevel(samp, uv, 0.0);
        }
    "#;
    let write_dxbc = compile_hlsl(write_hlsl, s!("main"), s!("cs_5_0"));
    let sample_dxbc = compile_hlsl(sample_hlsl, s!("main"), s!("cs_5_0"));

    // Root signatures via proxy slot 16.
    type CreateRootSigFn = unsafe extern "system" fn(
        *mut c_void,
        u32,
        *const u8,
        usize,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    let make_root_sig = |desc: &D3D12_ROOT_SIGNATURE_DESC| -> IUnknown {
        let mut blob: Option<ID3DBlob> = None;
        unsafe {
            D3D12SerializeRootSignature(desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
        }
        .expect("serialize root signature");
        let blob = blob.unwrap();
        let mut out = ptr::null_mut();
        let hr = unsafe {
            let f: CreateRootSigFn = vtable_slot(proxy_ptr, 16);
            f(
                proxy_ptr,
                0,
                blob.GetBufferPointer().cast(),
                blob.GetBufferSize(),
                &ID3D12RootSignature::IID,
                &mut out,
            )
        };
        assert!(hr.is_ok(), "CreateRootSignature failed: {hr:?}");
        unsafe { IUnknown::from_raw(out) }
    };

    let write_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let write_param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: &write_range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let write_rs = make_root_sig(&D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 1,
        pParameters: &write_param,
        ..Default::default()
    });

    let sample_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 1,
        },
    ];
    let samp_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let sample_params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 2,
                    pDescriptorRanges: sample_ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &samp_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let sample_rs = make_root_sig(&D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 2,
        pParameters: sample_params.as_ptr(),
        ..Default::default()
    });

    // Compute pipelines via proxy slot 62.
    type CreateComputePsoFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_COMPUTE_PIPELINE_STATE_DESC,
        u32,
        *const c_void,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    let make_pso = |rs: &IUnknown, code: &[u8]| -> IUnknown {
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        };
        unsafe {
            ptr::from_mut(&mut desc.pRootSignature)
                .cast::<*mut c_void>()
                .write(rs.as_raw());
        }
        let mut out = ptr::null_mut();
        let hr = unsafe {
            let f: CreateComputePsoFn = vtable_slot(proxy_ptr, 62);
            f(proxy_ptr, &desc, 0, ptr::null(), &ID3D12PipelineState::IID, &mut out)
        };
        core::mem::forget(desc.pRootSignature);
        assert!(hr.is_ok(), "CreateComputePipelineStateX failed: {hr:?}");
        unsafe { IUnknown::from_raw(out) }
    };
    let write_pso = make_pso(&write_rs, &write_dxbc);
    let sample_pso = make_pso(&sample_rs, &sample_dxbc);

    // Queue (slot 8), allocator (slot 9), fence (slot 36).
    type CreateQueueFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_COMMAND_QUEUE_DESC,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    type CreateAllocFn = unsafe extern "system" fn(
        *mut c_void,
        D3D12_COMMAND_LIST_TYPE,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    type CreateFenceFn = unsafe extern "system" fn(
        *mut c_void,
        u64,
        D3D12_FENCE_FLAGS,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    #[repr(C)]
    struct ListDesc {
        kind: D3D12_COMMAND_LIST_TYPE,
        flags: u32,
        node_mask: u32,
        reserved: u32,
    }
    type CreateListXFn = unsafe extern "system" fn(
        *mut c_void,
        *const ListDesc,
        *mut c_void,
        *mut c_void,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    let qdesc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
    };
    let mut queue_ptr = ptr::null_mut();
    assert!(unsafe {
        let f: CreateQueueFn = vtable_slot(proxy_ptr, 8);
        f(proxy_ptr, &qdesc, &ID3D12CommandQueue::IID, &mut queue_ptr)
    }
    .is_ok());
    let _queue = unsafe { IUnknown::from_raw(queue_ptr) };

    let mut alloc_ptr = ptr::null_mut();
    assert!(unsafe {
        let f: CreateAllocFn = vtable_slot(proxy_ptr, 9);
        f(proxy_ptr, D3D12_COMMAND_LIST_TYPE_DIRECT, &ID3D12CommandAllocator::IID, &mut alloc_ptr)
    }
    .is_ok());
    let _alloc = unsafe { IUnknown::from_raw(alloc_ptr) };

    let mut fence_ptr = ptr::null_mut();
    assert!(unsafe {
        let f: CreateFenceFn = vtable_slot(proxy_ptr, 36);
        f(proxy_ptr, 0, D3D12_FENCE_FLAG_NONE, &ID3D12Fence::IID, &mut fence_ptr)
    }
    .is_ok());
    let _fence = unsafe { IUnknown::from_raw(fence_ptr) };

    // Descriptor heaps: shader-visible CBV/SRV/UAV (8), CPU-only CBV/SRV/UAV (8), sampler (1).
    type CreateDescHeapFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_DESCRIPTOR_HEAP_DESC,
        *const GUID,
        *mut *mut c_void,
    ) -> HRESULT;
    type IncSizeFn = unsafe extern "system" fn(*mut c_void, D3D12_DESCRIPTOR_HEAP_TYPE) -> u32;
    type CpuStartFn = unsafe extern "system" fn(*mut c_void) -> usize;
    type GpuStartFn = unsafe extern "system" fn(*mut c_void) -> u64;

    let make_dheap = |ty, num, flags| -> IUnknown {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: num, Flags: flags, NodeMask: 1 };
        let mut out = ptr::null_mut();
        assert!(unsafe {
            let f: CreateDescHeapFn = vtable_slot(proxy_ptr, 14);
            f(proxy_ptr, &desc, &ID3D12DescriptorHeap::IID, &mut out)
        }
        .is_ok());
        unsafe { IUnknown::from_raw(out) }
    };
    let gpu_heap = make_dheap(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        8,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    );
    let cpu_heap = make_dheap(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        8,
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    );
    let samp_heap = make_dheap(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        1,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    );
    let view_inc = unsafe {
        let f: IncSizeFn = vtable_slot(proxy_ptr, 15);
        f(proxy_ptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize
    };
    let gpu_cpu_base = unsafe { vtable_slot::<CpuStartFn>(gpu_heap.as_raw(), 9)(gpu_heap.as_raw()) };
    let gpu_gpu_base = unsafe { vtable_slot::<GpuStartFn>(gpu_heap.as_raw(), 10)(gpu_heap.as_raw()) };
    let cpu_cpu_base = unsafe { vtable_slot::<CpuStartFn>(cpu_heap.as_raw(), 9)(cpu_heap.as_raw()) };
    let samp_cpu_base = unsafe { vtable_slot::<CpuStartFn>(samp_heap.as_raw(), 9)(samp_heap.as_raw()) };
    let samp_gpu_base = unsafe { vtable_slot::<GpuStartFn>(samp_heap.as_raw(), 10)(samp_heap.as_raw()) };

    // Point sampler in samp_heap[0].
    type CreateSamplerFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_SAMPLER_DESC,
        D3D12_CPU_DESCRIPTOR_HANDLE,
    );
    let sampler_desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe {
        let f: CreateSamplerFn = vtable_slot(proxy_ptr, 22);
        f(proxy_ptr, &sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: samp_cpu_base });
    }

    // Output structured buffer (5 * float4 = 80 bytes) + readback buffer.
    let buf_bytes = 5 * 16u64;
    let out_buf_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: buf_bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut out_buf_ptr = ptr::null_mut();
    assert!(unsafe {
        let f: CommittedFn = vtable_slot(proxy_ptr, 27);
        f(
            proxy_ptr,
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            &out_buf_desc,
            D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut out_buf_ptr,
        )
    }
    .is_ok());
    let _out_buf = unsafe { IUnknown::from_raw(out_buf_ptr) };

    let heap_readback = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let rb_desc = D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAG_NONE, ..out_buf_desc };
    let mut rb_ptr = ptr::null_mut();
    assert!(unsafe {
        let f: CommittedFn = vtable_slot(proxy_ptr, 27);
        f(
            proxy_ptr,
            &heap_readback,
            D3D12_HEAP_FLAG_NONE,
            &rb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut rb_ptr,
        )
    }
    .is_ok());
    let _rb = unsafe { IUnknown::from_raw(rb_ptr) };

    type CreateSrvFn = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *const D3D12_SHADER_RESOURCE_VIEW_DESC,
        D3D12_CPU_DESCRIPTOR_HANDLE,
    );
    type CreateUavFn = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *const D3D12_UNORDERED_ACCESS_VIEW_DESC,
        D3D12_CPU_DESCRIPTOR_HANDLE,
    );

    // Output buffer UAV at gpu_heap[2].
    let out_uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: 5,
                StructureByteStride: 16,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_NONE,
            },
        },
    };
    unsafe {
        let f: CreateUavFn = vtable_slot(proxy_ptr, 19);
        f(
            proxy_ptr,
            out_buf_ptr,
            ptr::null_mut(),
            &out_uav_desc,
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: gpu_cpu_base + 2 * view_inc },
        );
    }

    let mut fence_value = 0u64;
    for initial_state in [D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS] {
        let mut tex_ptr = ptr::null_mut();
        let hr = unsafe {
            let f: CommittedFn = vtable_slot(proxy_ptr, 27);
            f(
                proxy_ptr,
                &heap_default,
                D3D12_HEAP_FLAG_NONE,
                &tex_desc,
                initial_state,
                ptr::null(),
                &ID3D12Resource::IID,
                &mut tex_ptr,
            )
        };
        assert!(hr.is_ok(), "slot 27 R9G9B9E5 UAV (initial={initial_state:?}) failed: {hr:?}");
        let _tex = unsafe { IUnknown::from_raw(tex_ptr) };

        // Exercise ALL 4 UAV descriptor forms (R32_UINT, R9G9B9E5, UNKNOWN, nullptr) with 0 debug errors!
        let mut uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        unsafe {
            let f: CreateUavFn = vtable_slot(proxy_ptr, 19);
            // gpu_heap[0] and cpu_heap[0]: explicit R32_UINT UAV.
            f(proxy_ptr, tex_ptr, ptr::null_mut(), &uav_desc, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: gpu_cpu_base });
            f(proxy_ptr, tex_ptr, ptr::null_mut(), &uav_desc, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu_cpu_base });
            // cpu_heap[1]: R9G9B9E5_SHAREDEXP -> redirected to R32_UINT.
            uav_desc.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
            f(proxy_ptr, tex_ptr, ptr::null_mut(), &uav_desc, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu_cpu_base + view_inc });
            // cpu_heap[2]: DXGI_FORMAT_UNKNOWN -> redirected to R32_UINT.
            uav_desc.Format = DXGI_FORMAT_UNKNOWN;
            f(proxy_ptr, tex_ptr, ptr::null_mut(), &uav_desc, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu_cpu_base + 2 * view_inc });
            // cpu_heap[3]: null desc -> synthesized R32_UINT 2D UAV (avoids NVIDIA driver crash!).
            f(proxy_ptr, tex_ptr, ptr::null_mut(), ptr::null(), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu_cpu_base + 3 * view_inc });
        }

        // SRV at gpu_heap[1] with explicit R9G9B9E5_SHAREDEXP (plus null SRV at cpu_heap[4]).
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        unsafe {
            let f: CreateSrvFn = vtable_slot(proxy_ptr, 18);
            f(proxy_ptr, tex_ptr, &srv_desc, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: gpu_cpu_base + view_inc });
            f(proxy_ptr, tex_ptr, ptr::null(), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu_cpu_base + 4 * view_inc });
        }
        assert_no_debug_errors(&dev_native, "UAV and SRV creation");

        // Record command list: optional COMMON->UAV, ClearUAVUint, write CS, UAV->SRV, sample CS, copy to readback.
        type AllocResetFn = unsafe extern "system" fn(*mut c_void) -> HRESULT;
        assert!(unsafe { vtable_slot::<AllocResetFn>(alloc_ptr, 8)(alloc_ptr) }.is_ok());
        let ldesc = ListDesc {
            kind: D3D12_COMMAND_LIST_TYPE_DIRECT,
            flags: 0,
            node_mask: 1,
            reserved: 0,
        };
        let mut list_ptr = ptr::null_mut();
        assert!(unsafe {
            let f: CreateListXFn = vtable_slot(proxy_ptr, 73);
            f(proxy_ptr, &ldesc, alloc_ptr, ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list_ptr)
        }
        .is_ok());
        let _list = unsafe { IUnknown::from_raw(list_ptr) };

        type CloseFn = unsafe extern "system" fn(*mut c_void) -> HRESULT;
        type BarrierFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER);
        type SetHeapsFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
        type SetRootFn = unsafe extern "system" fn(*mut c_void, *mut c_void);
        type SetPsoFn = unsafe extern "system" fn(*mut c_void, *mut c_void);
        type SetTableFn = unsafe extern "system" fn(*mut c_void, u32, D3D12_GPU_DESCRIPTOR_HANDLE);
        type DispatchFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
        type ClearUavUintFn = unsafe extern "system" fn(
            *mut c_void,
            D3D12_GPU_DESCRIPTOR_HANDLE,
            D3D12_CPU_DESCRIPTOR_HANDLE,
            *mut c_void,
            *const u32,
            u32,
            *const windows::Win32::Foundation::RECT,
        );
        type CopyBufFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u64, *mut c_void, u64, u64);

        let emit_transition = |res: *mut c_void, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES| {
            let mut b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                ..Default::default()
            };
            unsafe {
                {
                    let t = &mut *b.Anonymous.Transition;
                    ptr::from_mut(&mut t.pResource).cast::<*mut c_void>().write(res);
                    t.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    t.StateBefore = before;
                    t.StateAfter = after;
                }
                vtable_slot::<BarrierFn>(list_ptr, 26)(list_ptr, 1, &b);
                ptr::from_mut(&mut (*b.Anonymous.Transition).pResource)
                    .cast::<*mut c_void>()
                    .write(ptr::null_mut());
            }
        };

        let heaps = [gpu_heap.as_raw(), samp_heap.as_raw()];
        unsafe { vtable_slot::<SetHeapsFn>(list_ptr, 32)(list_ptr, 2, heaps.as_ptr()) };

        if initial_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            emit_transition(tex_ptr, initial_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // Clear texture via R32_UINT UAV to RGB9E5 bits for (0.5, 1.0, 2.0) = 0xB8100401.
        let clear_vals = [0xB810_0401u32, 0, 0, 0];
        unsafe {
            vtable_slot::<ClearUavUintFn>(list_ptr, 53)(
                list_ptr,
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu_gpu_base },
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu_cpu_base },
                tex_ptr,
                clear_vals.as_ptr(),
                0,
                ptr::null(),
            );
        }

        // UAV barrier between clear and compute write.
        let mut uav_b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            ..Default::default()
        };
        unsafe {
            ptr::from_mut(&mut (*uav_b.Anonymous.UAV).pResource)
                .cast::<*mut c_void>()
                .write(tex_ptr);
            vtable_slot::<BarrierFn>(list_ptr, 26)(list_ptr, 1, &uav_b);
            ptr::from_mut(&mut (*uav_b.Anonymous.UAV).pResource)
                .cast::<*mut c_void>()
                .write(ptr::null_mut());
        }

        // Pass 1: write 4 packed HDR RGB9E5 texels via R32_UINT UAV.
        unsafe {
            vtable_slot::<SetRootFn>(list_ptr, 33)(list_ptr, write_rs.as_raw());
            vtable_slot::<SetPsoFn>(list_ptr, 25)(list_ptr, write_pso.as_raw());
            vtable_slot::<SetTableFn>(list_ptr, 35)(list_ptr, 0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu_gpu_base });
            vtable_slot::<DispatchFn>(list_ptr, 14)(list_ptr, 1, 1, 1);
        }

        // Transition R9G9B9E5 texture UAV -> NON_PIXEL_SHADER_RESOURCE.
        emit_transition(tex_ptr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Pass 2: sample via R9G9B9E5_SHAREDEXP SRV -> float4 structured buffer.
        unsafe {
            vtable_slot::<SetRootFn>(list_ptr, 33)(list_ptr, sample_rs.as_raw());
            vtable_slot::<SetPsoFn>(list_ptr, 25)(list_ptr, sample_pso.as_raw());
            vtable_slot::<SetTableFn>(list_ptr, 35)(
                list_ptr,
                0,
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu_gpu_base + view_inc as u64 },
            );
            vtable_slot::<SetTableFn>(list_ptr, 35)(
                list_ptr,
                1,
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: samp_gpu_base },
            );
            vtable_slot::<DispatchFn>(list_ptr, 14)(list_ptr, 1, 1, 1);
        }

        // Copy out_buf -> readback buffer (buffers decay back to COMMON after ExecuteCommandLists).
        emit_transition(out_buf_ptr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        unsafe {
            vtable_slot::<CopyBufFn>(list_ptr, 15)(list_ptr, rb_ptr, 0, out_buf_ptr, 0, buf_bytes);
        }
        assert!(unsafe { vtable_slot::<CloseFn>(list_ptr, 9)(list_ptr) }.is_ok());

        // Execute and wait on fence.
        type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
        type SignalFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
        type SetEventFn = unsafe extern "system" fn(*mut c_void, u64, HANDLE) -> HRESULT;
        fence_value += 1;
        unsafe {
            vtable_slot::<ExecFn>(queue_ptr, 10)(queue_ptr, 1, &list_ptr);
            assert!(vtable_slot::<SignalFn>(queue_ptr, 14)(queue_ptr, fence_ptr, fence_value).is_ok());
            let event = CreateEventW(None, false, false, None).expect("fence event");
            assert!(vtable_slot::<SetEventFn>(fence_ptr, 9)(fence_ptr, fence_value, event).is_ok());
            let waited = WaitForSingleObject(event, 10_000);
            let _ = CloseHandle(event);
            assert_eq!(waited.0, 0, "GPU fence wait timed out");
        }
        assert_no_debug_errors(&dev_native, "GPU execute");

        // Map readback buffer and assert EXACT IEEE HDR float4 results!
        type MapFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RANGE, *mut *mut c_void) -> HRESULT;
        type UnmapFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RANGE);
        let range = D3D12_RANGE { Begin: 0, End: buf_bytes as usize };
        let empty = D3D12_RANGE::default();
        let mut mapped = ptr::null_mut();
        assert!(unsafe { vtable_slot::<MapFn>(rb_ptr, 8)(rb_ptr, 0, &range, &mut mapped) }.is_ok());
        let actual: [[f32; 4]; 5] = unsafe { ptr::read(mapped.cast()) };
        unsafe { vtable_slot::<UnmapFn>(rb_ptr, 9)(rb_ptr, 0, &empty) };

        let expected: [[f32; 4]; 5] = [
            [1.0, 2.0, 4.0, 1.0],
            [1.0, 0.5, 0.25, 1.0],
            [2000.0, 12.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 1.0, 2.0, 1.0],
        ];
        assert_eq!(
            actual, expected,
            "RGB9E5 UAV->SRV HDR readback mismatch for initial_state={initial_state:?}"
        );
    }
    drop(proxy_device);
}
