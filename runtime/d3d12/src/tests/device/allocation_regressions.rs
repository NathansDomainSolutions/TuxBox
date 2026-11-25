use super::{
    assert_no_debug_errors, enable_desktop_debug_layer, find_non_typed_rgb9e5_adapter, vtable_slot,
};
use crate::device::allocation::{supports_typed_uav, valid_initial_texture_state};
use core::ffi::c_void;
use core::ptr;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{GUID, HRESULT, IUnknown, Interface};

#[repr(C)]
struct ListDesc {
    kind: D3D12_COMMAND_LIST_TYPE,
    flags: u32,
    node_mask: u32,
    reserved: u32,
}

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
type CreateListXFn = unsafe extern "system" fn(
    *mut c_void,
    *const ListDesc,
    *mut c_void,
    *mut c_void,
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
type CreateDescHeapFn = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_DESCRIPTOR_HEAP_DESC,
    *const GUID,
    *mut *mut c_void,
) -> HRESULT;
type IncSizeFn = unsafe extern "system" fn(*mut c_void, D3D12_DESCRIPTOR_HEAP_TYPE) -> u32;
type CpuStartFn = unsafe extern "system" fn(*mut c_void) -> usize;
type CreateUavFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *const D3D12_UNORDERED_ACCESS_VIEW_DESC,
    D3D12_CPU_DESCRIPTOR_HANDLE,
);
type BarrierFn = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER);
type CopyTexFn = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_TEXTURE_COPY_LOCATION,
    u32,
    u32,
    u32,
    *const D3D12_TEXTURE_COPY_LOCATION,
    *const D3D12_BOX,
);
type CloseFn = unsafe extern "system" fn(*mut c_void) -> HRESULT;
type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
type SignalFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> HRESULT;
type SetEventFn = unsafe extern "system" fn(*mut c_void, u64, HANDLE) -> HRESULT;

unsafe fn run_and_wait(queue: *mut c_void, list: *mut c_void, fence: *mut c_void, value: &mut u64) {
    assert!(unsafe { vtable_slot::<CloseFn>(list, 9)(list) }.is_ok());
    *value += 1;
    unsafe {
        vtable_slot::<ExecFn>(queue, 10)(queue, 1, &list);
        assert!(vtable_slot::<SignalFn>(queue, 14)(queue, fence, *value).is_ok());
        let ev = CreateEventW(None, false, false, None).expect("event");
        assert!(vtable_slot::<SetEventFn>(fence, 9)(fence, *value, ev).is_ok());
        let waited = WaitForSingleObject(ev, 10_000);
        let _ = CloseHandle(ev);
        assert_eq!(waited.0, 0, "fence timed out");
    }
}

#[test]
fn valid_initial_texture_state_enforces_render_target_flag() {
    let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let rt_uav = uav | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    assert!(valid_initial_texture_state(D3D12_RESOURCE_STATE_COMMON, uav));
    assert!(valid_initial_texture_state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav));
    assert!(!valid_initial_texture_state(D3D12_RESOURCE_STATE_RENDER_TARGET, uav));
    assert!(valid_initial_texture_state(D3D12_RESOURCE_STATE_RENDER_TARGET, rt_uav));
    assert!(!valid_initial_texture_state(D3D12_RESOURCE_STATE_DEPTH_WRITE, uav));
}

#[test]
fn non_common_packed_uav_supports_implicit_promotion_copy_queue_and_buffer_uav() {
    enable_desktop_debug_layer();
    let Some((adapter, raw_dev)) = find_non_typed_rgb9e5_adapter() else {
        return;
    };
    drop(raw_dev);

    let mut proxy_ptr = ptr::null_mut();
    let hr = unsafe { crate::device::create(adapter.as_raw(), ID3D12Device::IID, &mut proxy_ptr) };
    assert!(hr.is_ok(), "proxy create failed: {hr:?}");
    let proxy_device = unsafe { IUnknown::from_raw(proxy_ptr) };
    let dev_native = unsafe { &(*proxy_ptr.cast::<crate::device::Device>()).native }.clone();

    let heap_default = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let heap_upload = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let tex_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 64,
        Height: 64,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };

    // Create R9G9B9E5 texture in non-COMMON initial state (UNORDERED_ACCESS).
    let mut tex_ptr = ptr::null_mut();
    let hr = unsafe {
        vtable_slot::<CommittedFn>(proxy_ptr, 27)(
            proxy_ptr,
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            &tex_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut tex_ptr,
        )
    };
    assert!(hr.is_ok(), "slot 27 non-COMMON R9G9B9E5 failed: {hr:?}");
    let _tex = unsafe { IUnknown::from_raw(tex_ptr) };

    // F3 regression: non-null Buffer typed UAV with R9G9B9E5_SHAREDEXP and B5G5R5A1_UNORM.
    let buf_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buf_ptr = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CommittedFn>(proxy_ptr, 27)(
            proxy_ptr,
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            &buf_desc,
            D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut buf_ptr,
        )
    }
    .is_ok());
    let _buf = unsafe { IUnknown::from_raw(buf_ptr) };

    let dheap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: 4,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 1,
    };
    let mut dheap_ptr = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CreateDescHeapFn>(proxy_ptr, 14)(
            proxy_ptr,
            &dheap_desc,
            &ID3D12DescriptorHeap::IID,
            &mut dheap_ptr,
        )
    }
    .is_ok());
    let _dheap = unsafe { IUnknown::from_raw(dheap_ptr) };
    let inc = unsafe {
        vtable_slot::<IncSizeFn>(proxy_ptr, 15)(proxy_ptr, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            as usize
    };
    let cpu0 = unsafe { vtable_slot::<CpuStartFn>(dheap_ptr, 9)(dheap_ptr) };
    for (idx, (fmt, elems)) in [
        (DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 64u32),
        (DXGI_FORMAT_B5G5R5A1_UNORM, 128u32),
    ]
    .into_iter()
    .enumerate()
    {
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: fmt,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: elems,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                },
            },
        };
        unsafe {
            vtable_slot::<CreateUavFn>(proxy_ptr, 19)(
                proxy_ptr,
                buf_ptr,
                ptr::null_mut(),
                &uav,
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu0 + idx * inc },
            );
        }
    }
    assert!(unsafe { dev_native.GetDeviceRemovedReason() }.is_ok(), "buffer UAV triggered device removal");
    assert_no_debug_errors(&dev_native, "F3 buffer packed UAV");

    // Upload buffer for F4 implicit promotion and F1 COPY-queue copy.
    let upload_desc = D3D12_RESOURCE_DESC {
        Width: 64 * 256,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..buf_desc
    };
    let mut upload_ptr = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CommittedFn>(proxy_ptr, 27)(
            proxy_ptr,
            &heap_upload,
            D3D12_HEAP_FLAG_NONE,
            &upload_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut upload_ptr,
        )
    }
    .is_ok());
    let _upload = unsafe { IUnknown::from_raw(upload_ptr) };

    let make_queue = |kind| -> IUnknown {
        let d = D3D12_COMMAND_QUEUE_DESC { Type: kind, ..Default::default() };
        let mut p = ptr::null_mut();
        assert!(unsafe {
            vtable_slot::<CreateQueueFn>(proxy_ptr, 8)(proxy_ptr, &d, &ID3D12CommandQueue::IID, &mut p)
        }
        .is_ok());
        unsafe { IUnknown::from_raw(p) }
    };
    let make_alloc = |kind| -> IUnknown {
        let mut p = ptr::null_mut();
        assert!(unsafe {
            vtable_slot::<CreateAllocFn>(proxy_ptr, 9)(proxy_ptr, kind, &ID3D12CommandAllocator::IID, &mut p)
        }
        .is_ok());
        unsafe { IUnknown::from_raw(p) }
    };
    let make_list = |kind, alloc: *mut c_void| -> IUnknown {
        let d = ListDesc { kind, flags: 0, node_mask: 1, reserved: 0 };
        let mut p = ptr::null_mut();
        assert!(unsafe {
            vtable_slot::<CreateListXFn>(proxy_ptr, 73)(
                proxy_ptr,
                &d,
                alloc,
                ptr::null_mut(),
                &ID3D12GraphicsCommandList::IID,
                &mut p,
            )
        }
        .is_ok());
        unsafe { IUnknown::from_raw(p) }
    };
    let direct_q = make_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    let copy_q = make_queue(D3D12_COMMAND_LIST_TYPE_COPY);
    let mut fence_ptr = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CreateFenceFn>(proxy_ptr, 36)(
            proxy_ptr,
            0,
            D3D12_FENCE_FLAG_NONE,
            &ID3D12Fence::IID,
            &mut fence_ptr,
        )
    }
    .is_ok());
    let _fence = unsafe { IUnknown::from_raw(fence_ptr) };
    let mut fv = 0u64;

    let emit_transition = |list: *mut c_void, before, after| {
        let mut b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            ..Default::default()
        };
        unsafe {
            {
                let t = &mut *b.Anonymous.Transition;
                ptr::from_mut(&mut t.pResource).cast::<*mut c_void>().write(tex_ptr);
                t.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                t.StateBefore = before;
                t.StateAfter = after;
            }
            vtable_slot::<BarrierFn>(list, 26)(list, 1, &b);
            ptr::from_mut(&mut (*b.Anonymous.Transition).pResource)
                .cast::<*mut c_void>()
                .write(ptr::null_mut());
        }
    };

    // F4 step 1: transition UAV -> COMMON and execute.
    {
        let alloc = make_alloc(D3D12_COMMAND_LIST_TYPE_DIRECT);
        let list = make_list(D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.as_raw());
        emit_transition(list.as_raw(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        unsafe { run_and_wait(direct_q.as_raw(), list.as_raw(), fence_ptr, &mut fv) };
    }

    // F4 step 2: implicit promotion from COMMON via CopyTextureRegion, then COPY_DEST -> UAV.
    let copy_tex = |list: *mut c_void| {
        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..Default::default()
        };
        let mut src = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
                        Width: 64,
                        Height: 64,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
            ..Default::default()
        };
        unsafe {
            ptr::from_mut(&mut dst.pResource).cast::<*mut c_void>().write(tex_ptr);
            ptr::from_mut(&mut src.pResource).cast::<*mut c_void>().write(upload_ptr);
            vtable_slot::<CopyTexFn>(list, 16)(list, &dst, 0, 0, 0, &src, ptr::null());
            ptr::from_mut(&mut dst.pResource).cast::<*mut c_void>().write(ptr::null_mut());
            ptr::from_mut(&mut src.pResource).cast::<*mut c_void>().write(ptr::null_mut());
        }
    };
    {
        let alloc = make_alloc(D3D12_COMMAND_LIST_TYPE_DIRECT);
        let list = make_list(D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.as_raw());
        copy_tex(list.as_raw());
        emit_transition(list.as_raw(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        emit_transition(list.as_raw(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        unsafe { run_and_wait(direct_q.as_raw(), list.as_raw(), fence_ptr, &mut fv) };
    }
    assert_no_debug_errors(&dev_native, "F4 implicit promotion after UAV->COMMON");

    // F1 step: COPY queue copy on the non-COMMON-created resource (now in COMMON).
    {
        let alloc = make_alloc(D3D12_COMMAND_LIST_TYPE_COPY);
        let list = make_list(D3D12_COMMAND_LIST_TYPE_COPY, alloc.as_raw());
        copy_tex(list.as_raw());
        unsafe { run_and_wait(copy_q.as_raw(), list.as_raw(), fence_ptr, &mut fv) };
    }
    assert_no_debug_errors(&dev_native, "F1 COPY queue on packed UAV texture");
    drop(proxy_device);
}

#[test]
fn native_typed_uav_adapter_warp_keeps_native_path_untouched() {
    enable_desktop_debug_layer();
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory4>() }) else {
        return;
    };
    let Ok(warp) = (unsafe { factory.EnumWarpAdapter::<IDXGIAdapter1>() }) else {
        return;
    };
    let mut raw: Option<ID3D12Device> = None;
    if unsafe { D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_12_0, &mut raw) }.is_err() {
        return;
    }
    let Some(raw) = raw else { return };
    if !supports_typed_uav(&raw, DXGI_FORMAT_R9G9B9E5_SHAREDEXP) {
        return;
    }
    drop(raw);

    let mut proxy_ptr = ptr::null_mut();
    assert!(unsafe { crate::device::create(warp.as_raw(), ID3D12Device::IID, &mut proxy_ptr) }.is_ok());
    let proxy_device = unsafe { IUnknown::from_raw(proxy_ptr) };
    let dev_native = unsafe { &(*proxy_ptr.cast::<crate::device::Device>()).native }.clone();

    let heap_default = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let tex_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 32,
        Height: 32,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };

    // Slot 25 multi-resource batch stays native (not split).
    type AllocInfoFn = unsafe extern "system" fn(
        *mut D3D12_RESOURCE_ALLOCATION_INFO,
        *mut c_void,
        u32,
        u32,
        *const D3D12_RESOURCE_DESC,
    ) -> *mut D3D12_RESOURCE_ALLOCATION_INFO;
    let batch = [tex_desc, tex_desc];
    let mut proxy_info = D3D12_RESOURCE_ALLOCATION_INFO::default();
    unsafe {
        vtable_slot::<AllocInfoFn>(proxy_ptr, 25)(&mut proxy_info, proxy_ptr, 0, 2, batch.as_ptr());
    }
    let native_info = unsafe { dev_native.GetResourceAllocationInfo(0, &batch) };
    assert_eq!(proxy_info.SizeInBytes, native_info.SizeInBytes);
    assert_eq!(proxy_info.Alignment, native_info.Alignment);

    // Slot 27 + Slot 127 (count=0) + native R9G9B9E5 UAV succeed natively with 0 debug errors.
    let mut tex_ptr = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CommittedFn>(proxy_ptr, 27)(
            proxy_ptr,
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            &tex_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut tex_ptr,
        )
    }
    .is_ok());
    let _tex = unsafe { IUnknown::from_raw(tex_ptr) };

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
    let mut tex127 = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CommittedLayoutFn>(proxy_ptr, 127)(
            proxy_ptr,
            &heap_default,
            D3D12_HEAP_FLAG_NONE,
            &tex_desc,
            D3D12_BARRIER_LAYOUT_COMMON,
            ptr::null(),
            ptr::null_mut(),
            0,
            ptr::null(),
            &ID3D12Resource::IID,
            &mut tex127,
        )
    }
    .is_ok());
    let _tex127 = unsafe { IUnknown::from_raw(tex127) };

    let dheap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: 2,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 1,
    };
    let mut dheap_ptr = ptr::null_mut();
    assert!(unsafe {
        vtable_slot::<CreateDescHeapFn>(proxy_ptr, 14)(
            proxy_ptr,
            &dheap_desc,
            &ID3D12DescriptorHeap::IID,
            &mut dheap_ptr,
        )
    }
    .is_ok());
    let _dheap = unsafe { IUnknown::from_raw(dheap_ptr) };
    let cpu0 = unsafe { vtable_slot::<CpuStartFn>(dheap_ptr, 9)(dheap_ptr) };
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    unsafe {
        vtable_slot::<CreateUavFn>(proxy_ptr, 19)(
            proxy_ptr,
            tex_ptr,
            ptr::null_mut(),
            &uav,
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu0 },
        );
    }
    assert_no_debug_errors(&dev_native, "F2 WARP native R9G9B9E5 path");
    drop(proxy_device);
}
