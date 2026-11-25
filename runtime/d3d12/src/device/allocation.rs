//! Frosty reads a 16-byte result and hashes each 56-byte resource description.
//! Slot 25 passes the result in RCX and the receiver in RDX (title RVA 0x57945e9).
use super::Device;
use core::slice;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
};
use windows::core::Interface;

mod trace;

const _: () = assert!(
    size_of::<D3D12_RESOURCE_DESC>() == 56 && size_of::<D3D12_RESOURCE_ALLOCATION_INFO>() == 16
);

// DCC is selected by the console. The host driver chooses compression for its
// own image, so allocation and copy-footprint queries must match image creation.
pub(super) const DCC_FLAGS: i32 = 0x00c0_0000;

fn host_description(desc: &D3D12_RESOURCE_DESC) -> Option<D3D12_RESOURCE_DESC> {
    use super::format::HOST_FLAGS;
    // Console-only flag bits (including DCC) are dropped; the host driver
    // chooses its own compression for the image it creates.
    super::depth::primary(desc).or_else(|| {
        (desc.Flags.0 & !HOST_FLAGS != 0)
            .then(|| D3D12_RESOURCE_DESC { Flags: D3D12_RESOURCE_FLAGS(desc.Flags.0 & HOST_FLAGS), ..*desc })
    })
}

// Preserve packed sampling/render formats while allowing raw integer storage
// views on adapters without typed UAV support (BGRA5551 -> R16_UINT,
// R9G9B9E5_SHAREDEXP -> R32_UINT).
pub(crate) fn packed_uav_raw_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    match format {
        DXGI_FORMAT_B5G5R5A1_UNORM => Some(DXGI_FORMAT_R16_UINT),
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP => Some(DXGI_FORMAT_R32_UINT),
        _ => None,
    }
}

pub(super) fn packed_uav_formats(format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> &'static [DXGI_FORMAT] {
    if !flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
        return &[];
    }
    match format {
        DXGI_FORMAT_B5G5R5A1_UNORM => &[DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_R16_UINT],
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP => &[DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_UINT],
        _ => &[],
    }
}

pub(crate) fn supports_typed_uav(device: &ID3D12Device, format: DXGI_FORMAT) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: format,
        ..Default::default()
    };
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut support).cast(),
            size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
}

pub(super) fn needs_packed_uav_cast(
    device: &ID3D12Device,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
) -> bool {
    !packed_uav_formats(format, flags).is_empty() && !supports_typed_uav(device, format)
}

pub(super) fn resource_desc1(desc: &D3D12_RESOURCE_DESC) -> D3D12_RESOURCE_DESC1 {
    D3D12_RESOURCE_DESC1 {
        Dimension: desc.Dimension,
        Alignment: desc.Alignment,
        Width: desc.Width,
        Height: desc.Height,
        DepthOrArraySize: desc.DepthOrArraySize,
        MipLevels: desc.MipLevels,
        Format: desc.Format,
        SampleDesc: desc.SampleDesc,
        Layout: desc.Layout,
        Flags: desc.Flags,
        ..Default::default()
    }
}

/// Strip `ALLOW_UNORDERED_ACCESS` when querying copyable footprints for packed
/// UAV textures: `GetCopyableFootprints` validates resource flags without a
/// castable-format list and rejects formats lacking native typed UAV support.
pub(crate) fn copy_footprint_desc(desc: &D3D12_RESOURCE_DESC) -> Option<D3D12_RESOURCE_DESC> {
    let base = host_description(desc);
    let candidate = base.as_ref().unwrap_or(desc);
    if !packed_uav_formats(candidate.Format, candidate.Flags).is_empty() {
        Some(D3D12_RESOURCE_DESC {
            Flags: D3D12_RESOURCE_FLAGS(
                candidate.Flags.0 & !D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0,
            ),
            ..*candidate
        })
    } else {
        base
    }
}

/// Convert a legacy texture `D3D12_RESOURCE_STATES` mask (with buffer-only read
/// bits already stripped) into its enhanced-barrier initial layout.
pub(crate) fn initial_barrier_layout(state: D3D12_RESOURCE_STATES) -> Option<D3D12_BARRIER_LAYOUT> {
    if state == D3D12_RESOURCE_STATE_COMMON {
        return Some(D3D12_BARRIER_LAYOUT_COMMON);
    }
    if state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
        return Some(D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    }
    if state == D3D12_RESOURCE_STATE_RENDER_TARGET {
        return Some(D3D12_BARRIER_LAYOUT_RENDER_TARGET);
    }
    if state == D3D12_RESOURCE_STATE_COPY_DEST {
        return Some(D3D12_BARRIER_LAYOUT_COPY_DEST);
    }
    if state == D3D12_RESOURCE_STATE_COPY_SOURCE {
        return Some(D3D12_BARRIER_LAYOUT_COPY_SOURCE);
    }
    if state == D3D12_RESOURCE_STATE_RESOLVE_DEST {
        return Some(D3D12_BARRIER_LAYOUT_RESOLVE_DEST);
    }
    if state == D3D12_RESOURCE_STATE_RESOLVE_SOURCE {
        return Some(D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE);
    }
    if state == D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE {
        return Some(D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE);
    }
    let srv_bits = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0;
    if state.0 & !srv_bits == 0 {
        return Some(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE);
    }
    let texture_read_bits = srv_bits | D3D12_RESOURCE_STATE_COPY_SOURCE.0;
    if state == D3D12_RESOURCE_STATE_GENERIC_READ
        || (state.0 & texture_read_bits != 0
            && state.0 & !D3D12_RESOURCE_STATE_GENERIC_READ.0 == 0)
    {
        return Some(D3D12_BARRIER_LAYOUT_GENERIC_READ);
    }
    None
}

pub(super) fn valid_initial_texture_state(
    state: D3D12_RESOURCE_STATES,
    flags: D3D12_RESOURCE_FLAGS,
) -> bool {
    if state == D3D12_RESOURCE_STATE_RENDER_TARGET
        && !flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
    {
        return false;
    }
    initial_barrier_layout(state).is_some()
}

/// One-shot legacy transition queue used only when a legacy state-based slot
/// (`CreateCommittedResource`, `CreatePlacedResource`, `CreateReservedResource`)
/// creates a packed-UAV texture via `Create*Resource3/2(..., COMMON)` with a
/// non-`COMMON` requested initial state. Keeping the resource on the legacy
/// state timeline preserves native COPY-queue rules, implicit promotion/decay,
/// and all legacy barrier types without custom barrier translation.
pub(super) struct InitTransition {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
}

impl InitTransition {
    fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        };
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc)? };
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)? };
        let list: ID3D12GraphicsCommandList =
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)? };
        unsafe { list.Close()? };
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        Ok(Self { queue, allocator, list, fence, value: 0 })
    }

    fn apply(
        &mut self,
        resource: &ID3D12Resource,
        target: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<()> {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
        }
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: core::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: core::mem::ManuallyDrop::new(None),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COMMON,
                    StateAfter: target,
                }),
            },
        };
        unsafe {
            core::ptr::from_mut(&mut (*barrier.Anonymous.Transition).pResource)
                .cast::<*mut core::ffi::c_void>()
                .write(resource.as_raw());
            self.list.ResourceBarrier(&[barrier]);
            self.list.Close()?;
            let command_list: ID3D12CommandList = self.list.clone().into();
            self.queue.ExecuteCommandLists(&[Some(command_list)]);
            self.value += 1;
            self.queue.Signal(&self.fence, self.value)?;
            if self.fence.GetCompletedValue() < self.value {
                let event = CreateEventW(None, false, false, None)?;
                let set = self.fence.SetEventOnCompletion(self.value, event);
                if set.is_ok() {
                    WaitForSingleObject(event, INFINITE);
                }
                let _ = CloseHandle(event);
                set?;
            }
        }
        Ok(())
    }
}

pub(super) fn transition_initial_state(
    device: &Device,
    resource: &ID3D12Resource,
    target: D3D12_RESOURCE_STATES,
    flags: D3D12_RESOURCE_FLAGS,
) -> Result<(), windows::core::HRESULT> {
    if target == D3D12_RESOURCE_STATE_COMMON
        || flags.contains(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)
    {
        return Ok(());
    }
    let mut slot = device
        .state
        .init_transition
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let runner = match slot.as_mut() {
        Some(runner) => runner,
        None => match InitTransition::new(&device.native) {
            Ok(runner) => slot.insert(runner),
            Err(error) => return Err(error.code()),
        },
    };
    if let Err(error) = runner.apply(resource, target) {
        *slot = None;
        return Err(error.code());
    }
    Ok(())
}

pub(super) fn packed_requirements(
    device: &ID3D12Device, visible: u32, desc: &D3D12_RESOURCE_DESC1,
) -> Option<D3D12_RESOURCE_ALLOCATION_INFO> {
    let formats = packed_uav_formats(desc.Format, desc.Flags);
    if formats.is_empty() { return None; }
    let native = D3D12_RESOURCE_DESC1 { Flags: D3D12_RESOURCE_FLAGS(desc.Flags.0 & 0x1ff), ..*desc };
    let count = formats.len() as u32;
    let entries = formats.as_ptr();
    let api: ID3D12Device12 = device.cast().ok()?;
    Some(unsafe { api.GetResourceAllocationInfo3(visible, 1, &native, Some(&count), Some(&entries), None) })
}

pub(super) fn allocation_info_for(
    device: &ID3D12Device,
    visible: u32,
    desc: &D3D12_RESOURCE_DESC,
) -> D3D12_RESOURCE_ALLOCATION_INFO {
    let host = host_description(desc);
    let desc = host.as_ref().unwrap_or(desc);
    if needs_packed_uav_cast(device, desc.Format, desc.Flags)
        && let Some(info) = packed_requirements(device, visible, &resource_desc1(desc))
    {
        return info;
    }
    unsafe { device.GetResourceAllocationInfo(visible, core::slice::from_ref(desc)) }
}

// Like GetResourceAllocationInfo, this Xbox vtable entry places the struct
// return pointer before the device. GTA queries default, upload and readback.
pub(super) unsafe extern "system" fn custom_heap_properties(
    out: *mut D3D12_HEAP_PROPERTIES,
    device: &Device,
    node_mask: u32,
    heap_type: D3D12_HEAP_TYPE,
) -> *mut D3D12_HEAP_PROPERTIES {
    if !out.is_null() {
        unsafe { out.write(device.native.GetCustomHeapProperties(node_mask, heap_type)) };
    }
    out
}

pub(super) unsafe extern "system" fn info(
    out: *mut D3D12_RESOURCE_ALLOCATION_INFO,
    device: &Device,
    visible: u32,
    count: u32,
    descs: *const D3D12_RESOURCE_DESC,
) -> *mut D3D12_RESOURCE_ALLOCATION_INFO {
    let result = unsafe { info_impl(out, device, visible, count, descs) };
    if trace::enabled() && !result.is_null() && !descs.is_null() && count != 0 && visible <= 1 {
        let answer = unsafe { &*result };
        if trace::matches(answer.SizeInBytes) {
            let items = unsafe { slice::from_raw_parts(descs, count as usize) };
            trace::report("info", items, answer);
        }
    }
    result
}

unsafe fn info_impl(
    out: *mut D3D12_RESOURCE_ALLOCATION_INFO,
    device: &Device,
    visible: u32,
    count: u32,
    descs: *const D3D12_RESOURCE_DESC,
) -> *mut D3D12_RESOURCE_ALLOCATION_INFO {
    let Some(out) = (unsafe { out.as_mut() }) else {
        return out;
    };
    let invalid = D3D12_RESOURCE_ALLOCATION_INFO {
        SizeInBytes: u64::MAX,
        Alignment: 0,
    };
    if descs.is_null() || count == 0 || visible > 1 {
        *out = invalid;
        return out;
    }
    let descs = unsafe { slice::from_raw_parts(descs, count as usize) };
    let needs_packed = |d: &D3D12_RESOURCE_DESC| {
        !packed_uav_formats(d.Format, d.Flags).is_empty()
            && !super::dxgi::supports_native_packed_uav(device, d.Format)
    };
    if descs.iter().any(|d| super::placement::console_layout(d.Layout.0) || needs_packed(d)) {
        // The guest will allocate these bytes itself and pass the address to
        // CreatePlacedResourceX. Account for the Xbox planes and batch padding.
        let mut bytes = 0u64;
        let mut alignment = 256;
        for desc in descs {
            let layout = if let Some(pitch) = super::placement::nv12_pitch(desc) {
                let y = pitch * u64::from(desc.Height);
                D3D12_RESOURCE_ALLOCATION_INFO { SizeInBytes: y + y / 2, Alignment: 256 }
            } else if super::placement::console_layout(desc.Layout.0) {
                match super::placement::color_plane(desc) {
                    Ok(plane) => D3D12_RESOURCE_ALLOCATION_INFO {
                        SizeInBytes: plane.size,
                        Alignment: plane.alignment.max(desc.Alignment),
                    },
                    Err(_) => invalid,
                }
            } else if needs_packed(desc) {
                allocation_info_for(&device.native, visible, desc)
            } else {
                let mut item = invalid;
                unsafe { info(&mut item, device, visible, 1, desc) };
                item
            };
            if layout.Alignment == 0 || layout.SizeInBytes == u64::MAX {
                static BAD_LAYOUT: std::sync::Once = std::sync::Once::new();
                crate::diagnostics::log_once(&BAD_LAYOUT, || format!("Xbox layout {:#x} dim={} {}x{}x{} format={} flags={:#x} alignment={} mips={} samples={}", desc.Layout.0, desc.Dimension.0, desc.Width, desc.Height, desc.DepthOrArraySize, desc.Format.0, desc.Flags.0, desc.Alignment, desc.MipLevels, desc.SampleDesc.Count));
                *out = invalid;
                return out;
            }
            let Some(end) = bytes.checked_next_multiple_of(layout.Alignment)
                .and_then(|offset| offset.checked_add(layout.SizeInBytes)) else {
                *out = invalid;
                return out;
            };
            bytes = end;
            alignment = alignment.max(layout.Alignment);
        }
        *out = D3D12_RESOURCE_ALLOCATION_INFO {
            SizeInBytes: bytes.checked_next_multiple_of(alignment).unwrap_or(u64::MAX), Alignment: alignment,
        };
        return out;
    }
    let supported = descs.iter().all(|d| {
        matches!(
            d.Dimension,
            D3D12_RESOURCE_DIMENSION_BUFFER
                | D3D12_RESOURCE_DIMENSION_TEXTURE1D
                | D3D12_RESOURCE_DIMENSION_TEXTURE2D
                | D3D12_RESOURCE_DIMENSION_TEXTURE3D
        ) && matches!(
            d.Layout,
            D3D12_TEXTURE_LAYOUT_UNKNOWN | D3D12_TEXTURE_LAYOUT_ROW_MAJOR
        ) && d.Width != 0
            && d.Height != 0
            && d.DepthOrArraySize != 0
            && d.SampleDesc.Count != 0
    });
    *out = if supported {
        // Let the same adapter that creates the placed resource determine its
        // requirements, including padding between descriptors in a batch.
        if descs.iter().any(|d| host_description(d).is_some()) {
            let mut host = Vec::new();
            if host.try_reserve_exact(descs.len()).is_err() {
                *out = invalid;
                return out;
            }
            host.extend(descs.iter().map(|d| host_description(d).unwrap_or(*d)));
            unsafe { device.native.GetResourceAllocationInfo(visible, &host) }
        } else {
            unsafe { device.native.GetResourceAllocationInfo(visible, descs) }
        }
    } else {
        invalid
    };
    if out.SizeInBytes == u64::MAX {
        let d = descs[0];
        static INVALID: std::sync::Once = std::sync::Once::new();
        crate::diagnostics::log_once(&INVALID, || format!("allocation requires dim={} {}x{}x{} mips={} format={} layout={:#x} flags={:#x} align={}", d.Dimension.0,d.Width,d.Height,d.DepthOrArraySize,d.MipLevels,d.Format.0,d.Layout.0,d.Flags.0,d.Alignment));
    }
    out
}

// Device8 extends each description with a sampler-feedback mip region and may
// return a placement offset, alignment and size for each individual resource.
pub(super) unsafe extern "system" fn info2(
    out: *mut D3D12_RESOURCE_ALLOCATION_INFO,
    device: &Device,
    visible: u32,
    count: u32,
    descs: *const D3D12_RESOURCE_DESC1,
    details: *mut D3D12_RESOURCE_ALLOCATION_INFO1,
) -> *mut D3D12_RESOURCE_ALLOCATION_INFO {
    let Some(result) = (unsafe { out.as_mut() }) else { return out };
    let invalid = D3D12_RESOURCE_ALLOCATION_INFO { SizeInBytes: u64::MAX, Alignment: 0 };
    *result = invalid;
    if descs.is_null() || count == 0 || visible > 1 { return out; }
    let inputs = unsafe { slice::from_raw_parts(descs, count as usize) };
    let mut offset = 0u64;
    let mut alignment = 1;
    let mut placements = Vec::new();
    if !details.is_null() && placements.try_reserve_exact(inputs.len()).is_err() { return out; }
    for d in inputs {
        let requirement = if d.Layout.0 >= 0x100 {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: d.Dimension, Alignment: d.Alignment, Width: d.Width, Height: d.Height,
                DepthOrArraySize: d.DepthOrArraySize, MipLevels: d.MipLevels, Format: d.Format,
                SampleDesc: d.SampleDesc, Layout: d.Layout, Flags: d.Flags,
            };
            let mut item = invalid;
            unsafe { info(&mut item, device, visible, 1, &desc) };
            item
        } else if !super::dxgi::supports_native_packed_uav(device, d.Format)
            && let Some(requirement) = packed_requirements(&device.native, visible, d)
        {
            requirement
        } else if let Ok(host) = device.native.cast::<ID3D12Device8>() {
            // Image creation keeps only the desktop bits, DCC included, so the
            // requirements must be queried with the same flags.
            let mut native = *d;
            native.Flags.0 &= super::format::HOST_FLAGS;
            let guest = D3D12_RESOURCE_DESC {
                Dimension: d.Dimension, Alignment: d.Alignment, Width: d.Width, Height: d.Height,
                DepthOrArraySize: d.DepthOrArraySize, MipLevels: d.MipLevels, Format: d.Format,
                SampleDesc: d.SampleDesc, Layout: d.Layout, Flags: d.Flags,
            };
            if let Some(primary) = super::depth::primary(&guest) {
                native.Format = primary.Format;
                native.Flags = primary.Flags;
            }
            unsafe { host.GetResourceAllocationInfo2(visible, 1, &native, None) }
        } else { return out };
        if trace::matches(requirement.SizeInBytes) {
            let guest = D3D12_RESOURCE_DESC {
                Dimension: d.Dimension, Alignment: d.Alignment, Width: d.Width, Height: d.Height,
                DepthOrArraySize: d.DepthOrArraySize, MipLevels: d.MipLevels, Format: d.Format,
                SampleDesc: d.SampleDesc, Layout: d.Layout, Flags: d.Flags,
            };
            trace::report("info2-item", core::slice::from_ref(&guest), &requirement);
        }
        if requirement.Alignment == 0 || requirement.SizeInBytes == u64::MAX {
            crate::diagnostics::recording_error(format_args!("allocation info2 dim={} {}x{} format={} flags={:#x} layout={:#x} alignment={}", d.Dimension.0, d.Width, d.Height, d.Format.0, d.Flags.0, d.Layout.0, d.Alignment));
            return out;
        }
        let Some(start) = offset.checked_next_multiple_of(requirement.Alignment) else { return out };
        let Some(end) = start.checked_add(requirement.SizeInBytes) else { return out };
        if !details.is_null() {
            placements.push(D3D12_RESOURCE_ALLOCATION_INFO1 {
                Offset: start, Alignment: requirement.Alignment, SizeInBytes: requirement.SizeInBytes,
            });
        }
        offset = end;
        alignment = alignment.max(requirement.Alignment);
    }
    let Some(total) = offset.checked_next_multiple_of(alignment) else { return out };
    *result = D3D12_RESOURCE_ALLOCATION_INFO { SizeInBytes: total, Alignment: alignment };
    if !placements.is_empty() {
        unsafe { core::ptr::copy_nonoverlapping(placements.as_ptr(), details, placements.len()) };
    }
    out
}

// The shipping texture upload routine (RVA 0x57e12a1) supplies independent
// arrays with strides 32, 4 and 8 for layout, row count and unpadded row size.
pub(super) unsafe extern "system" fn footprints(
    device: &Device,
    desc: *const D3D12_RESOURCE_DESC,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: *mut u32,
    row_bytes: *mut u64,
    total: *mut u64,
) {
    const _: () = assert!(size_of::<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>() == 32);
    let native = unsafe { desc.as_ref() }.and_then(|d| {
        if super::dxgi::supports_native_packed_uav(device, d.Format) {
            host_description(d)
        } else {
            copy_footprint_desc(d)
        }
    });
    let desc = native.as_ref().map_or(desc, core::ptr::from_ref);
    // All four output arrays are optional. Their storage is owned by the guest.
    unsafe {
        (device.native.vtable().GetCopyableFootprints)(
            device.native.as_raw(),
            desc,
            first,
            count,
            base,
            layouts,
            rows,
            row_bytes,
            total,
        );
    }
}

#[cfg(test)]
#[path = "../tests/device/allocation.rs"]
mod tests;
