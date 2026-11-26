//! Whether the adapter's imports of section-backed memory alias that memory.
//!
//! XMem physical pages are section views. Wine maps sections as shared memory,
//! and RADV refuses to import shared memory as a host pointer. vkd3d-proton
//! then allocates unrelated memory and still reports success, so neither the
//! import result nor a CPU `Map` of the heap shows it: only a GPU read does.
//!
//! The method follows pull request #199 by smith-3.
use super::SectionImport;
use core::ffi::c_void;
use windows::Win32::Foundation::{CloseHandle, E_POINTER, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, Result};

const CANARY: [u32; 4] = [0x7e57_c0de, 0x5ec7_10a5, 0xa11a_5ed0, 0x0ddb_a115];
const SECTION_BYTES: usize = 65536;
/// A driver that never finishes a 16-byte copy must not hang the title.
const WAIT_MILLISECONDS: u32 = 5000;

pub(super) fn run(device: &ID3D12Device) -> SectionImport {
    probe(device).unwrap_or_else(|error| SectionImport::Unknown(error.code().0))
}

fn probe(device: &ID3D12Device) -> Result<SectionImport> {
    // SAFETY: an unnamed pagefile section of a fixed size.
    let section = unsafe {
        CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE, 0, SECTION_BYTES as u32, None)?
    };
    // SAFETY: the section is live and at least SECTION_BYTES long.
    let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SECTION_BYTES) };
    let mut abandoned = false;
    let result = if view.Value.is_null() {
        Err(windows::core::Error::from_win32())
    } else {
        // SAFETY: the view is a fresh writable mapping of SECTION_BYTES bytes.
        unsafe { core::ptr::copy_nonoverlapping(CANARY.as_ptr(), view.Value.cast(), CANARY.len()) };
        read_with_gpu(device, view.Value, &mut abandoned)
    };
    // A copy that may still be running keeps its view and section until exit.
    if !abandoned {
        // SAFETY: no GPU work can reach the view now.
        unsafe {
            if !view.Value.is_null() {
                let _ = UnmapViewOfFile(view);
            }
            let _ = CloseHandle(section);
        }
    }
    result
}

struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: this holds the only copy of the handle.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

fn buffer(flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size_of_val(&CANARY) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// Copy the imported view on the GPU and compare what arrived. Everything that
/// can fail is created before the copy is submitted. `abandoned` is set, and the
/// native objects are kept, if the copy cannot be shown to have finished.
fn read_with_gpu(device: &ID3D12Device, view: *mut c_void, abandoned: &mut bool) -> Result<SectionImport> {
    let bytes = size_of_val(&CANARY) as u64;
    let host: ID3D12Device3 = device.cast()?;
    // SAFETY: the view stays mapped while the heap can be used.
    let heap: ID3D12Heap = match unsafe { host.OpenExistingHeapFromAddress(view) } {
        Ok(heap) => heap,
        Err(_) => return Ok(SectionImport::Refused),
    };
    let mut source: Option<ID3D12Resource> = None;
    // SAFETY: the buffer fits at the start of the imported heap.
    unsafe {
        device.CreatePlacedResource(&heap, 0, &buffer(D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER),
            D3D12_RESOURCE_STATE_COMMON, None, &mut source)?
    };
    let source = source.ok_or(E_POINTER)?;
    let properties = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut readback: Option<ID3D12Resource> = None;
    // SAFETY: a plain readback buffer.
    unsafe {
        device.CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer(D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)?
    };
    let readback = readback.ok_or(E_POINTER)?;
    let kind = D3D12_COMMAND_LIST_TYPE_COPY;
    // SAFETY: a private copy queue, allocator, list and fence on a live device.
    let (queue, allocator, list, fence) = unsafe {
        let queue: ID3D12CommandQueue =
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: kind, ..Default::default() })?;
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(kind)?;
        let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, kind, &allocator, None)?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        (queue, allocator, list, fence)
    };
    // SAFETY: an unnamed manual-reset event.
    let event = Event(unsafe { CreateEventW(None, true, false, None)? });
    // SAFETY: both buffers hold `bytes` bytes. COMMON promotes to COPY_SOURCE.
    let commands = unsafe {
        fence.SetEventOnCompletion(1, event.0)?;
        list.CopyBufferRegion(&readback, 0, &source, 0, bytes);
        list.Close()?;
        list.cast::<ID3D12CommandList>()?
    };
    // SAFETY: the list is closed and its resources are held here.
    let finished = unsafe {
        queue.ExecuteCommandLists(&[Some(commands)]);
        queue.Signal(&fence, 1).is_ok() && WaitForSingleObject(event.0, WAIT_MILLISECONDS) == WAIT_OBJECT_0
    };
    if !finished {
        *abandoned = true;
        core::mem::forget((heap, source, readback, queue, allocator, list, fence, event));
        return Err(windows::core::Error::from(windows::Win32::Foundation::E_FAIL));
    }
    let mut mapped = core::ptr::null_mut();
    // SAFETY: the readback buffer holds `bytes` bytes and stays mapped below.
    unsafe { readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: bytes as usize }), Some(&mut mapped))? };
    // SAFETY: as above; the data is u32 aligned.
    let read = unsafe { core::slice::from_raw_parts(mapped.cast::<u32>(), CANARY.len()) } == CANARY;
    // SAFETY: mapped above, and nothing was written.
    unsafe { readback.Unmap(0, Some(&D3D12_RANGE::default())) };
    Ok(if read { SectionImport::Aliased } else { SectionImport::Substituted })
}
