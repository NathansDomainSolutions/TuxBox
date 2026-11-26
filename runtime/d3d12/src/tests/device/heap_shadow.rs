use super::*;
use crate::commands::page_mapping::GPU_APERTURE;
use windows::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEM_RELEASE, MEM_RESERVE, MapViewOfFile,
    PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc, VirtualFree,
};

struct MappedPage {
    section: windows::Win32::Foundation::HANDLE,
    ptr: *mut u8,
}

impl MappedPage {
    fn new(size: usize) -> Self {
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                None,
                PAGE_READWRITE,
                (size as u64 >> 32) as u32,
                size as u32,
                None,
            )
        }
        .expect("CreateFileMappingW");
        let view = unsafe { MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size) };
        assert!(!view.Value.is_null());
        Self {
            section,
            ptr: view.Value.cast(),
        }
    }

    fn as_slice_mut(&mut self, len: usize) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl Drop for MappedPage {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let view = windows::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.ptr.cast(),
            };
            let _ = unsafe { UnmapViewOfFile(view) };
        }
        if !self.section.is_invalid() {
            let _ = unsafe { CloseHandle(self.section) };
        }
    }
}

#[test]
fn diff_copy_chunks_preserves_untouched_gpu_bytes() {
    let mut guest = vec![0u8; 8192];
    let mut baseline = vec![0u8; 8192];
    let mut shadow = vec![0u8; 8192];

    // Page 0 [0..4096] stays clean on CPU while holding GPU writes.
    // Page 1 [4096..8192] sees CPU writes in chunk 2 [4224..4288], keeping chunk 1 [4160..4224] GPU writes.
    shadow[64..128].fill(0x99);
    shadow[4160..4224].fill(0xAA);
    guest[4224..4288].fill(0x55);

    let copied = diff_copy_chunks(&guest, &mut baseline, &mut shadow);
    assert_eq!(copied, 64);
    assert!(shadow[0..64].iter().all(|&b| b == 0));
    assert!(shadow[64..128].iter().all(|&b| b == 0x99));
    assert!(shadow[4160..4224].iter().all(|&b| b == 0xAA));
    assert!(shadow[4224..4288].iter().all(|&b| b == 0x55));
    assert!(baseline[4224..4288].iter().all(|&b| b == 0x55));
}

#[test]
fn contains_mapped_pages_and_readability_distinguish_mappings() {
    let mapped = MappedPage::new(65536);
    assert!(contains_mapped_pages(mapped.ptr as u64, 65536));
    assert!(is_guest_readable(mapped.ptr as u64, 65536));

    let private = unsafe { VirtualAlloc(None, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
    assert!(!private.is_null());
    assert!(!contains_mapped_pages(private as u64, 65536));
    assert!(is_guest_readable(private as u64, 65536));
    let _ = unsafe { VirtualFree(private, 0, MEM_RELEASE) };

    let reserved = unsafe { VirtualAlloc(None, 65536, MEM_RESERVE, PAGE_READWRITE) };
    assert!(!reserved.is_null());
    assert!(!contains_mapped_pages(reserved as u64, 65536));
    assert!(!is_guest_readable(reserved as u64, 65536));
    let _ = unsafe { VirtualFree(reserved, 0, MEM_RELEASE) };
}

#[test]
fn shadow_pages_sync_cpu_writes_and_stop_on_pin_retirement() {
    let mut page = MappedPage::new(65536);
    for (i, byte) in page.as_slice_mut(128).iter_mut().enumerate() {
        *byte = (i as u8).wrapping_add(1);
    }

    let pin = GPU_APERTURE.pin(page.ptr as u64, 65536).expect("pin");
    let shadow = ShadowPages::new(page.ptr as u64, 65536, pin.live_flag()).expect("shadow");
    register(&shadow);
    assert!(SHADOW_COUNT.load(Ordering::Acquire) >= 1);

    let shadow_slice = unsafe { core::slice::from_raw_parts_mut(shadow.shadow_base as *mut u8, 65536) };
    assert_eq!(shadow_slice[0], 1);
    assert_eq!(shadow_slice[127], 128);

    // Simulate GPU write in chunk 1 [64..128] and CPU write in chunk 0 [0..64].
    shadow_slice[64..128].fill(0xBB);
    page.as_slice_mut(64).fill(0x33);

    sync_shadow_imports();
    assert!(shadow_slice[0..64].iter().all(|&b| b == 0x33));
    assert!(shadow_slice[64..128].iter().all(|&b| b == 0xBB));

    // Once the pin is retired (remapped), guest writes must not overwrite pre-remap data.
    GPU_APERTURE.retire_pin_for_test(pin.key());
    page.as_slice_mut(64).fill(0x77);
    sync_shadow_imports();
    assert!(shadow_slice[0..64].iter().all(|&b| b == 0x33));
    assert!(shadow.is_retired());
}
