//! Console buffer strides by guest descriptor slot.
//!
//! A console buffer descriptor keeps its element stride in word 1. A
//! translated program that loads a buffer descriptor from the resource heap
//! and then indexes it (IDXEN) reads that slot's stride from this table
//! instead: root SRV t1, space32766, one u32 per guest slot. Guest slots never
//! move, so the table follows view creation and descriptor copies only.
use super::Arena;
use crate::device::blas::upload_buffer;
use windows::Win32::Foundation::E_POINTER;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Result;

#[derive(Default)]
pub(super) struct Strides {
    /// Empty until the first slot with a stride.
    values: Vec<u32>,
    upload: Option<Upload>,
}

struct Upload {
    native: ID3D12Resource,
    address: usize,
}

impl Upload {
    fn new(device: &ID3D12Device, values: &[u32], count: usize) -> Result<Self> {
        let native = upload_buffer(device, count.max(1) as u64 * 4)?;
        let mut data = std::ptr::null_mut();
        unsafe { native.Map(0, Some(&D3D12_RANGE::default()), Some(&mut data))? };
        if data.is_null() {
            return Err(E_POINTER.into());
        }
        let upload = Self { native, address: data.addr() };
        for slot in 0..count {
            upload.set(slot, values.get(slot).copied().unwrap_or(0));
        }
        Ok(upload)
    }

    fn set(&self, slot: usize, value: u32) {
        // Writes are serialized by the arena's stride lock. The buffer stays
        // mapped for its lifetime; the CPU never reads it back.
        unsafe { (self.address as *mut u32).add(slot).write_volatile(value) };
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        unsafe { self.native.Unmap(0, None) };
    }
}

impl Arena {
    /// Record the strides of consecutive guest slots starting at `address`.
    pub(crate) fn set_strides(&self, address: usize, values: &[u32]) {
        // Guest addresses step by the guest descriptor size (see `guest_stride`).
        let guest_stride = self.guest_stride();
        let Some(offset) = address.checked_sub(self.base()) else { return };
        if guest_stride == 0 || !offset.is_multiple_of(guest_stride) {
            return;
        }
        let first = offset / guest_stride;
        let count = self.count as usize;
        if first >= count {
            return;
        }
        let mut strides = self.strides.lock().unwrap_or_else(|e| e.into_inner());
        let Strides { values: table, upload } = &mut *strides;
        if table.is_empty() {
            if values.iter().all(|&v| v == 0) {
                return;
            }
            if table.try_reserve_exact(count).is_err() {
                return;
            }
            table.resize(count, 0);
        }
        for (slot, &value) in (first..count).zip(values) {
            table[slot] = value;
            if let Some(upload) = upload {
                upload.set(slot, value);
            }
        }
    }

    /// The GPU table of strides, created on first use.
    pub(crate) fn stride_table(&self) -> Result<ID3D12Resource> {
        let device = &self.native.as_ref().ok_or(E_POINTER)?.device;
        let mut strides = self.strides.lock().unwrap_or_else(|e| e.into_inner());
        if strides.upload.is_none() {
            let upload = Upload::new(device, &strides.values, self.count as usize)?;
            strides.upload = Some(upload);
        }
        Ok(strides.upload.as_ref().expect("stride table").native.clone())
    }
}
