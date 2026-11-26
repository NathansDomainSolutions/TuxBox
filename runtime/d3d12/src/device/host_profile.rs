//! What the host graphics driver can do, found once per device and logged.
//!
//! Lowerings and fallbacks choose from these capabilities, not from a vendor
//! name: the same card behaves differently under another driver or system.
//! `TUXBOX_HOST_PROFILE` replaces what the driver reports, so one machine can
//! exercise a path meant for another, for example `wave=32` or
//! `wave=32-64;section-import=substituted`.
mod section_probe;

use super::Device;
use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
};

const VARIABLE: &str = "TUXBOX_HOST_PROFILE";

/// Whether the GPU reads the guest's own pages after importing them. Xbox memory
/// is made of section views, which some drivers cannot import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SectionImport {
    /// The GPU read what the CPU wrote.
    Aliased,
    /// The import succeeded but the GPU read other memory.
    Substituted,
    /// The driver refused the import.
    Refused,
    /// The probe could not run, with the host's error code.
    Unknown(i32),
}

impl SectionImport {
    fn name(self) -> String {
        match self {
            Self::Aliased => "aliased".into(),
            Self::Substituted => "substituted".into(),
            Self::Refused => "refused".into(),
            Self::Unknown(code) => format!("unknown({code:#010x})"),
        }
    }
}

/// Capabilities given in `TUXBOX_HOST_PROFILE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Overrides {
    pub(crate) wave: Option<(u32, u32)>,
    pub(crate) section_import: Option<SectionImport>,
}

fn lane_count(text: &str) -> Option<u32> {
    let count: u32 = text.trim().parse().ok()?;
    (count.is_power_of_two() && (4..=128).contains(&count)).then_some(count)
}

/// Parse `key=value` settings separated by `;`. One bad setting rejects them
/// all, so a typing mistake cannot silently test the wrong path.
pub(crate) fn parse_overrides(text: &str) -> Result<Overrides, String> {
    let mut overrides = Overrides::default();
    for setting in text.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let bad = || format!("unusable setting {setting:?}");
        let (key, value) = setting.split_once('=').ok_or_else(bad)?;
        match (key.trim(), value.trim()) {
            ("wave", value) => {
                let (low, high) = value.split_once('-').unwrap_or((value, value));
                let range = (lane_count(low).ok_or_else(bad)?, lane_count(high).ok_or_else(bad)?);
                if range.0 > range.1 {
                    return Err(bad());
                }
                overrides.wave = Some(range);
            }
            ("section-import", "aliased") => overrides.section_import = Some(SectionImport::Aliased),
            ("section-import", "substituted") => overrides.section_import = Some(SectionImport::Substituted),
            ("section-import", "refused") => overrides.section_import = Some(SectionImport::Refused),
            _ => return Err(bad()),
        }
    }
    Ok(overrides)
}

fn overrides() -> Overrides {
    static PARSED: OnceLock<Overrides> = OnceLock::new();
    *PARSED.get_or_init(|| {
        let Ok(text) = std::env::var(VARIABLE) else { return Overrides::default() };
        parse_overrides(&text).unwrap_or_else(|reason| {
            crate::diagnostics::log(&format!("{VARIABLE} ignored: {reason}"));
            Overrides::default()
        })
    })
}

/// Per-device answers, each found when first needed.
#[derive(Default)]
pub(crate) struct Probes {
    wave: OnceLock<Option<(u32, u32)>>,
    section: OnceLock<SectionImport>,
    reported: OnceLock<()>,
}

/// The wave lane range `(min, max)` to choose shader paths by.
pub(super) fn wave_range(device: &Device) -> Option<(u32, u32)> {
    *device.state.host.wave.get_or_init(|| overrides().wave.or_else(|| host_wave_range(device)))
}

fn host_wave_range(device: &Device) -> Option<(u32, u32)> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    let ok = unsafe {
        device.native.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .is_ok();
    (ok && options.WaveOps.as_bool()
        && options.WaveLaneCountMin != 0
        && options.WaveLaneCountMax >= options.WaveLaneCountMin)
        .then_some((options.WaveLaneCountMin, options.WaveLaneCountMax))
}

/// Whether the GPU reads imported game memory. The first caller waits for the
/// probe, which runs one small GPU copy.
pub(super) fn section_import(device: &Device) -> SectionImport {
    *device.state.host.section.get_or_init(|| {
        overrides().section_import.unwrap_or_else(|| section_probe::run(&device.native))
    })
}

/// Whether imports of game memory need a copy the GPU can read. A probe that
/// could not run keeps direct imports.
pub(crate) fn needs_mirror(section: SectionImport) -> bool {
    matches!(section, SectionImport::Substituted | SectionImport::Refused)
}

/// The one line that describes the host, with overridden values marked.
pub(crate) fn describe(
    adapter: Option<(u32, u32)>,
    wave: Option<(u32, u32)>,
    section: SectionImport,
    descriptors: (u32, u32),
    overrides: Overrides,
) -> String {
    let adapter = adapter.map_or("unknown".into(), |(vendor, device)| format!("{vendor:04x}:{device:04x}"));
    let mark = |set: bool| if set { " (set)" } else { "" };
    let wave_text = wave.map_or("none".into(), |(low, high)| format!("{low}-{high}"));
    format!(
        "host profile: adapter={adapter} wave={wave_text}{} section-import={}{} descriptor-bytes={}/{}",
        mark(overrides.wave.is_some()),
        section.name(),
        mark(overrides.section_import.is_some()),
        descriptors.0,
        descriptors.1,
    )
}

/// Log the profile once per device.
pub(super) fn report(device: &Device) {
    device.state.host.reported.get_or_init(|| {
        if !crate::diagnostics::logging() {
            return;
        }
        let overrides = overrides();
        let section = section_import(device);
        let adapter = super::dxgi::adapter_ids(device);
        // SAFETY: both are valid descriptor heap types on a live device.
        let descriptors = unsafe {
            (
                device.native.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                device.native.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER),
            )
        };
        crate::diagnostics::log(&describe(adapter, wave_range(device), section, descriptors, overrides));
    });
}

#[cfg(test)]
#[path = "../tests/device/host_profile.rs"]
mod tests;
