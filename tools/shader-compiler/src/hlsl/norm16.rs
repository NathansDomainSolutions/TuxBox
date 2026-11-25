//! Two-channel 16-bit normalized packs: `v_cvt_pknorm_u16_f32` (UNORM16) and
//! `v_cvt_pknorm_i16_f32` (SNORM16), and their half-float source forms
//! `v_cvt_pknorm_u16_f16` and `v_cvt_pknorm_i16_f16`. Source 0 fills the low
//! half and source 1 the high half. The conversions follow the D3D float-to-normalized rules:
//! UNORM clamps to [0, 1] and scales by 65535; SNORM clamps to [-1, 1] and
//! scales by 32767. Both round to nearest even, and NaN converts to 0.

/// Whether `mnemonic` is a normalized 16-bit pack, and if so whether it is signed.
pub(super) fn signed(mnemonic: &str) -> Option<bool> {
    match mnemonic {
        "v_cvt_pknorm_u16_f32" | "v_cvt_pknorm_u16_f16" => Some(false),
        "v_cvt_pknorm_i16_f32" | "v_cvt_pknorm_i16_f16" => Some(true),
        _ => None,
    }
}

/// The packed 32-bit result for float expressions `lo` and `hi`. Each is
/// evaluated once per use, so callers pass temporaries for long expressions.
pub(super) fn pack(signed: bool, lo: &str, hi: &str) -> String {
    let half = |x: &str| if signed {
        // clamp() would turn NaN into a bound, so NaN is tested on the bits.
        format!("(tb_isnan({x}) ? 0u : (asuint((int)round(clamp({x}, -1.0, 1.0) * 32767.0)) & 0xffffu))")
    } else {
        // saturate() returns 0 for NaN.
        format!("(uint)round(saturate({x}) * 65535.0)")
    };
    format!("({} | ({} << 16))", half(lo), half(hi))
}

/// The float a pixel export passes to the render target for a 16-bit `word`
/// (the bits in the low half) written by a pack. SNORM -32768 maps to -1.0.
pub(super) fn export(signed: bool, word: &str) -> String {
    if signed {
        format!("max((float)(asint(({word}) << 16) >> 16) / 32767.0, -1.0)")
    } else {
        format!("((float){word} / 65535.0)")
    }
}

#[cfg(test)]
#[path = "../tests/hlsl/norm16.rs"]
mod tests;
