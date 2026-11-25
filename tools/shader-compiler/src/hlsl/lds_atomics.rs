//! Group-shared (LDS) atomics beyond the forms HLSL has an `Interlocked*`
//! intrinsic for. Each one becomes a compare-exchange loop on the containing
//! dword of `tb_lds`, so the update is applied atomically per lane like the
//! hardware operation, and the `_rtn` forms return the value seen before it.
//!
//! The update rules follow LLVM's selection patterns for gfx10 in
//! DSInstructions.td: `ds_inc_u32`/`ds_dec_u32` implement `uinc_wrap` and
//! `udec_wrap`, `ds_add_f32`/`ds_min_f32`/`ds_max_f32` implement `fadd`,
//! `fmin` and `fmax`, and `ds_cmpst_b32` takes the compare value in data0 and
//! the new value in data1 (the opposite of the buffer compare-swap order).
//!
//! The two-address exchanges (`ds_wrxchg2_rtn_b32`, `ds_wrxchg2st64_rtn_b32`) are
//! two independent `InterlockedExchange`s. 64-bit atomics, `ds_cmpst_f32` and
//! `ds_permute_b32` are refused with the reason given in [`refusal`].

/// The new dword written for `old`, given data0 `d` and (for the two-data
/// forms) data1 `e`, or `None` when the mnemonic isn't one of these forms.
pub(super) fn update(mnemonic: &str, old: &str, d: &str, e: &str) -> Option<String> {
    Some(match mnemonic.replace("_rtn", "").as_str() {
        "ds_rsub_u32" => format!("({d} - {old})"),
        "ds_inc_u32" => format!("(({old} >= {d}) ? 0u : {old} + 1u)"),
        "ds_dec_u32" => format!("(({old} == 0u || {old} > {d}) ? {d} : {old} - 1u)"),
        "ds_min_i32" => format!("asuint(min(asint({old}), asint({d})))"),
        "ds_max_i32" => format!("asuint(max(asint({old}), asint({d})))"),
        "ds_mskor_b32" => format!("(({old} & ~{d}) | {e})"),
        "ds_add_f32" => format!("asuint(asfloat({old}) + asfloat({d}))"),
        "ds_min_f32" => format!("asuint(min(asfloat({old}), asfloat({d})))"),
        "ds_max_f32" => format!("asuint(max(asfloat({old}), asfloat({d})))"),
        _ => return None,
    })
}

/// Forms with a native intrinsic that the shared atomic path doesn't cover yet.
pub(super) fn native(mnemonic: &str) -> bool {
    matches!(mnemonic.replace("_rtn", "").as_str(), "ds_sub_u32" | "ds_cmpst_b32")
}

/// True when `mnemonic` takes a second data operand.
pub(super) fn two_data(mnemonic: &str) -> bool {
    matches!(mnemonic.replace("_rtn", "").as_str(), "ds_mskor_b32" | "ds_cmpst_b32")
}

/// Byte offsets of the two dwords `ds_wrxchg2_rtn_b32` / `ds_wrxchg2st64_rtn_b32`
/// exchange, from the instruction's 16-bit offset field. These share the
/// two-address layout of `ds_write2_b32` / `ds_write2st64_b32` (LLVM's
/// `DS_1A2D_Off8_RET`: offset0 in bits 0-7, offset1 in bits 8-15), counting 4-byte
/// elements, or 64 of them (256 bytes) for st64. Each returns the dword it replaced:
/// vdst gets offset0's old value and vdst+1 offset1's.
///
/// When both offsets name the same dword of `tb_lds` (after the 8192-dword wrap the
/// translator applies to every LDS address) the result depends on which exchange the
/// hardware applies first, which isn't documented, so that case is refused.
pub(super) fn pair_offsets(mnemonic: &str, offset: u32) -> Result<(u32, u32), String> {
    let scale = if mnemonic.contains("2st64") { 64 } else { 1 };
    let (e0, e1) = ((offset & 0xff) * scale, ((offset >> 8) & 0xff) * scale);
    if e0.abs_diff(e1) % 8192 == 0 {
        return Err("two-address LDS exchange of one dword twice (order not documented)".into());
    }
    Ok((e0 * 4, e1 * 4))
}

/// Why an LDS form the translator knows about is refused, or `None` for any other
/// mnemonic.
pub(super) fn refusal(mnemonic: &str) -> Option<&'static str> {
    let base = mnemonic.replace("_rtn", "");
    Some(match base.as_str() {
        // tb_lds is `groupshared uint[]`: HLSL can't alias it as uint64_t, and two 32-bit
        // atomics would let another lane see (or update) one dword between them.
        "ds_add_u64" | "ds_sub_u64" | "ds_rsub_u64" | "ds_inc_u64" | "ds_dec_u64" | "ds_min_i64" | "ds_max_i64"
        | "ds_min_u64" | "ds_max_u64" | "ds_and_b64" | "ds_or_b64" | "ds_xor_b64" | "ds_mskor_b64" | "ds_cmpst_b64"
        | "ds_cmpst_f64" | "ds_min_f64" | "ds_max_f64" | "ds_wrxchg_b64" | "ds_wrxchg2_b64" | "ds_wrxchg2st64_b64" => {
            "64-bit LDS atomics need one atomic across two tb_lds dwords (not available on a uint groupshared array)"
        }
        // LLVM has no selection pattern for these, so nothing pins whether the compare is
        // a float compare (-0 == +0, NaN never equal) or bitwise.
        "ds_cmpst_f32" => "ds_cmpst_f32 compare semantics (float or bitwise) are not pinned",
        // LLVM's int_amdgcn_ds_permute documents neither which lane wins when several
        // write the same lane nor what an unwritten lane receives.
        "ds_permute_b32" => "forward permute conflict and unwritten-lane results are not pinned",
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../tests/hlsl/lds_atomics.rs"]
mod tests;
