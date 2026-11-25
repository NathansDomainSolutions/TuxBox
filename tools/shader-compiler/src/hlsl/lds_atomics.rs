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

#[cfg(test)]
mod tests {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;

    fn words(bytes: [u8; 8]) -> [u32; 2] {
        [u32::from_le_bytes(bytes[..4].try_into().unwrap()), u32::from_le_bytes(bytes[4..].try_into().unwrap())]
    }

    fn compute(bytes: [u8; 8]) -> Result<String, String> {
        translate(Stage::Compute, &words(bytes))
    }

    /// Renames the loop's temporary to t0, so expectations don't depend on
    /// how many temporaries the prologue used.
    fn renumber(hlsl: &str) -> String {
        let marker = "if (exec_lo & 1u) { uint t";
        let at = hlsl.find(marker).expect(hlsl) + marker.len();
        let n: String = hlsl[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
        let name = format!("t{n}");
        let mut out = String::new();
        let mut rest = hlsl;
        while let Some(i) = rest.find(&name) {
            let next = rest[i + name.len()..].chars().next();
            let prev = rest[..i].chars().last();
            out.push_str(&rest[..i]);
            if next.is_some_and(|c| c.is_ascii_digit()) || prev.is_some_and(|c| c.is_ascii_alphanumeric()) {
                out.push_str(&name);
            } else {
                out.push_str("t0");
            }
            rest = &rest[i + name.len()..];
        }
        out.push_str(rest);
        out
    }

    #[test]
    fn subtract_uses_a_negated_add() {
        // LLVM gfx1030: ds_sub_u32 v1, v2 offset:4
        let hlsl = renumber(&compute([0x04, 0x00, 0x04, 0xd8, 0x01, 0x02, 0x00, 0x00]).unwrap());
        assert!(hlsl.contains("uint t0a = ((v1 + 0x4u) >> 2) & 8191u;"), "{hlsl}");
        assert!(hlsl.contains("uint t0d = v2;") && hlsl.contains("InterlockedAdd(tb_lds[t0a], 0u - (t0d), t0); }"), "{hlsl}");
    }

    #[test]
    fn compare_store_takes_the_compare_value_first() {
        // ds_cmpst_rtn_b32 v8, v1, v2, v3 offset:12
        let hlsl = renumber(&compute([0x0c, 0x00, 0xc0, 0xd8, 0x01, 0x02, 0x03, 0x08]).unwrap());
        assert!(hlsl.contains("uint t0a = ((v1 + 0xcu) >> 2) & 8191u;"), "{hlsl}");
        assert!(hlsl.contains("uint t0d = v2;") && hlsl.contains("uint t0e = v3;"), "{hlsl}");
        assert!(hlsl.contains("InterlockedCompareExchange(tb_lds[t0a], t0d, t0e, t0); }"), "{hlsl}");
        assert!(hlsl.contains("v8 = t0;"), "{hlsl}");
        // ds_cmpst_b32 v1, v2, v3 (no return)
        let hlsl = renumber(&compute([0x00, 0x00, 0x40, 0xd8, 0x01, 0x02, 0x03, 0x00]).unwrap());
        assert!(hlsl.contains("InterlockedCompareExchange(tb_lds[t0a], t0d, t0e, t0); }"), "{hlsl}");
        assert!(!hlsl.contains("v8 = t0"), "{hlsl}");
    }

    #[test]
    fn loop_forms_apply_the_documented_update() {
        let cases: [([u8; 8], &str, bool); 9] = [
            // ds_rsub_rtn_u32 v8, v1, v2
            ([0x00, 0x00, 0x88, 0xd8, 0x01, 0x02, 0x00, 0x08], "_n = (t0d - t0s);", true),
            // ds_inc_u32 v1, v2
            ([0x00, 0x00, 0x0c, 0xd8, 0x01, 0x02, 0x00, 0x00], "_n = ((t0s >= t0d) ? 0u : t0s + 1u);", false),
            // ds_dec_rtn_u32 v8, v1, v2 offset:8
            ([0x08, 0x00, 0x90, 0xd8, 0x01, 0x02, 0x00, 0x08], "_n = ((t0s == 0u || t0s > t0d) ? t0d : t0s - 1u);", true),
            // ds_min_i32 v1, v2
            ([0x00, 0x00, 0x14, 0xd8, 0x01, 0x02, 0x00, 0x00], "_n = asuint(min(asint(t0s), asint(t0d)));", false),
            // ds_max_rtn_i32 v8, v1, v2
            ([0x00, 0x00, 0x98, 0xd8, 0x01, 0x02, 0x00, 0x08], "_n = asuint(max(asint(t0s), asint(t0d)));", true),
            // ds_mskor_rtn_b32 v8, v1, v2, v3
            ([0x00, 0x00, 0xb0, 0xd8, 0x01, 0x02, 0x03, 0x08], "_n = ((t0s & ~t0d) | t0e);", true),
            // ds_add_rtn_f32 v8, v1, v2
            ([0x00, 0x00, 0x54, 0xd9, 0x01, 0x02, 0x00, 0x08], "_n = asuint(asfloat(t0s) + asfloat(t0d));", true),
            // ds_min_f32 v1, v2
            ([0x00, 0x00, 0x48, 0xd8, 0x01, 0x02, 0x00, 0x00], "_n = asuint(min(asfloat(t0s), asfloat(t0d)));", false),
            // ds_max_rtn_f32 v8, v1, v2
            ([0x00, 0x00, 0xcc, 0xd8, 0x01, 0x02, 0x00, 0x08], "_n = asuint(max(asfloat(t0s), asfloat(t0d)));", true),
        ];
        for (bytes, update, ret) in cases {
            let hlsl = renumber(&compute(bytes).unwrap());
            assert!(hlsl.contains(update), "{update}\n{hlsl}");
            assert!(hlsl.contains("InterlockedCompareExchange(tb_lds[t0a], t0s, t0_n, t0);"), "{hlsl}");
            assert!(hlsl.contains("} while (t0 != t0s);"), "{hlsl}");
            assert_eq!(hlsl.contains("v8 = t0"), ret, "{hlsl}");
        }
    }

    #[test]
    fn data_is_read_before_the_destination_is_written() {
        // ds_max_rtn_i32 v2, v1, v2: the destination aliases data0.
        let w = [0x00, 0x00, 0x98, 0xd8, 0x01, 0x02, 0x00, 0x02];
        let hlsl = compute(w).unwrap();
        let read = hlsl.find("d = v2;").expect(&hlsl);
        let write = hlsl.find("v2 = t").expect(&hlsl);
        assert!(read < write, "{hlsl}");
    }

    #[test]
    fn global_data_share_forms_are_refused() {
        // ds_sub_u32 v1, v2 gds
        let err = compute([0x00, 0x00, 0x06, 0xd8, 0x01, 0x02, 0x00, 0x00]).unwrap_err();
        assert!(err.contains("GDS"), "{err}");
    }

    #[test]
    fn global_data_share_is_refused_by_the_native_forms_too() {
        let cases: [[u8; 8]; 4] = [
            // LLVM gfx1030: ds_add_u32 v1, v2 gds
            [0x00, 0x00, 0x02, 0xd8, 0x01, 0x02, 0x00, 0x00],
            // LLVM gfx1030: ds_write_b32 v1, v2 gds
            [0x00, 0x00, 0x36, 0xd8, 0x01, 0x02, 0x00, 0x00],
            // LLVM gfx1030: ds_read_u8 v5, v1 gds
            [0x00, 0x00, 0xea, 0xd8, 0x01, 0x00, 0x00, 0x05],
            // LLVM gfx1030: ds_read_b32 v5, v1 gds
            [0x00, 0x00, 0xda, 0xd8, 0x01, 0x00, 0x00, 0x05],
        ];
        for bytes in cases {
            let err = compute(bytes).unwrap_err();
            assert!(err.contains("GDS"), "{err}");
        }
    }
}
