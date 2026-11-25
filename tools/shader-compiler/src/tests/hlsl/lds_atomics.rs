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
    renumber_at(hlsl, "if (exec_lo & 1u) { uint t")
}

/// Like `renumber`, for the temporary whose number follows `marker`.
fn renumber_at(hlsl: &str, marker: &str) -> String {
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
fn paired_exchanges_swap_two_dwords() {
    // LLVM gfx1030: ds_wrxchg2_rtn_b32 v[8:9], v1, v2, v3 offset0:1 offset1:7
    let hlsl = renumber(&compute([0x01, 0x07, 0xb8, 0xd8, 0x01, 0x02, 0x03, 0x08]).unwrap());
    assert!(hlsl.contains("if (exec_lo & 1u) { uint t0p = v1; uint t0d = v2; uint t0e = v3;"), "{hlsl}");
    assert!(hlsl.contains("InterlockedExchange(tb_lds[((t0p + 0x4u) >> 2) & 8191u], t0d, t0a);"), "{hlsl}");
    assert!(hlsl.contains("InterlockedExchange(tb_lds[((t0p + 0x1cu) >> 2) & 8191u], t0e, t0b); }"), "{hlsl}");
    assert!(hlsl.contains("v8 = t0a;") && hlsl.contains("v9 = t0b;"), "{hlsl}");
    // LLVM gfx1030: ds_wrxchg2st64_rtn_b32 v[8:9], v1, v2, v3 offset0:2 offset1:255
    let hlsl = compute([0x02, 0xff, 0xbc, 0xd8, 0x01, 0x02, 0x03, 0x08]).unwrap();
    assert!(hlsl.contains("p + 0x200u) >> 2) & 8191u]") && hlsl.contains("p + 0xff00u) >> 2) & 8191u]"), "{hlsl}");
}

#[test]
fn paired_exchange_reads_operands_before_writing_the_destination() {
    // LLVM gfx1030: ds_wrxchg2_rtn_b32 v[2:3], v2, v3, v2 offset1:1
    let hlsl = compute([0x00, 0x01, 0xb8, 0xd8, 0x02, 0x03, 0x02, 0x02]).unwrap();
    let read = hlsl.find("p = v2; uint t").expect(&hlsl);
    assert!(read < hlsl.find("v2 = t").expect(&hlsl) && read < hlsl.find("v3 = t").expect(&hlsl), "{hlsl}");
}

#[test]
fn paired_exchanges_of_one_dword_are_refused() {
    // LLVM gfx1030: ds_wrxchg2_rtn_b32 v[8:9], v1, v2, v3 offset0:5 offset1:5
    let err = compute([0x05, 0x05, 0xb8, 0xd8, 0x01, 0x02, 0x03, 0x08]).unwrap_err();
    assert!(err.contains("one dword twice"), "{err}");
    // LLVM gfx1030: ds_wrxchg2st64_rtn_b32 v[8:9], v1, v2, v3 offset0:1 offset1:129
    // (64 * 128 dwords apart: the same dword after the 8192-dword wrap)
    let err = compute([0x01, 0x81, 0xbc, 0xd8, 0x01, 0x02, 0x03, 0x08]).unwrap_err();
    assert!(err.contains("one dword twice"), "{err}");
    // LLVM gfx1030: ds_wrxchg2_rtn_b32 v[8:9], v1, v2, v3 offset1:1 gds
    let err = compute([0x00, 0x01, 0xba, 0xd8, 0x01, 0x02, 0x03, 0x08]).unwrap_err();
    assert!(err.contains("GDS"), "{err}");
}

#[test]
fn paired_exchange_keeps_the_branch_convergence_rule() {
    let [a, b] = words([0x01, 0x07, 0xb8, 0xd8, 0x01, 0x02, 0x03, 0x08]);
    // LLVM gfx1030: s_cbranch_scc0 0, then ds_wrxchg2_rtn_b32 v[8:9], v1, v2, v3 offset0:1 offset1:7
    let err = translate(Stage::Compute, &[0xbf84_0000, a, b]).unwrap_err();
    assert!(err.contains("branch convergence"), "{err}");
}

#[test]
fn unpinned_and_wide_forms_are_refused_with_a_reason() {
    let cases: [([u8; 8], &str); 6] = [
        // LLVM gfx1030: ds_add_u64 v1, v[2:3]
        ([0x00, 0x00, 0x00, 0xd9, 0x01, 0x02, 0x00, 0x00], "64-bit LDS atomics"),
        // LLVM gfx1030: ds_cmpst_rtn_b64 v[4:5], v1, v[2:3], v[6:7]
        ([0x00, 0x00, 0xc0, 0xd9, 0x01, 0x02, 0x06, 0x04], "64-bit LDS atomics"),
        // LLVM gfx1030: ds_wrxchg2_rtn_b64 v[4:7], v1, v[2:3], v[8:9] offset1:1
        ([0x00, 0x01, 0xb8, 0xd9, 0x01, 0x02, 0x08, 0x04], "64-bit LDS atomics"),
        // LLVM gfx1030: ds_cmpst_f32 v1, v2, v3
        ([0x00, 0x00, 0x44, 0xd8, 0x01, 0x02, 0x03, 0x00], "not pinned"),
        // LLVM gfx1030: ds_cmpst_rtn_f32 v4, v1, v2, v3
        ([0x00, 0x00, 0xc4, 0xd8, 0x01, 0x02, 0x03, 0x04], "not pinned"),
        // LLVM gfx1030: ds_permute_b32 v4, v1, v2
        ([0x00, 0x00, 0xc8, 0xda, 0x01, 0x02, 0x00, 0x04], "forward permute"),
    ];
    for (bytes, reason) in cases {
        let err = compute(bytes).unwrap_err();
        assert!(err.contains(reason), "{reason}: {err}");
    }
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

#[test]
fn byte_loads_fill_one_half_of_the_destination() {
    let cases: [([u8; 8], &str); 4] = [
        // LLVM gfx1030: ds_read_u8_d16 v5, v1 offset:3
        ([0x03, 0x00, 0x88, 0xda, 0x01, 0x00, 0x00, 0x05], "v5 = ((v5 & 0xffff0000u) | ((tb_lds[(t0a >> 2) & 8191u] >> t0s) & 0xffu));"),
        // LLVM gfx1030: ds_read_u8_d16_hi v5, v1
        ([0x00, 0x00, 0x8c, 0xda, 0x01, 0x00, 0x00, 0x05], "v5 = ((v5 & 0xffffu) | (((tb_lds[(t0a >> 2) & 8191u] >> t0s) & 0xffu) << 16));"),
        // LLVM gfx1030: ds_read_i8_d16 v5, v1 offset:1
        ([0x01, 0x00, 0x90, 0xda, 0x01, 0x00, 0x00, 0x05],
         "v5 = ((v5 & 0xffff0000u) | (asuint(asint(((tb_lds[(t0a >> 2) & 8191u] >> t0s) & 0xffu) << 24) >> 24) & 0xffffu));"),
        // LLVM gfx1030: ds_read_i8_d16_hi v5, v1 offset:2
        ([0x02, 0x00, 0x94, 0xda, 0x01, 0x00, 0x00, 0x05],
         "v5 = ((v5 & 0xffffu) | (asuint(asint(((tb_lds[(t0a >> 2) & 8191u] >> t0s) & 0xffu) << 24) >> 24) << 16));"),
    ];
    for (bytes, expect) in cases {
        let hlsl = renumber_at(&compute(bytes).unwrap(), &format!("(v1) + {:#x}u; uint t", bytes[0]));
        assert!(hlsl.contains(expect), "{expect}\n{hlsl}");
    }
    // LLVM gfx1030: ds_write_b8_d16_hi v1, v2 offset:1
    let hlsl = compute([0x01, 0x00, 0x80, 0xda, 0x01, 0x02, 0x00, 0x00]).unwrap();
    assert!(hlsl.contains("((((v2 >> 16)) & 0xffu) << t"), "{hlsl}");
    assert!(hlsl.contains("a = (v1) + 0x1u;"), "{hlsl}");
}
