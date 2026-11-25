use super::*;
use crate::rdna2::disassemble;

#[test]
fn signed_half_add_and_subtract_preserve_the_untouched_half_with_immediates() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;

    // LLVM gfx1030: v_add_nc_i16 v7, v4, -1
    //              v_sub_nc_i16 v8, 0x1234, v5 op_sel:[0,1,1]
    let hlsl = translate(
        Stage::Compute,
        &[0xd70d_0007, 0x0001_8304, 0xd70e_5008, 0x0002_0aff, 0x1234],
    )
    .unwrap();
    assert!(
        hlsl.contains("(v7 & 0xffff0000u) | (((v4 & 0xffffu) + (0xffffffffu & 0xffffu)) & 0xffffu)"),
        "{hlsl}"
    );
    assert!(
        hlsl.contains("(v8 & 0xffffu) | (((0x1234u & 0xffffu) - (v5 >> 16)) << 16)"),
        "{hlsl}"
    );
    // v_add_nc_i16 v7, v4, -1 op_sel:[0,1,0]
    assert!(translate(Stage::Compute, &[0xd70d_1007, 0x0001_8304]).is_err());
}

#[test]
fn inline_half_multiply_preserves_the_high_half() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;

    // LLVM gfx1030: v_mul_lo_u16 v4, v0, 7
    //               v_mul_lo_u16 v4, -1, v0
    for (sources, product) in [
        (0x0001_0f00, "(v0 & 0xffffu) * (0x7u & 0xffffu)"),
        (0x0002_00c1, "(0xffffffffu & 0xffffu) * (v0 & 0xffffu)"),
    ] {
        let hlsl = translate(Stage::Compute, &[0xd705_0004, sources]).unwrap();
        assert!(hlsl.contains(&format!("(v4 & 0xffff0000u) | (({product}) & 0xffffu)")), "{hlsl}");
    }

    // A constant has no high half to select. A trailing literal is read like
    // an inline integer (LLVM gfx1030: v_mul_lo_u16 v4, v0, 0x1234).
    assert!(translate(Stage::Compute, &[0xd705_1004, 0x0001_0f00]).is_err());
    let hlsl = translate(Stage::Compute, &[0xd705_0004, 0x0001_ff00, 0x1234]).unwrap();
    assert!(hlsl.contains("(v4 & 0xffff0000u) | (((v0 & 0xffffu) * (0x1234u & 0xffffu)) & 0xffffu)"), "{hlsl}");
}

#[test]
fn ordinary_half_forms_preserve_the_other_half() {
    for (first, mnemonic) in [
        (0xd705_4804u32, "v_mul_lo_u16"),
        (0xd709_4804u32, "v_max_u16"),
        (0xd70b_4804u32, "v_min_u16"),
    ] {
        let decode = |a: u32, b: u32| {
            let bytes: Vec<u8> = [a,b].iter().flat_map(|v| v.to_le_bytes()).collect();
            disassemble(&bytes).remove(0)
        };
        let ins=decode(first,0x0202_0b04);
        assert_eq!(ins.mnemonic,mnemonic);
        assert!(new_half(&ins));
        let expr=expression(&ins).unwrap().unwrap();
        assert!(expr.contains("v4 & 0xffffu"));
        let expr=expression(&decode(first & !0x4000,0x0202_0b04)).unwrap().unwrap();
        assert!(expr.contains("v4 & 0xffff0000u"));
        assert!(expression(&decode(first | 0x8000,0x0202_0b04)).is_err());
        // s4 is ordinary data; exec_lo holds only this lane's bit.
        assert!(expression(&decode(first,0x0202_0a04)).unwrap().unwrap().contains("(s4 >> 16)"));
        assert!(expression(&decode(first,0x0202_0a7e)).is_err());
    }
}

#[test]
fn new_16_bit_forms_select_halves_and_preserve_the_other_half() {
    let decode = |a: u32, b: u32| {
        let bytes: Vec<u8> = [a, b].iter().flat_map(|v| v.to_le_bytes()).collect();
        disassemble(&bytes).remove(0)
    };
    // LLVM gfx1030 words for `v7, v4, v5[, v6]` without op_sel.
    for (first, second, name) in [
        (0xd70d_0007u32, 0x0202_0b04u32, "v_add_nc_i16"),
        (0xd70e_0007, 0x0202_0b04, "v_sub_nc_i16"),
        (0xd740_0007, 0x041a_0b04, "v_mad_u16"),
        (0xd75e_0007, 0x041a_0b04, "v_mad_i16"),
        (0xd752_0007, 0x041a_0b04, "v_min3_i16"),
        (0xd753_0007, 0x041a_0b04, "v_min3_u16"),
        (0xd755_0007, 0x041a_0b04, "v_max3_i16"),
        (0xd756_0007, 0x041a_0b04, "v_max3_u16"),
        (0xd758_0007, 0x041a_0b04, "v_med3_i16"),
        (0xd759_0007, 0x041a_0b04, "v_med3_u16"),
    ] {
        let ins = decode(first, second);
        assert_eq!(ins.mnemonic, name);
        assert!(new_half(&ins), "{name}");
        let low = expression(&ins).unwrap().unwrap();
        assert!(low.contains("v7 & 0xffff0000u") && low.contains("(v4 & 0xffffu)"), "{name}: {low}");
        // op_sel source 0 high and destination high.
        let high = expression(&decode(first | 0x4800, second)).unwrap().unwrap();
        assert!(high.contains("v7 & 0xffffu") && high.contains("(v4 >> 16)"), "{name}: {high}");
        if name.ends_with("_i16") && !name.contains("_nc_") && !name.starts_with("v_mad") {
            assert!(low.contains("asint("), "{name} compares signed");
        }
        if name.contains("_nc_i16") {
            let clamped = expression(&decode(first | 0x8000, second)).unwrap().unwrap();
            assert!(clamped.contains("0x7fff8000u, 0x80007fffu"), "{name}: {clamped}");
        } else {
            assert!(expression(&decode(first | 0x8000, second)).is_err(), "{name} clamp");
        }
        let scalar = expression(&decode(first, (second & !511) | 4)).unwrap().unwrap();
        assert!(scalar.contains("(s4 & 0xffffu)"), "{name}: {scalar}");
        for mask in [106, 107, 126, 127] {
            assert!(expression(&decode(first, (second & !511) | mask)).is_err(), "{name} mask {mask}");
        }
        assert!(expression(&decode(first | 0x100, second)).is_err(), "{name} abs");
    }
    // Source 2 selection for three-operand forms.
    let med = expression(&decode(0xd759_2007, 0x041a_0b04)).unwrap().unwrap();
    assert!(med.contains("(v6 >> 16)"));
}

#[test]
fn new_packed_integer_forms_accept_ordinary_sources_only() {
    // LLVM gfx1030 native words. The default high selectors include
    // source 2 via dword0 bit 14 for the three-operand forms.
    for (first, second, name) in [
        (0xcc02_4004, 0x1a02_0b04, "v_pk_add_i16"),
        (0xcc00_2804, 0x141a_0b04, "v_pk_mad_i16"),
        (0xcc09_4004, 0x1c1a_0b04, "v_pk_mad_u16"),
        (0xcc0c_4804, 0x1202_0b04, "v_pk_max_u16"),
        (0xcc0d_4004, 0x1a02_0b04, "v_pk_min_u16"),
    ] {
        let native = |a: u32, b: u32| {
            let bytes: Vec<u8> = [a,b].iter().flat_map(|v| v.to_le_bytes()).collect();
            disassemble(&bytes).remove(0)
        };
        let ins = native(first,second);
        assert_eq!(ins.mnemonic, name);
        assert!(expression(&ins).unwrap().is_some());
        if matches!(name, "v_pk_max_u16" | "v_pk_min_u16") {
            assert!(expression(&native(first | 0x8000, second)).is_err());
        } else {
            let clamped = expression(&native(first | 0x8000, second)).unwrap().unwrap();
            if name.ends_with("_i16") {
                assert!(clamped.contains("0x7fff8000u, 0x80007fffu"));
                assert!(clamped.contains("^ 0x80000000u"));
            }
        }
        assert!(expression(&native(first | 0x8000, (second & !511) | 4)).is_err());
        assert!(expression(&native(first, (second & !511) | 4)).is_err());
        assert!(expression(&native(first | 0x100, second)).is_err());
    }
    let bytes: Vec<u8> = [0xcc09_4004u32,0x1c1a_0b04].iter().flat_map(|v| v.to_le_bytes()).collect();
    let expr = expression(&disassemble(&bytes)[0]).unwrap().unwrap();
    assert!(expr.contains("(v6 & 0xffffu)"));
    assert!(expr.contains("(v6 >> 16)"));
}

#[test]
fn ordinary_16_bit_scalar_constant_and_literal_sources() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;
    let ok = |words: &[u32], want: &str| {
        let hlsl = translate(Stage::Compute, words).unwrap();
        assert!(hlsl.contains(want), "{words:x?}: {hlsl}");
    };
    // LLVM gfx1030: v_max3_i16 v5, -1, v2, v3
    ok(&[0xd755_0005, 0x040e_04c1], "max(max((asint((0xffffffffu & 0xffffu) << 16) >> 16)");
    // LLVM gfx1030: v_med3_u16 v5, v1, s2, m0 op_sel:[0,1,0,1]
    ok(&[0xd759_5005, 0x01f0_0501],
        "v5 = ((v5 & 0xffffu) | (max(min((v1 & 0xffffu), (s2 >> 16)), min(max((v1 & 0xffffu), (s2 >> 16)), (m0 & 0xffffu))) << 16));");
    // LLVM gfx1030: v_mad_u16 v5, 0x1234, v2, v3
    ok(&[0xd740_0005, 0x040e_04ff, 0x1234], "(((0x1234u & 0xffffu) * (v2 & 0xffffu) + (v3 & 0xffffu)) & 0xffffu)");
    // LLVM gfx1030: v_mul_lo_u16 v5, 0x1234, s2
    ok(&[0xd705_0005, 0x0000_04ff, 0x1234], "(((0x1234u & 0xffffu) * (s2 & 0xffffu)) & 0xffffu)");
    // LLVM gfx1030: v_add_nc_i16 v5, v1, -1 clamp
    ok(&[0xd70d_8005, 0x0001_8301], "v5 = ((v5 & 0xffff0000u) | ((clamp((asuint(((asint((v1 & 0xffffu) << 16) >> 16) + (asint((0xffffffffu & 0xffffu) << 16) >> 16))) ^ 0x80000000u), 0x7fff8000u, 0x80007fffu) ^ 0x80000000u) & 0xffffu));");
    // LLVM gfx1030: v_sub_nc_i16 v5, s1, v2 op_sel:[0,0,1] clamp
    ok(&[0xd70e_c005, 0x0002_0401], "v5 = ((v5 & 0xffffu) | ((clamp((asuint(((asint((s1 & 0xffffu) << 16) >> 16) - (asint((v2 & 0xffffu) << 16) >> 16))) ^ 0x80000000u), 0x7fff8000u, 0x80007fffu) ^ 0x80000000u) << 16));");
    let refused = |words: &[u32], why: &str| {
        let err = translate(Stage::Compute, words).unwrap_err();
        assert!(err.contains(why), "{words:x?}: {err}");
    };
    // LLVM gfx1030: v_min3_u16 v5, 1, v2, v3 op_sel:[1,0,0,0]
    refused(&[0xd753_0805, 0x040e_0481], "high-half selection");
    // LLVM gfx10 disassembler: v_max3_i16 v5, 0x3800, v2, v3 and v1, v2, 0xc400
    refused(&[0xd755_0005, 0x040e_04f0], "inline float constant in a 16-bit integer operand");
    refused(&[0xd755_0005, 0x03de_0501], "inline float constant in a 16-bit integer operand");
    // LLVM gfx10 disassembler: v_max_u16 v5, exec_hi, v2
    refused(&[0xd709_0005, 0x0002_047f], "source or encoding");
    // LLVM gfx10 disassembler: v_max3_u16 v5, v1, v2, v3 clamp; v_mad_u16 ... clamp
    refused(&[0xd756_8005, 0x040e_0501], "clamp");
    refused(&[0xd740_8005, 0x040e_0501], "clamp");
    // LLVM gfx10 disassembler: v_cmp_eq_i16_e64 s10, v1, 0x3800 (inline 0.5)
    refused(&[0xd48a_000a, 0x0001_e101], "inline float constant in a 16-bit integer compare");
}
