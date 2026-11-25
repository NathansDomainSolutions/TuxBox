use super::*;
use crate::rdna2::disassemble;

fn native(a: u32, b: u32) -> Insn {
    let bytes: Vec<u8> = [a, b].iter().flat_map(|v| v.to_le_bytes()).collect();
    disassemble(&bytes).remove(0)
}

#[test]
fn ordinary_32_bit_forms_accept_vgprs_without_modifiers() {
    // LLVM gfx1030 words for `v7, v4, v5[, v6]`.
    for (first, second, name, want) in [
        (0xd77f_0007u32, 0x0202_0b04u32, "v_add_nc_i32", "(v4 + v5)"),
        (0xd776_0007, 0x0202_0b04, "v_sub_nc_i32", "(v4 - v5)"),
        (0xd773_0007, 0x041a_0b04, "v_mad_u32_u16", "((v4 & 0xffffu) * (v5 & 0xffffu) + v6)"),
        (0xd775_0007, 0x041a_0b04, "v_mad_i32_i16", "asint(v6)"),
    ] {
        let ins = native(first, second);
        assert_eq!(ins.mnemonic, name);
        assert!(expression(&ins).unwrap().unwrap().contains(want), "{name}");
        // Clamp only for the signed add/subtract; abs, neg and EXEC/VCC
        // data remain refused. An SGPR is ordinary data.
        assert_eq!(expression(&native(first | 0x8000, second)).is_ok(), name.contains("_nc_"), "{name} clamp");
        assert!(expression(&native(first | 0x100, second)).is_err(), "{name} abs");
        assert!(expression(&native(first, (second & !511) | 4)).unwrap().unwrap().contains("s4"), "{name} sgpr");
        assert!(expression(&native(first, (second & !511) | 126)).is_err(), "{name} exec");
        assert!(expression(&native(first, second | 1 << 29)).is_err(), "{name} neg");
        // Destination or addend half selection has no meaning here.
        assert!(expression(&native(first | 0x4000, second)).is_err(), "{name} dst sel");
    }
    // op_sel:[1,1,0,0] selects the high multiplicand halves.
    let expr = expression(&native(0xd773_1807, 0x041a_0b04)).unwrap().unwrap();
    assert_eq!(expr, "((v4 >> 16) * (v5 >> 16) + v6)");
    assert!(expression(&native(0xd77f_0807, 0x0202_0b04)).is_err());
}

#[test]
fn unsigned_mad_accepts_a_full_width_inline_integer_accumulator() {
    // LLVM gfx1030: v_mad_u32_u16 v4, v1, v2, 7 (then -1).
    for (sources, addend) in [(0x021e_0501, "0x7u"), (0x0306_0501, "0xffffffffu")] {
        let expr = expression(&native(0xd773_0004, sources)).unwrap().unwrap();
        assert_eq!(expr, format!("((v1 & 0xffffu) * (v2 & 0xffffu) + {addend})"));
    }
    // v_mad_u32_u16 v4, 7, v2, v3 and v_mad_i32_i16 v4, v1, v2, 7: a 16-bit
    // multiplicand takes the constant's low half; the addend is full width.
    let expr = expression(&native(0xd773_0004, 0x040e_0487)).unwrap().unwrap();
    assert_eq!(expr, "((0x7u & 0xffffu) * (v2 & 0xffffu) + v3)");
    let expr = expression(&native(0xd775_0004, 0x021e_0501)).unwrap().unwrap();
    assert!(expr.ends_with("+ asint(0x7u))"), "{expr}");
}

#[test]
fn integer_packs_clamp_each_word_before_combining_halves() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;

    // LLVM gfx1030: v_cvt_pk_{u16_u32,i16_i32} v4, v1, v2
    for (opcode, low, high) in [
        (0xd76a_0004, "min(v1, 65535u)", "min(v2, 65535u)"),
        (0xd76b_0004, "(asuint(clamp(asint(v1), -32768, 32767)) & 0xffffu)",
                      "asuint(clamp(asint(v2), -32768, 32767))"),
    ] {
        let body = [opcode, 0x0002_0501];
        let hlsl = translate(Stage::Compute, &body).unwrap();
        assert!(hlsl.contains(&format!("({low} | ({high} << 16))")), "{hlsl}");
        assert!(translate(Stage::Compute, &[opcode | 0x0800, body[1]]).is_err());
        assert!(translate(Stage::Compute, &[opcode, body[1] | (1 << 29)]).is_err());
    }
}

#[test]
fn signed_add_and_subtract_accept_clamp() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;

    // LLVM gfx1030: v_add_nc_i32 / v_sub_nc_i32 v4, v1, v2 clamp
    for (opcode, result) in [(0xd77f_8004, "(v1 + v2)"), (0xd776_8004, "(v1 - v2)")] {
        let hlsl = translate(Stage::Compute, &[opcode, 0x0002_0501]).unwrap();
        assert!(hlsl.contains(&format!("? (0x7fffffffu + (v1 >> 31)) : {result}")), "{hlsl}");
        assert!(translate(Stage::Compute, &[opcode | 0x0100, 0x0002_0501]).is_err());
    }
}

#[test]
fn signed_add_and_subtract_accept_immediate_operands() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;

    // LLVM gfx1030: v_add_nc_i32 v7, v4, -1
    //              v_sub_nc_i32 v8, 0x80000000, v7
    let hlsl = translate(
        Stage::Compute,
        &[0xd77f_0007, 0x0001_8304, 0xd776_0008, 0x0002_0eff, 0x8000_0000],
    )
    .unwrap();
    assert!(hlsl.contains("v4 + 0xffffffffu"), "{hlsl}");
    assert!(hlsl.contains("0x80000000u - v7"), "{hlsl}");
}

#[test]
fn byte_sad_accepts_integer_constants() {
    // LLVM gfx1030: v_sad_u8 v4, v1, v2, 0
    let expr = expression(&native(0xd55a_0004, 0x0202_0501)).unwrap().unwrap();
    assert_eq!(expr.matches("max(").count(), 4);
    assert!(expr.ends_with("+ 0x0u)"), "{expr}");

    // Any integer constant is ordinary data in every SAD operand.
    for (opcode, sources, tail) in [
        (0xd55a_0004, 0x0206_0501, "+ 0x1u)"), // v_sad_u8 v4, v1, v2, 1
        (0xd55a_0004, 0x040e_0480, "+ v3)"),   // v_sad_u8 v4, 0, v2, v3
        (0xd55b_0004, 0x0202_0501, "+ 0x0u)"), // v_sad_hi_u8 v4, v1, v2, 0
    ] {
        let expr = expression(&native(opcode, sources)).unwrap().unwrap();
        assert!(expr.ends_with(tail), "{expr}");
    }
}

#[test]
fn sums_of_absolute_differences_accumulate_into_source_2() {
    // LLVM gfx1030 words for `v7, v4, v5, v6`.
    for (first, name, fields) in [
        (0xd55a_0007u32, "v_sad_u8", 4), (0xd55b_0007, "v_sad_hi_u8", 4),
        (0xd55c_0007, "v_sad_u16", 2), (0xd55d_0007, "v_sad_u32", 1),
    ] {
        let ins = native(first, 0x041a_0b04);
        assert_eq!(ins.mnemonic, name);
        let expr = expression(&ins).unwrap().unwrap();
        assert_eq!(expr.matches("max(").count(), fields, "{name}: {expr}");
        assert!(expr.ends_with("+ v6)"), "{name}: {expr}");
        assert_eq!(name == "v_sad_hi_u8", expr.contains("<< 16"), "{name}");
        assert!(expression(&native(first | 0x8000, 0x041a_0b04)).is_err(), "{name} clamp");
        assert!(expression(&native(first | 0x0800, 0x041a_0b04)).is_err(), "{name} op_sel");
        assert!(expression(&native(first, 0x041a_0b04 & !511 | 4)).unwrap().unwrap().contains("s4"), "{name} sgpr");
        assert!(expression(&native(first, 0x041a_0b04 & !511 | 106)).is_err(), "{name} vcc");
    }
    // Masked and quad SADs remain refused.
    assert!(!handles(&native(0xd571_0007, 0x041a_0b04)));
}

#[test]
fn signed_64_bit_multiply_add_needs_a_null_carry() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;
    // v_mad_i64_i32 v[10:11], null, v4, v5, v[8:9]
    let hlsl = translate(Stage::Compute, &[0xd577_7d0a, 0x0422_0b04]).unwrap();
    assert!(hlsl.contains("tb_mulhi_i32(v4, v5) + v9"), "{hlsl}");
    assert!(hlsl.contains("v10 = t") && hlsl.contains("v11 = t"), "{hlsl}");
    // A real carry destination (vcc_lo) is not modelled for the signed form.
    assert!(translate(Stage::Compute, &[0xd577_6a0a, 0x0422_0b04]).is_err());
    // The unsigned form now also accepts the null carry LLVM emits.
    let hlsl = translate(Stage::Compute, &[0xd576_7d0a, 0x0422_0b04]).unwrap();
    assert!(hlsl.contains("tb_mulhi_u32(v4, v5) + v9"), "{hlsl}");
}

#[test]
fn scalar_constant_and_literal_sources() {
    use crate::hlsl::float64::tests::translate;
    use crate::Stage;
    let ok = |words: &[u32], want: &str| {
        let hlsl = translate(Stage::Compute, words).unwrap();
        assert!(hlsl.contains(want), "{words:x?}: {hlsl}");
    };
    // LLVM gfx1030: v_sad_u8 v5, s1, v2, v3
    ok(&[0xd55a_0005, 0x040e_0401], "((s1 >> 24u) & 0xffu)");
    // LLVM gfx1030: v_sad_u8 v5, v1, -1, v3
    ok(&[0xd55a_0005, 0x040d_8301], "((0xffffffffu >> 8u) & 0xffu)");
    // LLVM gfx1030: v_sad_u8 v5, v1, v2, 0.5 (a 32-bit operand: f32 bits)
    ok(&[0xd55a_0005, 0x03c2_0501], "+ 0x3f000000u)");
    // LLVM gfx1030: v_sad_hi_u8 v5, m0, v2, 0x12345678
    ok(&[0xd55b_0005, 0x03fe_047c, 0x1234_5678], "<< 16) + 0x12345678u)");
    // LLVM gfx1030: v_sad_u16 v5, v1, -16, s3
    ok(&[0xd55c_0005, 0x000d_a101], "((0xfffffff0u >> 16u) & 0xffffu))) + s3)");
    // LLVM gfx1030: v_sad_u32 v5, v1, 4.0, s3
    ok(&[0xd55d_0005, 0x000d_ed01], "v5 = ((max(v1, 0x40800000u) - min(v1, 0x40800000u)) + s3);");
    // LLVM gfx1030: v_mad_u32_u16 v5, s1, v2, v3 op_sel:[1,0,0,0]
    ok(&[0xd773_0805, 0x040e_0401], "v5 = ((s1 >> 16) * (v2 & 0xffffu) + v3);");
    // LLVM gfx1030: v_mad_u32_u16 v5, 0x1234, v2, v3
    ok(&[0xd773_0005, 0x040e_04ff, 0x1234], "v5 = ((0x1234u & 0xffffu) * (v2 & 0xffffu) + v3);");
    // LLVM gfx1030: v_mad_i32_i16 v5, v1, -1, -4.0 (only the addend is 32-bit)
    ok(&[0xd775_0005, 0x03dd_8301], "(asint((0xffffffffu & 0xffffu) << 16) >> 16) + asint(0xc0800000u))");
    // LLVM gfx1030: v_mad_i32_i16 v5, 64, s2, m0
    ok(&[0xd775_0005, 0x01f0_04c0], "(asint((s2 & 0xffffu) << 16) >> 16) + asint(m0))");
    // LLVM gfx1030: v_add_nc_i32 v1, v2, v3 clamp
    ok(&[0xd77f_8001, 0x0002_0702], "v1 = ((((~(v2 ^ v3)) & (v2 ^ (v2 + v3))) >> 31) != 0u ? (0x7fffffffu + (v2 >> 31)) : (v2 + v3));");
    // LLVM gfx1030: v_sub_nc_i32 v1, v2, s3 clamp
    ok(&[0xd776_8001, 0x0000_0702], "v1 = ((((v2 ^ s3) & (v2 ^ (v2 - s3))) >> 31) != 0u ? (0x7fffffffu + (v2 >> 31)) : (v2 - s3));");
    let refused = |words: &[u32], why: &str| {
        let err = translate(Stage::Compute, words).unwrap_err();
        assert!(err.contains(why), "{words:x?}: {err}");
    };
    // LLVM gfx1030: v_mad_u32_u16 v5, 0x1234, v2, v3 op_sel:[1,0,0,0]
    refused(&[0xd773_0805, 0x040e_04ff, 0x1234], "high-half selection");
    // LLVM gfx10 disassembler: v_mad_i32_i16 v5, 0x3800, v2, v3 (inline 0.5)
    refused(&[0xd775_0005, 0x040e_04f0], "inline float constant in a 16-bit integer operand");
    // LLVM gfx10 disassembler: v_sad_u8 v5, exec_lo, v2, v3 and vcc_hi
    refused(&[0xd55a_0005, 0x040e_047e], "source or encoding");
    refused(&[0xd55a_0005, 0x040e_046b], "source or encoding");
    // LLVM gfx10 disassembler: v_sad_u8 v5, ttmp11, v2, v3
    assert!(translate(Stage::Compute, &[0xd55a_0005, 0x040e_0477]).is_err());
    // LLVM gfx10 disassembler: v_mad_u32_u16 v5, v1, v2, v3 clamp / op_sel:[0,0,1,0]
    refused(&[0xd773_8005, 0x040e_0501], "modifiers");
    refused(&[0xd773_2005, 0x040e_0501], "modifiers");
}
