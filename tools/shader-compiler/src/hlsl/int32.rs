//! Ordinary VOP3 integer forms with a 32-bit result: the signed no-carry
//! add/subtract, the 16x16+32 multiply-add and the sums of absolute
//! differences (SAD) over bytes, halves and words. Sources may be VGPRs,
//! SGPRs, M0, inline constants or a literal; EXEC and VCC are not read as
//! data. Clamp is modelled only where LLVM 18 pins it (the signed add and
//! subtract); other modifiers are refused.
use super::{int16::{half, sixteen_bit_source}, r, Enc, Insn, Opnd};

pub(super) fn handles(ins: &Insn) -> bool {
    ins.enc == Enc::Vop3 && matches!(ins.mnemonic.as_str(),
        "v_add_nc_i32" | "v_sub_nc_i32" | "v_mad_u32_u16" | "v_mad_i32_i16"
        | "v_sad_u8" | "v_sad_hi_u8" | "v_sad_u16" | "v_sad_u32")
}

pub(super) fn expression(ins: &Insn) -> Result<Option<String>, String> {
    if !handles(ins) {
        return Ok(None);
    }
    let mad = ins.mnemonic.starts_with("v_mad_");
    let arity = if mad || ins.mnemonic.starts_with("v_sad_") { 3 } else { 2 };
    if !ins.is_complete() || ins.raw.len() < 2 || ins.src.len() < arity {
        return Err("unsupported ordinary 32-bit integer source or encoding".into());
    }
    let (w0, w1) = (ins.raw[0], ins.raw[1]);
    for (i, source) in ins.src[..arity].iter().enumerate() {
        if mad && i < 2 {
            // VOP_I32_I16_I16_I32: the multiplicands are 16-bit operands.
            sixteen_bit_source(source, (w0 >> (11 + i)) & 1 != 0)?;
        } else if !matches!(source, Opnd::Vgpr(_) | Opnd::Sgpr(_) | Opnd::M0
            | Opnd::ConstInt(_) | Opnd::ConstF32(_) | Opnd::Literal(_)) {
            // A 32-bit integer operand reads an inline float as its f32 bit
            // pattern (LLVM 18 getInlineImmVal32, isInlinableLiteral32).
            return Err("unsupported ordinary 32-bit integer source or encoding".into());
        }
    }
    // LLVM 18 selects saddsat/ssubsat on i32 as V_ADD_I32/V_SUB_I32 with
    // clamp (VOP3Instructions.td, VOPBinOpClampPat).
    let clamped = w0 & 0x8000 != 0;
    let saturating = matches!(ins.mnemonic.as_str(), "v_add_nc_i32" | "v_sub_nc_i32");
    // RDNA2 ISA 6.4: op_sel bits 11 and 12 pick the 16-bit half of the two
    // multiplicands. The 32-bit addend and result take no half selection.
    let allowed_select = if mad { 0x1800 } else { 0 };
    if (clamped && !saturating) || w0 & 0x0700 != 0 || w0 & 0x7800 & !allowed_select != 0 || w1 >> 27 != 0 {
        return Err("unsupported ordinary 32-bit integer modifiers".into());
    }
    let (a, b) = (r(&ins.src[0])?, r(&ins.src[1])?);
    // |x - y| of unsigned fields, then the 32-bit accumulator S2 (modular).
    let sad = |width: u32| -> String {
        let mask = (1u64 << width) - 1;
        (0..32 / width).map(|i| {
            let x = format!("(({a} >> {}u) & {mask:#x}u)", i * width);
            let y = format!("(({b} >> {}u) & {mask:#x}u)", i * width);
            format!("(max({x}, {y}) - min({x}, {y}))")
        }).collect::<Vec<_>>().join(" + ")
    };
    Ok(Some(match ins.mnemonic.as_str() {
        // RDNA2 ISA: SAD_U8 sums four byte differences, SAD_HI_U8 places
        // that sum in the high half, SAD_U16 sums two half differences.
        "v_sad_u8" => format!("({} + {})", sad(8), r(&ins.src[2])?),
        "v_sad_hi_u8" => format!("((({}) << 16) + {})", sad(8), r(&ins.src[2])?),
        "v_sad_u16" => format!("({} + {})", sad(16), r(&ins.src[2])?),
        "v_sad_u32" => format!("((max({a}, {b}) - min({a}, {b})) + {})", r(&ins.src[2])?),
        // Without clamp these wrap exactly like the unsigned forms. With
        // clamp an overflowing result saturates towards the sign of S0:
        // 0x7fffffff + (S0 >> 31) is INT_MAX or INT_MIN.
        "v_add_nc_i32" if clamped => format!(
            "((((~({a} ^ {b})) & ({a} ^ ({a} + {b}))) >> 31) != 0u ? (0x7fffffffu + ({a} >> 31)) : ({a} + {b}))"),
        "v_sub_nc_i32" if clamped => format!(
            "(((({a} ^ {b}) & ({a} ^ ({a} - {b}))) >> 31) != 0u ? (0x7fffffffu + ({a} >> 31)) : ({a} - {b}))"),
        "v_add_nc_i32" => format!("({a} + {b})"),
        "v_sub_nc_i32" => format!("({a} - {b})"),
        _ => {
            let x = half(&ins.src[0], (w0 >> 11) & 1)?;
            let y = half(&ins.src[1], (w0 >> 12) & 1)?;
            let c = r(&ins.src[2])?;
            if ins.mnemonic == "v_mad_u32_u16" {
                format!("({x} * {y} + {c})")
            } else {
                let s = |v: &str| format!("(asint({v} << 16) >> 16)");
                format!("asuint({} * {} + asint({c}))", s(&x), s(&y))
            }
        }
    }))
}

#[cfg(test)]
mod tests {
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
}
