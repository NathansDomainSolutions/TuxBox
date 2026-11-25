//! VOP3 operates on one selected half of a register. VOP3P computes both
//! halves independently, with separate source selectors for each result.
use super::{r, Enc, Insn, Opnd};

enum Operation {
    Add,
    Subtract,
    Multiply,
    MultiplyAdd,
    UnsignedMax,
    UnsignedMin,
    ShiftLeft,
    ShiftRight,
    ArithmeticShiftRight,
    SignedMax,
    SignedMin,
    UnsignedMin3,
    UnsignedMax3,
    UnsignedMed3,
    SignedMin3,
    SignedMax3,
    SignedMed3,
}

pub(super) fn half(src: &Opnd, select: u32) -> Result<String, String> {
    let value = r(src)?;
    Ok(if select == 0 {
        format!("({value} & 0xffffu)")
    } else {
        format!("({value} >> 16)")
    })
}

/// Sources of the ordinary (non-packed) 16-bit forms. A scalar register
/// holds 32 bits like a VGPR and either half can be selected. Integer inline
/// constants and literals are 32-bit values whose low half is the 16-bit
/// operand; selecting their high half is not modelled. EXEC and VCC hold
/// only this lane's bit in the translator, not the wave mask, so they are
/// not read as integer data.
///
/// Inline float constants stay refused: LLVM 18 does not agree with itself
/// on their value for a 16-bit integer operand. The disassembler prints the
/// half-precision pattern (getInlineImmVal16: 0.5 -> 0x3800), while
/// SIInstrInfo::isInlineConstant never uses them for OPERAND_REG_*_INT16
/// because they "appear to not work" and "seem" to read the low 16 bits of
/// the 32-bit pattern (bugzilla 46302), and the encoder (getLit16IntEncoding)
/// emits such values as literals.
pub(super) fn sixteen_bit_source(source: &Opnd, high: bool) -> Result<(), String> {
    match source {
        Opnd::Vgpr(_) | Opnd::Sgpr(_) | Opnd::M0 => Ok(()),
        Opnd::ConstInt(_) | Opnd::Literal(_) if !high => Ok(()),
        Opnd::ConstInt(_) | Opnd::Literal(_) => Err("high-half selection of a 16-bit integer constant".into()),
        Opnd::ConstF32(_) => Err("inline float constant in a 16-bit integer operand".into()),
        _ => Err("unsupported ordinary 16-bit integer source or encoding".into()),
    }
}

pub(super) fn new_half(ins: &Insn) -> bool {
    ins.enc == Enc::Vop3 && matches!(ins.mnemonic.as_str(),
        "v_mul_lo_u16" | "v_max_u16" | "v_min_u16" | "v_add_nc_i16" | "v_sub_nc_i16"
        | "v_mad_u16" | "v_mad_i16" | "v_min3_u16" | "v_max3_u16" | "v_med3_u16"
        | "v_min3_i16" | "v_max3_i16" | "v_med3_i16")
}

pub(super) fn expression(ins: &Insn) -> Result<Option<String>, String> {
    if ins.enc != Enc::Vop3 {
        return Ok(None);
    }
    let op = match ins.mnemonic.as_str() {
        // Without clamp the signed and unsigned forms have the same low 16 bits.
        "v_add_nc_u16" | "v_add_nc_i16" | "v_pk_add_u16" | "v_pk_add_i16" => Operation::Add,
        "v_sub_nc_u16" | "v_sub_nc_i16" | "v_pk_sub_i16" | "v_pk_sub_u16" => Operation::Subtract,
        "v_mul_lo_u16" | "v_pk_mul_lo_u16" => Operation::Multiply,
        "v_mad_u16" | "v_mad_i16" | "v_pk_mad_i16" | "v_pk_mad_u16" => Operation::MultiplyAdd,
        "v_max_u16" | "v_pk_max_u16" => Operation::UnsignedMax,
        "v_min_u16" | "v_pk_min_u16" => Operation::UnsignedMin,
        "v_lshlrev_b16" | "v_pk_lshlrev_b16" => Operation::ShiftLeft,
        "v_lshrrev_b16" | "v_pk_lshrrev_b16" => Operation::ShiftRight,
        "v_ashrrev_i16" | "v_pk_ashrrev_i16" => Operation::ArithmeticShiftRight,
        "v_max_i16" | "v_pk_max_i16" => Operation::SignedMax,
        "v_min_i16" | "v_pk_min_i16" => Operation::SignedMin,
        "v_min3_u16" => Operation::UnsignedMin3,
        "v_max3_u16" => Operation::UnsignedMax3,
        "v_med3_u16" => Operation::UnsignedMed3,
        "v_min3_i16" => Operation::SignedMin3,
        "v_max3_i16" => Operation::SignedMax3,
        "v_med3_i16" => Operation::SignedMed3,
        _ => return Ok(None),
    };
    let packed = ins.op >= 0x400;
    let new_packed = matches!(ins.mnemonic.as_str(),
        "v_pk_add_i16" | "v_pk_mad_i16" | "v_pk_mad_u16" | "v_pk_max_u16" | "v_pk_min_u16");
    let arity = if matches!(op, Operation::MultiplyAdd | Operation::UnsignedMin3 | Operation::UnsignedMax3
        | Operation::UnsignedMed3 | Operation::SignedMin3 | Operation::SignedMax3 | Operation::SignedMed3) { 3 } else { 2 };
    if !ins.is_complete() || ins.raw.len() < 2 || ins.src.len() < arity {
        return Err("incomplete 16-bit integer instruction".into());
    }
    // Do not interpret an EXEC/VCC selector as packed integer data.
    if new_packed && (!packed || !ins.src[..arity].iter().all(|s| matches!(s, Opnd::Vgpr(_)))) {
        return Err("unsupported packed integer source or encoding".into());
    }
    let (w0, w1) = (ins.raw[0], ins.raw[1]);
    if new_half(ins) {
        if packed {
            return Err("unsupported ordinary 16-bit integer source or encoding".into());
        }
        for (i, source) in ins.src[..arity].iter().enumerate() {
            sixteen_bit_source(source, (w0 >> (11 + i)) & 1 != 0)?;
        }
        // A two-source form has no source 2 selector (LLVM prints only
        // op_sel:[src0, src1, dst]).
        if arity == 2 && w0 & 0x2000 != 0 {
            return Err("unsupported ordinary 16-bit integer modifiers".into());
        }
    }
    // RDNA2 ISA sections 6.5 and 12.10: clamp arithmetic to the
    // representable signed/unsigned 16-bit interval, independently per half.
    let clamped = w0 & 0x8000 != 0;
    let saturating_arithmetic = packed && matches!(op,
        Operation::Add | Operation::Subtract | Operation::Multiply | Operation::MultiplyAdd);
    // LLVM 18 selects saddsat/ssubsat on i16 as V_ADD_I16/V_SUB_I16 with
    // clamp (VOP3Instructions.td, OpSelBinOpClampPat): signed saturation of
    // the selected halves. The sources were checked above.
    let signed_saturating_half = !packed
        && matches!(ins.mnemonic.as_str(), "v_add_nc_i16" | "v_sub_nc_i16");
    if clamped && !signed_saturating_half && (!saturating_arithmetic
        || !ins.src[..arity].iter().all(|s| matches!(s, Opnd::Vgpr(_)))) {
        return Err("unsupported packed integer clamp form or source".into());
    }
    if w0 & 0x0700 != 0
        || w1 >> (if packed { 29 } else { 27 }) != 0
        || ins.src[..arity].iter().any(|s| matches!(s, Opnd::ConstF32(_)))
    {
        return Err("unsupported 16-bit integer modifiers or inline float".into());
    }
    let result = |selectors: u32| -> Result<String, String> {
        let a = half(&ins.src[0], selectors & 1)?;
        let b = half(&ins.src[1], (selectors >> 1) & 1)?;
        let signed = |x: &str| format!("(asint({x} << 16) >> 16)");
        if clamped {
            let signed_op = matches!(ins.mnemonic.as_str(),
                "v_pk_add_i16" | "v_pk_sub_i16" | "v_pk_mad_i16" | "v_add_nc_i16" | "v_sub_nc_i16");
            if signed_op {
                let (a, b) = (signed(&a), signed(&b));
                let value = match op {
                    Operation::Add => format!("({a} + {b})"),
                    Operation::Subtract => format!("({a} - {b})"),
                    Operation::MultiplyAdd => {
                        let c = signed(&half(&ins.src[2], (selectors >> 2) & 1)?);
                        format!("({a} * {b} + {c})")
                    }
                    _ => unreachable!("signed clamp arithmetic"),
                };
                // Keep the saturated representation unsigned before packing. The
                // sign-biased ordering is exactly the signed ordering, but
                // avoids narrow signed intermediates in the host optimizer.
                return Ok(format!("(clamp((asuint({value}) ^ 0x80000000u), 0x7fff8000u, 0x80007fffu) ^ 0x80000000u)"));
            }
            // 65535*65535+65535 fits in uint32, and unsigned subtract
            // must clamp at zero before any modular underflow.
            let value = match op {
                Operation::Add => format!("({a} + {b})"),
                Operation::Subtract => format!("(max({a}, {b}) - {b})"),
                Operation::Multiply => format!("({a} * {b})"),
                Operation::MultiplyAdd => {
                    let c = half(&ins.src[2], (selectors >> 2) & 1)?;
                    format!("({a} * {b} + {c})")
                }
                _ => unreachable!("unsigned clamp arithmetic"),
            };
            return Ok(format!("min({value}, 65535u)"));
        }
        Ok(match op {
            Operation::Add => format!("({a} + {b})"),
            Operation::Subtract => format!("({a} - {b})"),
            Operation::Multiply => format!("({a} * {b})"),
            // Without clamp, signed and unsigned multiply-add produce the
            // same low 16 bits. The uint expression is modular even if the
            // intermediate wraps at 32 bits.
            Operation::MultiplyAdd => {
                let c = half(&ins.src[2], (selectors >> 2) & 1)?;
                format!("({a} * {b} + {c})")
            }
            Operation::UnsignedMax => format!("max({a}, {b})"),
            Operation::UnsignedMin => format!("min({a}, {b})"),
            Operation::ShiftLeft => format!("({b} << ({a} & 15u))"),
            Operation::ShiftRight => format!("({b} >> ({a} & 15u))"),
            Operation::ArithmeticShiftRight => format!("asuint({} >> ({a} & 15u))", signed(&b)),
            Operation::SignedMax => format!("asuint(max({}, {}))", signed(&a), signed(&b)),
            Operation::SignedMin => format!("asuint(min({}, {}))", signed(&a), signed(&b)),
            Operation::UnsignedMin3 | Operation::UnsignedMax3 | Operation::UnsignedMed3
            | Operation::SignedMin3 | Operation::SignedMax3 | Operation::SignedMed3 => {
                let c = half(&ins.src[2], (selectors >> 2) & 1)?;
                let signed_op = matches!(op, Operation::SignedMin3 | Operation::SignedMax3 | Operation::SignedMed3);
                let (a, b, c) = if signed_op { (signed(&a), signed(&b), signed(&c)) } else { (a, b, c) };
                let value = match op {
                    Operation::UnsignedMin3 | Operation::SignedMin3 => format!("min(min({a}, {b}), {c})"),
                    Operation::UnsignedMax3 | Operation::SignedMax3 => format!("max(max({a}, {b}), {c})"),
                    _ => format!("max(min({a}, {b}), min(max({a}, {b}), {c}))"),
                };
                if signed_op { format!("asuint({value})") } else { value }
            }
        })
    };
    let lo = result(w0 >> 11)?;
    Ok(Some(if packed {
        let hi = result(((w1 >> 27) & 3) | (((w0 >> 14) & 1) << 2))?;
        format!("(({lo} & 0xffffu) | ({hi} << 16))")
    } else {
        let dst = r(&ins.dst[0])?;
        if (w0 & (1 << 14)) != 0 {
            format!("(({dst} & 0xffffu) | ({lo} << 16))")
        } else {
            format!("(({dst} & 0xffff0000u) | ({lo} & 0xffffu))")
        }
    }))
}

#[cfg(test)]
#[path = "../tests/hlsl/int16.rs"]
mod tests;
