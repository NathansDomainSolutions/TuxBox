//! V_PERMLANE16_B32 and V_PERMLANEX16_B32 (VOP3, RDNA2) in complete wave32
//! compute waves.
//!
//! Within each row of 16 lanes, lane i reads VSRC0 from the lane whose 4-bit
//! index is bits [4i+3:4i] of {SRC2, SRC1} (SRC1 holds lanes 0-7, SRC2 lanes
//! 8-15); permlanex16 reads from the other row of the same 32-lane pair. That
//! selector layout is the RDNA ISA pseudocode, as the established permlanex16
//! lowering already encodes it; LLVM release/18.x does not spell it out, but
//! its atomic optimizer relies on all-ones selectors reading lane 15 of the
//! other row. Pinned from LLVM: SRC1/SRC2 are scalar operands (VOP3 SSrc_b32,
//! and SIInstrInfo inserts readfirstlane for vector values). op_sel holds the
//! DPP mode bits [fi, bound_ctrl] (AMDGPUModifierSyntax "dpp_op_sel"; any other
//! op_sel bit is rejected by the assembler and refused here, including the
//! op_sel[3] bit that the specialized cull lowerings accept in one encoding).
//! For an inactive source lane:
//! fi=1 fetches its register value, otherwise bound_ctrl=1 reads zero and
//! bound_ctrl=0 leaves the destination unchanged (the tied vdst_in; LLVM's
//! InstCombine drops that old value only when fi or bound_ctrl is set).
//! Inactive destination lanes are never written.
//!
//! The plain permlanex16 form (op_sel 0) keeps the established lowering in
//! the emitter. Every other form needs the verified wave32 layout that
//! ds_bpermute_b32 also requires, with every host lane converged, and
//! selectors proven to be ordinary scalars rather than lane masks.
use super::{scalar_bits, Emitter, Enc, Insn, Opnd, r};

const NAMES: [&str; 2] = ["v_permlane16_b32", "v_permlanex16_b32"];

fn op_sel(ins: &Insn) -> u32 {
    ins.raw.first().map_or(0, |w| (w >> 11) & 0xf)
}

/// Forms handled here rather than by the established permlanex16 lowering.
pub(super) fn verified_form(ins: &Insn) -> bool {
    ins.enc == Enc::Vop3 && NAMES.contains(&ins.mnemonic.as_str())
        && !(ins.mnemonic == "v_permlanex16_b32" && op_sel(ins) == 0)
}

pub(super) fn emit(em: &mut Emitter<'_>, idx: usize, ins: &Insn) -> Result<bool, String> {
    if !verified_form(ins) { return Ok(false); }
    let m = ins.mnemonic.as_str();
    let mode = op_sel(ins);
    if mode & !3 != 0 || ins.aux.mask != 0 || ins.aux.imm != 0 || ins.aux.imm2 != 0 {
        return Err(format!("{m} modifiers unsupported"));
    }
    let (fetch_inactive, bound_zero) = (mode & 1 != 0, mode & 2 != 0);
    let ([Opnd::Vgpr(_)], [Opnd::Vgpr(_), _, _]) = (ins.dst.as_slice(), ins.src.as_slice()) else {
        return Err(format!("{m} needs a vector source and destination"));
    };
    if em.required_wave_size != Some(32) || em.portable_wave64 || em.pixel_wave
        || em.wave_cull || em.page_waves || em.fspx_wave.is_some() {
        return Err(format!("{m} requires a verified wave32 compute layout"));
    }
    if !scalar_bits::ordinary_selectors(em, idx) {
        return Err(format!("{m} needs proven ordinary scalar selectors (not a wave mask)"));
    }
    let (value, low, high) = (r(&ins.src[0])?, r(&ins.src[1])?, r(&ins.src[2])?);
    let t = em.tmp;
    em.tmp += 1;
    let row = if m == "v_permlanex16_b32" { format!("((t{t}lane & 16u) ^ 16u)") } else { format!("(t{t}lane & 16u)") };
    em.line(&format!("uint t{t}lane = WaveGetLaneIndex(); uint t{t}sel = (t{t}lane & 8u) ? {high} : {low};"));
    em.line(&format!("uint t{t}src = {row} | ((t{t}sel >> (4u * (t{t}lane & 7u))) & 15u);"));
    em.line(&format!("uint t{t}value = WaveReadLaneAt({value}, t{t}src);"));
    let d = &ins.dst[0];
    if fetch_inactive {
        return em.wv(d, &format!("t{t}value")).map(|_| true);
    }
    em.line(&format!("uint t{t}active = WaveReadLaneAt(exec_lo & 1u, t{t}src);"));
    if bound_zero {
        em.wv(d, &format!("(t{t}active != 0u) ? t{t}value : 0u"))?;
    } else {
        em.line(&format!("if (t{t}active != 0u) {{"));
        em.wv(d, &format!("t{t}value"))?;
        em.line("}");
    }
    Ok(true)
}

#[cfg(test)]
#[path = "../tests/hlsl/permlane.rs"]
mod tests;
