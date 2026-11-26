//! A saved EXEC is represented by one bit per host invocation. WQM on
//! that value must exchange bits with the other three pixels of the quad.
//!
//! Boundaries: this module admits the straight-prefix optimization only when
//! decoded instructions and raw DPP16 fields prove converged axis moves.
use super::{Emitter, Enc, Insn, Opnd, writes_scalar, is_quad_exchange, OP_QUAD_MOVE};

/// Return the entry WQM index for converged FI=1 axis swaps in a straight prefix.
/// Raw DPP16 fields and lowered operands must agree before bypassing wave lane reads.
/// After the final swap, branches may target only instruction boundaries beyond that
/// swap (or the program end); no tail backedge may re-enter the converged prefix.
pub(super) fn straight_prefix(code: &[Insn]) -> Option<usize> {
    let last = code.iter().rposition(is_quad_exchange)?;
    let mut entry = None;
    for (at, ins) in code.iter().enumerate() {
        let m = ins.mnemonic.as_str();
        if is_quad_exchange(ins) && !straight_axis_move(ins) {
            return None;
        }
        // Only resolved relative scalar branches have a known destination.
        let branch = ins.enc == Enc::Sopp && Emitter::branch_cond(ins.op).is_some();
        if m.contains("branch") && !branch
            || matches!(m, "s_setpc_b64" | "s_swappc_b64" | "s_rfe_b64" | "s_call_b64") {
            return None;
        }
        if branch {
            let target = ins.offset as i64 + 4 + 4 * ins.aux.imm;
            let end = code.last().map(|i| (i.offset + i.dwords * 4) as i64)?;
            if at <= last || target <= code[last].offset as i64
                || !(target == end || code.binary_search_by_key(&target, |i| i.offset as i64).is_ok()) {
                return None;
            }
        }
        if at > last { continue; }
        if m.starts_with("s_endpgm") || matches!(m, "s_sendmsghalt" | "s_trap" | "s_sethalt")
            || m.starts_with("v_cmpx") || m.contains("_saveexec_") || m.contains("_wrexec_") {
            return None;
        }
        let writes_exec = ins.dst.iter().any(|o| matches!(o, Opnd::ExecLo | Opnd::ExecHi));
        if writes_exec {
            if entry.is_some() || m != "s_wqm_b64" || ins.dst != [Opnd::ExecLo]
                || ins.src != [Opnd::ExecLo] {
                return None;
            }
            entry = Some(at);
        }
        // Entry setup may save EXEC or load scalar constants, but vector work
        // must see the whole quad from its first instruction onward.
        if entry.is_none() && !(m.starts_with("s_") || m == "v_nop") {
            return None;
        }
    }
    entry
}

/// Admit only native VOP1 DPP16 axis moves with FI=1 and every destination enabled.
/// The raw opcode, registers and selector must match their lowered representation.
fn straight_axis_move(ins: &Insn) -> bool {
    let [word, control] = ins.raw.as_slice() else { return false };
    ins.enc == Enc::Vop1 && ins.op == OP_QUAD_MOVE && ins.mnemonic == "v_mov_b32"
        && ins.dwords == 2 && word & 0xfe01_ffff == 0x7e00_02fa
        && control >> 24 == 0xff && control & (1 << 18) != 0
        // Bit 17 is reserved; source modifiers are outside this integer-move proof.
        && control & 0x00f2_0000 == 0 && ins.aux.sdwa.is_none()
        && matches!(ins.aux.imm, 0xb1 | 0x4e)
        && ins.aux.imm == i64::from((control >> 8) & 0x1ff)
        && ins.src == [Opnd::Vgpr(*control as u8)]
        && ins.dst == [Opnd::Vgpr((word >> 17) as u8)]
}

pub(super) fn saved_exec(code: &[Insn], at: usize) -> bool {
    let op = &code[at];
    let [Opnd::Sgpr(low)] = op.src.as_slice() else {
        return false;
    };
    if op.mnemonic != "s_wqm_b64" || *low >= 127 {
        return false;
    }
    let pair = Opnd::Sgpr(*low);
    let mut saved = false;
    for ins in &code[..at] {
        if !writes_scalar(ins, *low) && !writes_scalar(ins, *low + 1) {
            continue;
        }
        if ins.dst != [pair] {
            return false;
        }
        match ins.mnemonic.as_str() {
            "s_mov_b64" if ins.src == [Opnd::ExecLo] => saved = true,
            "s_andn2_b64" if saved && ins.src == [pair, Opnd::ExecLo] => {}
            _ => return false,
        }
    }
    // A loop back to this instruction could carry a later register definition.
    saved
        && !code[at..].iter().any(|ins| {
            ins.enc == Enc::Sopp
                && Emitter::branch_cond(ins.op).is_some()
                && ins.offset as i64 + 4 + 4 * ins.aux.imm <= op.offset as i64
        })
}

pub(super) fn guards_expansion(code: &[Insn], at: usize) -> bool {
    let Some([branch, wqm]) = code.get(at + 1..at + 3) else {
        return false;
    };
    branch.mnemonic == "s_cbranch_scc0"
        && branch.aux.imm > 0
        && code[at].dst == wqm.src
        && saved_exec(code, at + 2)
}

pub(super) fn vote(id: usize, mask: &str) -> String {
    // Use bitwise OR so all four quad operations execute even where the
    // current lane's bit is already set.
    format!(
        "uint qb{id} = {mask} & 1u; uint q{id} = qb{id} | QuadReadAcrossX(qb{id}) | QuadReadAcrossY(qb{id}) | QuadReadAcrossDiagonal(qb{id});"
    )
}

#[cfg(test)]
#[path = "../tests/hlsl/pixel_quad.rs"]
mod tests;
