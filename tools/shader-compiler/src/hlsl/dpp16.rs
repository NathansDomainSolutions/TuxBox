//! DPP16 and DPP8 source fetches of ordinary VOP1/VOP2 instructions (source
//! zero selector 250, or 233/234 for DPP8). The second word holds the real
//! source VGPR and the controls. DPP16:
//!
//! ```text
//!   [7:0] src0  [16:8] dpp_ctrl  [17] reserved  [18] fi  [19] bound_ctrl
//!   [20] src0 neg  [21] src0 abs  [22] src1 neg  [23] src1 abs
//!   [27:24] bank_mask  [31:28] row_mask
//! ```
//! (LLVM VOPInstructions.td `VOP_DPPe`; dpp_ctrl values from SIDefines.h `DppCtrl`.)
//!
//! Every invocation reads the source lane outside the guest branch guards, as
//! the other DPP16 exchanges do (see `is_quad_exchange`), so the shader needs
//! complete 32-lane compute waves. At the instruction itself:
//!
//! - An *invalid* source lane is one outside the row (row_shl/row_shr), or, with
//!   fi=0, one whose EXEC bit is clear (AMDGPUModifierSyntax.rst, `fi`).
//! - bound_ctrl=1 (written `bound_ctrl:1`, or `bound_ctrl:0` before GFX11)
//!   reads zero from an invalid lane; bound_ctrl=0 disables the write instead,
//!   so the destination keeps its value (GCNDPPCombine.cpp combining rules).
//! - A lane whose row_mask bit is clear keeps its destination, whatever
//!   bound_ctrl says (IntrinsicsAMDGPU.td `update.dpp`: a move of `old` then the
//!   DPP move; AMDGPUAtomicOptimizer.cpp uses row_mask 0xa for rows 1 and 3).
//! - Source abs/neg (f32 sources only) flip or clear bit 31 of the fetched bits,
//!   after bound_ctrl's zero. The order is inferred from GCNDPPCombine folding
//!   abs/neg into the user of a bound_ctrl-zero DPP move; the sign of a zero read
//!   from an invalid lane under neg is speculative, not checked on hardware.
//!
//! bank_mask: AMDGPUModifierSyntax.rst says "the lanes of a wavefront are
//! organized in four *rows* and four *banks*" and "Each of the 4 bits in the
//! mask controls one bank (0 - disabled, 1 - enabled)", but not which lanes
//! form a bank. GCNDPPCombine.cpp treats row_mask and bank_mask alike as a lane
//! mask (`MaskAllLanes = row_mask == 0xF && bank_mask == 0xF`; the old value
//! only matters when not all lanes are masked in), as does
//! AMDGPUInstCombineIntrinsic.cpp ("If bound_ctrl = 1, row mask = bank mask =
//! 0xf we can omit old value"), so a lane in a disabled bank keeps its
//! destination. Without the lane layout only bank_mask 0xf (every lane) and 0
//! (no lane, so nothing is written) are taken; row_mask 0 also writes nothing.
//!
//! Row masks: rows are 16 lanes. The host lane index stands for the guest lane
//! index modulo the host wave width. Where the guest wave width is not known, a
//! 32-lane host wave may hold either half of a guest wave64, so only row masks
//! that read the same for rows r and r + 2 are taken. Asymmetric masks are taken
//! where host and guest lanes agree (`check_row_masks`). GFX9 wave shifts and
//! row broadcasts and modifiers on non-f32 sources are refused.
//!
//! DPP8 (VOPInstructions.td `VOP_DPP8e`: `Inst{39-32} = src0`, `Inst{63-40} =
//! dpp8`; SIDefines.h `DPP8_FI_0 = 0xE9`, `DPP8_FI_1 = 0xEA` in the src0 field)
//! has no masks, bound_ctrl or modifiers. Lane i reads lane `(i & ~7) | sel[i & 7]`
//! of its group of eight, where AMDGPUAsmParser.cpp `parseDPP8` packs
//! `DPP8 |= (Sels[i] << (i * 3))` and AMDGPUModifierSyntax.rst `dpp8_sel` says
//! "The first value selects which lane to read from to supply data into lane 0".
//! Its `fi` (AMDGPUModifierSyntax.rst): "fi:0 Fetch zero when accessing data
//! from inactive lanes. fi:1 Fetch pre-existing values from inactive lanes."
//! Every active lane writes.
use super::{r, Emitter, Enc, Insn, Opnd};

/// `aux.imm2` of a VOP1/VOP2 instruction whose source zero is a DPP16 (DPP8)
/// fetch. The plain VOP1/VOP2 encodings have no output modifier, so this field
/// is free.
const TAG: i64 = 0xd916 << 32;
const TAG8: i64 = 0xd908 << 32;

pub(super) fn marked(i: &Insn) -> bool {
    matches!(i.enc, Enc::Vop1 | Enc::Vop2) && (i.aux.imm2 == TAG || i.aux.imm2 == TAG8)
}

fn is_dpp8(i: &Insn) -> bool {
    i.aux.imm2 == TAG8
}

/// Source lane of `lane` under the DPP8 selectors (`word >> 8`).
fn dpp8_select(word: u32, lane: &str) -> String {
    format!("({lane} & ~7u) | (({:#x}u >> (3u * ({lane} & 7u))) & 7u)", word >> 8)
}

/// Whether row_mask or bank_mask disables every lane (see the module notes).
fn writes_nothing(c: &Controls) -> bool {
    c.row_mask == 0 || c.bank_mask == 0
}

struct Controls {
    src0: u8,
    ctrl: u32,
    fi: bool,
    bound_zero: bool,
    neg0: bool,
    abs0: bool,
    neg1: bool,
    abs1: bool,
    bank_mask: u32,
    row_mask: u32,
}

fn decode(word: u32) -> Controls {
    let bit = |n: u32| (word >> n) & 1 != 0;
    Controls {
        src0: word as u8,
        ctrl: (word >> 8) & 0x1ff,
        fi: bit(18),
        bound_zero: bit(19),
        neg0: bit(20),
        abs0: bit(21),
        neg1: bit(22),
        abs1: bit(23),
        bank_mask: (word >> 24) & 15,
        row_mask: word >> 28,
    }
}

/// Source lane of `lane` under `ctrl`, and the condition under which it lies
/// inside the row (`None`: always). Lane rules: AMDGPUModifierSyntax.rst
/// `dpp16_ctrl`. row_shr reads lane - n (AMDGPUAtomicOptimizer.cpp builds its
/// scans and its shift right by one from row_shr); row_ror ("rotate right")
/// wraps in the same direction, and row_shl reads lane + n. row_xmask's XOR is
/// also AMDGPUAtomicOptimizer's butterfly. Out-of-row lanes still name a lane of
/// the row, so every host read stays in range.
pub(super) fn lane_select(ctrl: u32, lane: &str) -> Result<(String, Option<String>), String> {
    let n = ctrl & 15;
    Ok(match ctrl {
        0x000..=0x0ff => (format!("({lane} & ~3u) | (({ctrl:#x}u >> (2u * ({lane} & 3u))) & 3u)"), None),
        0x101..=0x10f => (
            format!("({lane} & ~15u) | (({lane} + {n}u) & 15u)"),
            Some(format!("({lane} & 15u) < {}u", 16 - n)),
        ),
        0x111..=0x11f => (
            format!("({lane} & ~15u) | (({lane} - {n}u) & 15u)"),
            Some(format!("({lane} & 15u) >= {n}u")),
        ),
        0x121..=0x12f => (format!("({lane} & ~15u) | (({lane} - {n}u) & 15u)"), None),
        0x140 => (format!("{lane} ^ 15u"), None),
        0x141 => (format!("{lane} ^ 7u"), None),
        0x150..=0x15f => (format!("({lane} & ~15u) | {n}u"), None),
        0x160..=0x16f => (format!("{lane} ^ {n}u"), None),
        0x100 | 0x110 | 0x120 => return Err(format!("reserved DPP16 shift by zero {ctrl:#x}")),
        0x130..=0x13f => return Err(format!("GFX9 wave shift DPP control {ctrl:#x}")),
        0x142 | 0x143 => return Err(format!("GFX9 row broadcast DPP control {ctrl:#x}")),
        _ => return Err(format!("reserved DPP16 control {ctrl:#x}")),
    })
}

fn f32_source(mnemonic: &str) -> bool {
    mnemonic.ends_with("_f32") && mnemonic != "v_ldexp_f32"
}

/// Check each remaining VOP1/VOP2 DPP16 or DPP8 instruction, name its source
/// VGPR and mark it. Runs after the wave-reduction lowerings have claimed their
/// shapes. Asymmetric row masks are checked later, by `check_row_masks`.
pub(super) fn prepare(insns: &mut [Insn]) -> Result<(), String> {
    let mut exited = false;
    for ins in insns.iter_mut() {
        if ins.mnemonic == "s_endpgm" {
            exited = true;
        }
        let marker = match ins.src.first() {
            Some(&Opnd::Raw(m @ (233 | 234 | 250))) if matches!(ins.enc, Enc::Vop1 | Enc::Vop2) => m,
            _ => continue,
        };
        let dpp8 = marker != 250;
        let at = ins.offset;
        let m = ins.mnemonic.as_str();
        let kind = if dpp8 { "DPP8" } else { "DPP16" };
        let bad = |what: String| format!("{kind} {m} at {at:#x}: {what}");
        let word = match ins.raw[..] {
            [first, word] if first & 0x1ff == u32::from(marker) => word,
            _ => return Err(bad(format!("missing {kind} control word"))),
        };
        if exited {
            return Err(bad("source fetch following an early exit".into()));
        }
        let c = decode(word);
        if !dpp8 {
            lane_select(c.ctrl, "lane").map_err(bad)?;
            if word & (1 << 17) != 0 {
                return Err(bad("reserved control bit 17".into()));
            }
            if !matches!(c.bank_mask, 0 | 0xf) && c.row_mask != 0 {
                return Err(bad(format!("unmodelled bank_mask {:#x}", c.bank_mask)));
            }
        }
        let lane_local = matches!(ins.dst[..], [Opnd::Vgpr(_)]) && match ins.enc {
            Enc::Vop1 => ins.src.len() == 1,
            _ => matches!(ins.src[..], [_, Opnd::Vgpr(_)]),
        };
        let unsupported = m.contains("64") || m.contains("_co_") || m.starts_with("v_movrel") || m.starts_with("v_swap")
            || m.ends_with("ak_f32") || m.ends_with("mk_f32") || m.ends_with("ak_f16") || m.ends_with("mk_f16")
            || matches!(m, "v_readfirstlane_b32" | "v_nop" | "v_pipeflush" | "v_clrexcp");
        if !lane_local || unsupported {
            return Err(bad("not a lane-local VGPR operation".into()));
        }
        if !dpp8 && (c.neg0 || c.abs0) && !f32_source(m) {
            return Err(bad("abs/neg on a source that is not f32".into()));
        }
        if !dpp8 && (c.neg1 || c.abs1) && !(ins.enc == Enc::Vop2 && f32_source(m)) {
            return Err(bad("abs/neg on a second source that is not f32".into()));
        }
        ins.src[0] = Opnd::Vgpr(c.src0);
        ins.aux.imm2 = if dpp8 { TAG8 } else { TAG };
    }
    Ok(())
}

/// Refuse DPP16 row masks that depend on which guest lanes a host wave holds,
/// unless `lanes_known`: the host lane index is the guest lane index (the host
/// wave width equals a known guest wave width, or a 32-thread group fills lanes
/// 0-31 of a single guest wave of either width). Rows the guest wave lacks hold
/// no lane, so their mask bits do not matter.
pub(super) fn check_row_masks(insns: &[Insn], lanes_known: bool) -> Result<(), String> {
    for ins in insns.iter().filter(|i| marked(i) && !is_dpp8(i)) {
        let c = decode(ins.raw[1]);
        if !lanes_known && !writes_nothing(&c) && c.row_mask & 3 != c.row_mask >> 2 {
            return Err(format!("DPP16 {} at {:#x}: row_mask {:#x} depends on the guest wave width",
                ins.mnemonic, ins.offset, c.row_mask));
        }
    }
    Ok(())
}

/// Read the source lane, outside every guard. `skipped` is the condition
/// under which this invocation branches over the instruction (`None`: never);
/// such a lane counts as inactive for fi=0.
pub(super) fn fetch(em: &mut Emitter, i: usize, ins: &Insn, skipped: Option<&str>) -> Result<(), String> {
    let c = decode(ins.raw[1]);
    let (src, in_row) = if is_dpp8(ins) {
        // DPP8 fi lives in the source zero selector: 234 fetches inactive lanes.
        (dpp8_select(ins.raw[1], &format!("q{i}lane")), None)
    } else {
        lane_select(c.ctrl, &format!("q{i}lane"))?
    };
    let fi = if is_dpp8(ins) { ins.raw[0] & 0x1ff == 234 } else { c.fi };
    em.touch(&ins.src[0]);
    let source = r(&ins.src[0])?;
    em.line(&format!("uint q{i}lane = WaveGetLaneIndex(), q{i}src = {src};"));
    let active = (!fi).then(|| {
        let exec = skipped.map_or("exec_lo".into(), |s| format!("({s} ? 0u : exec_lo)"));
        format!("(WaveReadLaneAt({exec}, q{i}src) & 1u) != 0u")
    });
    let valid: Vec<String> = in_row.into_iter().chain(active).collect();
    if !valid.is_empty() {
        em.line(&format!("bool q{i}ok = {};", valid.join(" && ")));
    }
    em.line(&format!("uint q{i} = WaveReadLaneAt({source}, q{i}src);"));
    em.dpp16_fetched.insert(i);
    Ok(())
}

/// Apply bound_ctrl, the modifiers and the row mask to the fetched value, then
/// translate the instruction with it as an ordinary source.
pub(super) fn emit(em: &mut Emitter, idx: usize, ins: &Insn, is_ps: bool, br: Option<usize>) -> Result<(), String> {
    if em.dispatched {
        return Err("DPP16 source fetch in a block-dispatched program".into());
    }
    if !em.dpp16_fetched.contains(&idx) {
        return Err("DPP16 source fetch outside the convergent emitter".into());
    }
    let mut plain = ins.clone();
    plain.aux.imm2 = 0;
    plain.dwords = 1;
    if is_dpp8(ins) {
        // fi=0 (233) fetches zero from an inactive lane; every active lane writes.
        let src0 = ins.raw[1] & 0xff;
        let value = if ins.raw[0] & 0x1ff == 233 { format!("(q{idx}ok ? q{idx} : 0u)") } else { format!("q{idx}") };
        plain.raw = vec![(ins.raw[0] & !0x1ff) | (256 + src0)];
        let t = em.tmp;
        em.tmp += 1;
        em.line(&format!("uint tb_s{t} = {value};"));
        plain.src[0] = Opnd::Temp(t as u16);
        return em.emit_insn_body(idx, &plain, is_ps, br);
    }
    let c = decode(ins.raw[1]);
    if writes_nothing(&c) {
        em.line(&format!("// DPP16 row_mask {:#x} bank_mask {:#x}: no lane writes", c.row_mask, c.bank_mask));
        return Ok(());
    }
    let always_valid = c.fi && lane_select(c.ctrl, "lane")?.1.is_none();
    let mut value = if c.bound_zero && !always_valid { format!("(q{idx}ok ? q{idx} : 0u)") } else { format!("q{idx}") };
    if c.abs0 {
        value = format!("({value} & 0x7fffffffu)");
    }
    if c.neg0 {
        value = format!("({value} ^ 0x80000000u)");
    }
    plain.raw = vec![(ins.raw[0] & !0x1ff) | (256 + u32::from(c.src0))];
    let mut sources = vec![(0, value)];
    if c.abs1 || c.neg1 {
        let mut bits = r(&ins.src[1])?;
        if c.abs1 {
            bits = format!("({bits} & 0x7fffffffu)");
        }
        if c.neg1 {
            bits = format!("({bits} ^ 0x80000000u)");
        }
        sources.push((1, bits));
    }
    for (i, bits) in sources {
        let t = em.tmp;
        em.tmp += 1;
        em.line(&format!("uint tb_s{t} = {bits};"));
        plain.src[i] = Opnd::Temp(t as u16);
    }
    let mut enabled = Vec::new();
    if c.row_mask != 0xf {
        // Equal mask bits for rows r and r + 2, so a 32- or 64-lane host wave reads
        // alike, or host lanes that are the guest lanes (`check_row_masks`).
        enabled.push(format!("(({:#x}u >> (q{idx}lane >> 4u)) & 1u) != 0u", c.row_mask));
    }
    if !c.bound_zero && !always_valid {
        enabled.push(format!("q{idx}ok"));
    }
    if enabled.is_empty() {
        return em.emit_insn_body(idx, &plain, is_ps, br);
    }
    em.line(&format!("if ({}) {{", enabled.join(" && ")));
    let result = em.emit_insn_body(idx, &plain, is_ps, br);
    em.line("}");
    result
}

#[cfg(test)]
#[path = "../tests/hlsl/dpp16.rs"]
mod tests;
