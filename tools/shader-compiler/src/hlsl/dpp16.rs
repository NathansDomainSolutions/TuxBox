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
mod tests {
    use crate::{hlsl, ShaderExtract, Stage};

    const V4: [u32; 2] = [0x7e08_02ff, 0x1357_2468];
    const V5: [u32; 2] = [0x7e0a_02ff, 0x2468_1357];
    const ENDPGM: u32 = 0xbf81_0000;

    /// Translate `v_mov_b32 v4/v5, literal`, then `body`, then `s_endpgm`,
    /// as a `stage` program with a `threads` x 1 x 1 group.
    fn translate(stage: Stage, threads: u32, body: &[u32]) -> Result<String, String> {
        let words: Vec<u32> = V4.iter().chain(&V5).chain(body).chain(&[ENDPGM]).copied().collect();
        let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let ex = ShaderExtract { flat_inputs: Vec::new(), flat_outputs: Vec::new(),
            stage, shader_model: (6, 0), required_features: None, bitcode_stripped: true,
            fspx_config: None, ps09_pixel_inputs: None, fssm: Vec::new(), entry: Some("main".into()),
            isa_data: isa.clone(), isa, isa_offset_in_fssm: 0, psv_threads: None, isa_preamble: [0, threads, 1, 1],
            has_input_sig: false, has_output_sig: false, has_psv0: false, has_root_sig: false,
            input_sig: Vec::new(), output_sig: Vec::new(), resources: Vec::new(), gs_info: None,
            mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
        };
        hlsl::translate(&ex).map(|t| t.hlsl)
    }

    fn compute(body: &[u32]) -> String {
        translate(Stage::Compute, 64, body).unwrap_or_else(|e| panic!("{e}"))
    }

    fn has(hlsl: &str, lines: &[&str]) {
        for line in lines {
            assert!(hlsl.contains(line), "missing `{line}` in\n{hlsl}");
        }
    }

    #[test]
    fn quad_perm_reads_the_selected_lane_of_the_quad() {
        // LLVM gfx1030: v_bfrev_b32_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
        let hlsl = compute(&[0x7e0a_70fa, 0xff00_e401]);
        has(&hlsl, &[
            "uint q2lane = WaveGetLaneIndex(), q2src = (q2lane & ~3u) | ((0xe4u >> (2u * (q2lane & 3u))) & 3u);",
            // fi=0: an inactive source lane is invalid; bound_ctrl=0 then keeps v5.
            "bool q2ok = (WaveReadLaneAt(exec_lo, q2src) & 1u) != 0u;",
            "uint q2 = WaveReadLaneAt(v1, q2src);",
            "uint tb_s0 = q2;",
            "if (q2ok) {",
            "if (exec_lo & 1u) v5 = reversebits(tb_s0);",
        ]);
        // LLVM gfx1030: v_bfrev_b32_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf bound_ctrl:1 fi:1
        let hlsl = compute(&[0x7e0a_70fa, 0xff0c_1b01]);
        has(&hlsl, &["((0x1bu >> (2u * (q2lane & 3u))) & 3u)", "uint tb_s0 = q2;", "v5 = reversebits(tb_s0);"]);
        // fi=1 within the quad: every source is valid, so every enabled lane writes.
        assert!(!hlsl.contains("q2ok") && !hlsl.contains("if (q2"), "{hlsl}");
    }

    #[test]
    fn row_shifts_zero_or_keep_out_of_row_lanes() {
        // LLVM gfx1030: v_ceil_f32_dpp v5, -|v1| row_shr:3 row_mask:0xf bank_mask:0xf bound_ctrl:1
        let hlsl = compute(&[0x7e0a_44fa, 0xff39_1301]);
        has(&hlsl, &[
            "q2src = (q2lane & ~15u) | ((q2lane - 3u) & 15u);",
            "bool q2ok = (q2lane & 15u) >= 3u && (WaveReadLaneAt(exec_lo, q2src) & 1u) != 0u;",
            // bound_ctrl=1 reads zero from an invalid lane, then abs and neg apply.
            "uint tb_s0 = (((q2ok ? q2 : 0u) & 0x7fffffffu) ^ 0x80000000u);",
            "if (exec_lo & 1u) v5 = asuint(ceil(asfloat(tb_s0)));",
        ]);
        assert!(!hlsl.contains("if (q2ok"), "{hlsl}");
        // LLVM gfx1030: v_ceil_f32_dpp v5, |v1| row_shl:15 row_mask:0xa bank_mask:0xf
        let hlsl = compute(&[0x7e0a_44fa, 0xaf21_0f01]);
        has(&hlsl, &[
            "q2src = (q2lane & ~15u) | ((q2lane + 15u) & 15u);",
            "bool q2ok = (q2lane & 15u) < 1u && (WaveReadLaneAt(exec_lo, q2src) & 1u) != 0u;",
            "uint tb_s0 = (q2 & 0x7fffffffu);",
            // Rows 0 and 2 are masked off; invalid lanes keep v5 (bound_ctrl=0).
            "if (((0xau >> (q2lane >> 4u)) & 1u) != 0u && q2ok) {",
        ]);
        // LLVM gfx1030: v_not_b32_dpp v2, v3 row_ror:7 row_mask:0x5 bank_mask:0xf
        let hlsl = compute(&[0x7e04_6efa, 0x5f01_2703]);
        has(&hlsl, &[
            "q2src = (q2lane & ~15u) | ((q2lane - 7u) & 15u);",
            "bool q2ok = (WaveReadLaneAt(exec_lo, q2src) & 1u) != 0u;",
            "uint q2 = WaveReadLaneAt(v3, q2src);",
            "if (((0x5u >> (q2lane >> 4u)) & 1u) != 0u && q2ok) {",
            "v2 = (~tb_s0);",
        ]);
    }

    #[test]
    fn mirrors_shares_and_xor_masks() {
        for (word, src) in [
            // LLVM gfx1030: v_not_b32_dpp v2, v3 row_mirror row_mask:0xf bank_mask:0xf fi:1
            (0xff05_4003, "q2src = q2lane ^ 15u;"),
            // LLVM gfx1030: v_not_b32_dpp v2, v3 row_half_mirror row_mask:0xf bank_mask:0xf
            (0xff01_4103, "q2src = q2lane ^ 7u;"),
            // LLVM gfx1030: v_not_b32_dpp v2, v3 row_share:5 row_mask:0xf bank_mask:0xf
            (0xff01_5503, "q2src = (q2lane & ~15u) | 5u;"),
            // LLVM gfx1030: v_not_b32_dpp v2, v3 row_xmask:9 row_mask:0xf bank_mask:0xf
            (0xff01_6903, "q2src = q2lane ^ 9u;"),
        ] {
            let hlsl = compute(&[0x7e04_6efa, word]);
            has(&hlsl, &[src, "uint q2 = WaveReadLaneAt(v3, q2src);", "v2 = (~tb_s0);"]);
            assert_eq!(hlsl.contains("q2ok"), word & (1 << 18) == 0, "{hlsl}");
        }
    }

    #[test]
    fn vop2_forms_take_both_source_modifiers() {
        // LLVM gfx1030: v_add_f32_dpp v4, -v1, |v2| row_shr:1 row_mask:0xf bank_mask:0xf
        let hlsl = compute(&[0x0608_04fa, 0xff91_1101]);
        has(&hlsl, &[
            "bool q2ok = (q2lane & 15u) >= 1u && (WaveReadLaneAt(exec_lo, q2src) & 1u) != 0u;",
            "uint tb_s0 = (q2 ^ 0x80000000u);",
            "uint tb_s1 = (v2 & 0x7fffffffu);",
            "if (q2ok) {",
            "asfloat(tb_s0) + asfloat(tb_s1)",
        ]);
        // LLVM gfx1030: v_mul_f32_dpp v4, v1, -v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1 fi:1
        let hlsl = compute(&[0x1008_04fa, 0xff4c_b101]);
        has(&hlsl, &["uint tb_s0 = q2;", "uint tb_s1 = (v2 ^ 0x80000000u);", "asfloat(tb_s0) * asfloat(tb_s1)"]);
        // LLVM gfx1030: v_and_b32_dpp v4, v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        let hlsl = compute(&[0x3608_04fa, 0xff09_6101]);
        has(&hlsl, &["q2src = q2lane ^ 1u;", "uint tb_s0 = (q2ok ? q2 : 0u);", "v4 = (tb_s0 & v2);"]);
        assert!(!hlsl.contains("if (q2ok"), "{hlsl}");
    }

    #[test]
    fn moves_outside_the_reduction_shapes_and_branch_skips() {
        // LLVM gfx1030: v_mov_b32_dpp v4, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
        let hlsl = compute(&[0x7e08_02fa, 0xff09_1401]);
        has(&hlsl, &["q2src = (q2lane & ~15u) | ((q2lane - 4u) & 15u);", "uint tb_s0 = (q2ok ? q2 : 0u);", "v4 = tb_s0;"]);
        // LLVM gfx1030: s_cbranch_execz 2; v_not_b32_dpp v2, v3 row_mirror row_mask:0xf bank_mask:0xf
        // A lane that branches over the instruction counts as inactive for fi=0.
        let hlsl = compute(&[0xbf88_0002, 0x7e04_6efa, 0xff01_4003]);
        has(&hlsl, &["bool q3ok = (WaveReadLaneAt(((j4) ? 0u : exec_lo), q3src) & 1u) != 0u;", "if (q3ok) {"]);
    }

    #[test]
    fn unmodelled_controls_and_operations_are_refused() {
        let refused = |body: &[u32], what: &str| {
            let err = translate(Stage::Compute, 64, body).expect_err(what);
            assert!(err.contains(what), "{err}");
        };
        // VOP1 forms throughout: after a failed compute translation, the lone-lane
        // retry turns an idempotent VOP2 DPP (v_and_b32, v_or_b32, ...) into a copy.
        // LLVM gfx1030: v_not_b32_dpp v2, v3 row_shl:2 row_mask:0xf bank_mask:0x3
        refused(&[0x7e04_6efa, 0xf301_0203], "unmodelled bank_mask 0x3");
        // LLVM gfx1030: v_not_b32_dpp v2, v3 row_shl:2 row_mask:0x2 bank_mask:0xf
        refused(&[0x7e04_6efa, 0x2f01_0203], "row_mask 0x2 depends on the guest wave width");
        // LLVM gfx900: v_mov_b32_dpp v4, v1 row_bcast:15 row_mask:0xf bank_mask:0xf
        refused(&[0x7e08_02fa, 0xff01_4201], "GFX9 row broadcast DPP control 0x142");
        // LLVM gfx900: v_mov_b32_dpp v4, v1 wave_shl:1 row_mask:0xf bank_mask:0xf
        refused(&[0x7e08_02fa, 0xff01_3001], "GFX9 wave shift DPP control 0x130");
        // Hand-built: dpp_ctrl 0x100 (row_shl by zero) and 0x170 are reserved.
        refused(&[0x7e08_02fa, 0xff01_0001], "reserved DPP16 shift by zero 0x100");
        refused(&[0x7e08_02fa, 0xff01_7001], "reserved DPP16 control 0x170");
        // LLVM gfx1030: v_add_co_ci_u32_dpp v4, vcc_lo, v1, v2, vcc_lo row_shr:1 row_mask:0xf bank_mask:0xf
        refused(&[0x5008_04fa, 0xff01_1101], "not a lane-local VGPR operation");
        // Hand-built: v_not_b32_dpp v2, |v3| row_mirror (abs on an integer source).
        refused(&[0x7e04_6efa, 0xff21_4003], "abs/neg on a source that is not f32");
        // LLVM gfx1030: v_cvt_f32_i32_dpp v4, v1 row_shr:1, with the hand-set neg bit.
        refused(&[0x7e08_0afa, 0xff11_1101], "abs/neg on a source that is not f32");
        // s_endpgm; v_not_b32_dpp v2, v3 row_mirror row_mask:0xf bank_mask:0xf
        refused(&[0xbf81_0000, 0x7e04_6efa, 0xff01_4003], "following an early exit");
        // v_not_b32_dpp v2, v3 row_mirror; s_cbranch_scc0 -3 (back to the DPP instruction)
        refused(&[0x7e04_6efa, 0xff01_4003, 0xbf84_fffd], "DPP16 exchange inside backward control flow");
    }

    #[test]
    fn masks_that_enable_no_lane_write_nothing() {
        for body in [
            // LLVM gfx1030: v_not_b32_dpp v2, v3 row_shl:2 row_mask:0x2 bank_mask:0x0
            [0x7e04_6efa, 0x2001_0203],
            // LLVM gfx1030: v_not_b32_dpp v2, v3 row_shl:2 row_mask:0x0 bank_mask:0x3
            [0x7e04_6efa, 0x0301_0203],
            // LLVM gfx1030: v_not_b32_dpp v2, v3 quad_perm:[1,0,3,2] row_mask:0x3 bank_mask:0x0 bound_ctrl:1
            [0x7e04_6efa, 0x3008_b103],
        ] {
            let hlsl = compute(&body);
            let masks = format!("row_mask {:#x} bank_mask {:#x}", body[1] >> 28, (body[1] >> 24) & 15);
            has(&hlsl, &[&format!("// DPP16 {masks}: no lane writes")]);
            assert!(!hlsl.contains("v2 = (~"), "{hlsl}");
        }
    }

    #[test]
    fn asymmetric_row_masks_where_host_lanes_are_guest_lanes() {
        // LLVM gfx1030: v_not_b32_dpp v2, v3 row_shl:2 row_mask:0x2 bank_mask:0xf
        let shl = [0x7e04_6efa, 0x2f01_0203];
        // LLVM gfx1030: v_not_b32_dpp v2, v3 row_ror:3 row_mask:0x1 bank_mask:0xf bound_ctrl:1
        let ror = [0x7e04_6efa, 0x1f09_2303];
        // A 32-thread group is lanes 0-31 of one guest wave, whatever its width.
        let hlsl = translate(Stage::Compute, 32, &shl).unwrap_or_else(|e| panic!("{e}"));
        has(&hlsl, &["if (((0x2u >> (q2lane >> 4u)) & 1u) != 0u && q2ok) {", "v2 = (~tb_s0);"]);
        let hlsl = translate(Stage::Compute, 32, &ror).unwrap_or_else(|e| panic!("{e}"));
        has(&hlsl, &["uint tb_s0 = (q2ok ? q2 : 0u);", "if (((0x1u >> (q2lane >> 4u)) & 1u) != 0u) {"]);
        let err = translate(Stage::Compute, 64, &ror).expect_err("refused");
        assert!(err.contains("row_mask 0x1 depends on the guest wave width"), "{err}");
        // A reduction's EXEC save shows the guest width; the host wave takes it.
        let wave64 = [
            0xbe80_25c1, // LLVM gfx1030 (wave64): s_or_saveexec_b64 s[0:1], -1
            0xd501_0001, 0x0002_0880, // LLVM gfx1030: v_cndmask_b32_e64 v1, 0, v4, s[0:1]
            0x4a02_02e9, 0xde54_c101, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
            0x4a02_02e9, 0x9360_1201, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
            0x4a02_02e9, 0x0009_2401, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
            0x4a02_02fa, 0xff09_4001, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
            0xd778_0002, 0x0201_0101, // LLVM gfx1030: v_permlanex16_b32 v2, v1, 0, 0
            0x4a02_0302, // LLVM gfx1030: v_add_nc_u32_e32 v1, v2, v1
            0xd760_0003, 0x0001_4101, // LLVM gfx1030: v_readlane_b32 s3, v1, 32
            0x4a02_0203, // LLVM gfx1030: v_add_nc_u32_e32 v1, s3, v1
            0xd760_0002, 0x0001_0101, // LLVM gfx1030: v_readlane_b32 s2, v1, 0
            0xbefe_0400, // LLVM gfx1030: s_mov_b64 exec, s[0:1]
        ];
        let wave32 = [
            0xbe80_3dc1, // LLVM gfx1030: s_or_saveexec_b32 s0, -1
            0xd501_0001, 0x0002_0880, // LLVM gfx1030: v_cndmask_b32_e64 v1, 0, v4, s0
            0x4a02_02e9, 0xde54_c101, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
            0x4a02_02e9, 0x9360_1201, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
            0x4a02_02e9, 0x0009_2401, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
            0x4a02_02fa, 0xff09_4001, // LLVM gfx1030: v_add_nc_u32_dpp v1, v1, v1 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
            0xd778_0002, 0x0201_0101, // LLVM gfx1030: v_permlanex16_b32 v2, v1, 0, 0
            0x4a02_0302, // LLVM gfx1030: v_add_nc_u32_e32 v1, v2, v1
            0xd760_0002, 0x0001_0101, // LLVM gfx1030: v_readlane_b32 s2, v1, 0
            0xbefe_0300, // LLVM gfx1030: s_mov_b32 exec_lo, s0
        ];
        for reduction in [&wave64[..], &wave32[..]] {
            let body: Vec<u32> = reduction.iter().chain(&shl).copied().collect();
            let hlsl = compute(&body);
            has(&hlsl, &["((0x2u >> (", "& 1u) != 0u && q", "v2 = (~tb_s"]);
        }
    }

    #[test]
    fn dpp8_reads_within_groups_of_eight() {
        // LLVM gfx1030: v_mov_b32_dpp v4, v1 dpp8:[7,6,5,4,3,2,1,0]
        let hlsl = compute(&[0x7e08_02e9, 0x0539_7701]);
        has(&hlsl, &[
            "uint q2lane = WaveGetLaneIndex(), q2src = (q2lane & ~7u) | ((0x53977u >> (3u * (q2lane & 7u))) & 7u);",
            // fi=0: an inactive source lane reads zero; every active lane writes.
            "bool q2ok = (WaveReadLaneAt(exec_lo, q2src) & 1u) != 0u;",
            "uint q2 = WaveReadLaneAt(v1, q2src);",
            "uint tb_s0 = (q2ok ? q2 : 0u);",
            "if (exec_lo & 1u) v4 = tb_s0;",
        ]);
        assert!(!hlsl.contains("if (q2ok"), "{hlsl}");
        // LLVM gfx1030: v_mov_b32_dpp v4, v1 dpp8:[1,0,3,2,5,4,7,6] fi:1
        let hlsl = compute(&[0x7e08_02ea, 0xde54_c101]);
        has(&hlsl, &["((0xde54c1u >> (3u * (q2lane & 7u))) & 7u)", "uint tb_s0 = q2;", "v4 = tb_s0;"]);
        assert!(!hlsl.contains("q2ok"), "{hlsl}");
        // LLVM gfx1030: v_sub_f32_dpp v5, v1, v2 dpp8:[0,0,0,0,7,7,7,7]
        let hlsl = compute(&[0x080a_04e9, 0xfff0_0001]);
        has(&hlsl, &["((0xfff000u >> (3u * (q2lane & 7u))) & 7u)", "uint tb_s0 = (q2ok ? q2 : 0u);", "asfloat(tb_s0) - asfloat(v2)"]);
        // LLVM gfx1030: v_add_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
        let hlsl = compute(&[0x4a0a_04ea, 0x0539_7701]);
        has(&hlsl, &["uint tb_s0 = q2;", "v5 = (tb_s0 + v2);"]);
        // LLVM gfx1030: v_min_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
        let hlsl = compute(&[0x1e0a_04e9, 0xfac6_8801]);
        has(&hlsl, &["((0xfac688u >> (3u * (q2lane & 7u))) & 7u)", "uint tb_s0 = (q2ok ? q2 : 0u);"]);
        // LLVM gfx1030: s_cbranch_execz 2; v_cvt_f32_i32_dpp v4, v1 dpp8:[1,1,1,1,1,1,1,1]
        // A lane that branches over the instruction counts as inactive for fi=0.
        let hlsl = compute(&[0xbf88_0002, 0x7e08_0ae9, 0x2492_4901]);
        has(&hlsl, &["bool q3ok = (WaveReadLaneAt(((j4) ? 0u : exec_lo), q3src) & 1u) != 0u;", "uint tb_s0 = (q3ok ? q3 : 0u);"]);
    }

    #[test]
    fn unmodelled_dpp8_operations_are_refused() {
        let refused = |body: &[u32], what: &str| {
            let err = translate(Stage::Compute, 64, body).expect_err(what);
            assert!(err.contains(what), "{err}");
        };
        // LLVM gfx1030: v_add_co_ci_u32_dpp v0, vcc_lo, v0, v0, vcc_lo dpp8:[7,6,5,4,3,2,1,0]
        refused(&[0x5000_00e9, 0x0539_7700], "DPP8 v_add_co_ci_u32 at 0x10: not a lane-local VGPR operation");
        // LLVM gfx1030: s_endpgm; v_mov_b32_dpp v4, v1 dpp8:[7,6,5,4,3,2,1,0]
        refused(&[0xbf81_0000, 0x7e08_02e9, 0x0539_7701], "following an early exit");
        // LLVM gfx1030: s_or_saveexec_b32 s0, -1; v_sub_f32_dpp v5, v1, v2 dpp8:[0,0,0,0,7,7,7,7]
        // After an EXEC save a DPP8 operation is taken for a reduction, which this is not.
        refused(&[0xbe80_3dc1, 0x080a_04e9, 0xfff0_0001], "unsupported reduction op v_sub_f32");
        // LLVM gfx1030: v_mov_b32_dpp v4, v1 dpp8:[7,6,5,4,3,2,1,0], in a 16-thread group.
        let err = translate(Stage::Compute, 16, &[0x7e08_02e9, 0x0539_7701]).expect_err("refused");
        assert!(err.contains("wave operation requires complete 32-lane compute waves"), "{err}");
    }

    #[test]
    fn incomplete_waves_and_other_stages_are_refused() {
        // LLVM gfx1030: v_not_b32_dpp v2, v3 row_mirror row_mask:0xf bank_mask:0xf
        let body = [0x7e04_6efa, 0xff01_4003];
        for (stage, threads) in [(Stage::Compute, 16), (Stage::Compute, 48), (Stage::Pixel, 1)] {
            let err = translate(stage, threads, &body).expect_err("refused");
            assert!(err.contains("wave operation requires complete 32-lane compute waves"), "{err}");
        }
        assert!(translate(Stage::Compute, 32, &body).is_ok());
    }
}
