//! Ordinary scalar bit operations (RDNA2 ISA, SOP1/SOPC).
//! EXEC and VCC are represented per invocation elsewhere in the emitter, not
//! as arbitrary 64-bit scalar values. Never infer a full wave mask from them.
use super::{Emitter, Enc, Insn, Opnd, Res, branch_targets, hi, r};
use crate::rdna2::clobbers_scalar;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
struct Shape {
    src: usize,
    dst: usize,
    reads_dst: bool,
    indexed: bool,
}

fn shape(m: &str) -> Option<Shape> {
    let (src, dst, reads_dst, indexed) = match m {
        "s_bcnt0_i32_b32" | "s_ff0_i32_b32" | "s_flbit_i32" => (1, 1, false, false),
        "s_bcnt0_i32_b64" | "s_ff0_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64" => (2, 1, false, false),
        "s_brev_b64" => (2, 2, false, false),
        "s_bitset0_b64" | "s_bitset1_b64" => (1, 2, true, false),
        "s_bitcmp0_b64" | "s_bitcmp1_b64" => (2, 0, false, true),
        "s_quadmask_b32" => (1, 1, false, false),
        "s_quadmask_b64" => (2, 2, false, false),
        "s_bitreplicate_b64_b32" => (1, 2, false, false),
        _ => return None,
    };
    Some(Shape { src, dst, reads_dst, indexed })
}

fn register(o: &Opnd, width: usize) -> Option<usize> {
    let Opnd::Sgpr(n) = *o else { return None };
    let n = n as usize;
    (n + width <= 106 && (width == 1 || n % 2 == 0)).then_some(n)
}

fn ordinary(known: &[bool; 106], o: &Opnd, width: usize) -> bool {
    match o {
        Opnd::ConstInt(_) => true,
        Opnd::Literal(_) => width == 1,
        _ => register(o, width).is_some_and(|n| known[n..n+width].iter().all(|v| *v)),
    }
}

fn inputs_known(known: &[bool; 106], ins: &Insn, form: Shape) -> bool {
    ins.is_complete()
        && ins.src.len() == if form.indexed { 2 } else { 1 }
        && ins.src.first().is_some_and(|o| ordinary(known, o, form.src))
        && (!form.indexed || ordinary(known, &ins.src[1], 1))
        && (!form.reads_dst || ins.dst.first().is_some_and(|o| ordinary(known, o, form.dst)))
        && if form.dst == 0 { ins.dst == [Opnd::Scc] }
           else { ins.dst.first().and_then(|d| register(d, form.dst)).is_some() }
}

/// Conservative local reaching definitions, not a general mask analysis.
/// A branch entry, call, unknown write, or register pair with an unknown half
/// loses the proof. Actual numeric scalar loads may provide runtime values.
fn ordinary_sites(em: &Emitter<'_>) -> BTreeSet<usize> {
    let mut sites = BTreeSet::new();
    let mut known = [false; 106];
    let mut known_scc = false;
    let compute = em.ex.stage == crate::Stage::Compute;
    let ops = &em.insns;
    // Private packed and mesh paths can replace native scalar operations
    // with ABI-specific logic. Their provenance requires a separate proof.
    if em.packed.is_some() || em.mesh_bindings.is_some()
        || ops.iter().any(|i| matches!(i.mnemonic.as_str(),
            "s_setpc_b64" | "s_swappc_b64" | "s_rfe_b64" | "s_call_b64"))
    {
        return sites;
    }
    let targets = branch_targets(ops);
    for (at, ins) in ops.iter().enumerate() {
        if targets.contains(&(ins.offset as i64)) { known.fill(false); known_scc = false; }
        if ins.mnemonic.starts_with("tb_") || ins.enc == Enc::Sopp
            && (Emitter::branch_cond(ins.op).is_some() || ins.op == 1) {
            known.fill(false);
            known_scc = false;
            continue;
        }
        let m = ins.mnemonic.as_str();
        let src = |k: usize, width| ins.src.get(k).is_some_and(|o| ordinary(&known, o, width));
        let width = if !ins.is_complete() { 0 }
        else if let Some(form) = shape(m) {
            if inputs_known(&known, ins, form) {
                sites.insert(at);
                form.dst
            } else { 0 }
        } else {
            match m {
                "s_movk_i32" => 1,
                "s_mov_b32" | "s_not_b32" | "s_brev_b32" | "s_bcnt1_i32_b32"
                | "s_ff1_i32_b32" | "s_flbit_i32_b32" | "s_sext_i32_i8" | "s_sext_i32_i16"
                | "s_abs_i32" if src(0, 1) => 1,
                "s_mov_b64" | "s_not_b64" if src(0, 2) => 2,
                "s_bcnt1_i32_b64" if src(0, 2) => 1,
                "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
                | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32"
                | "s_add_u32" | "s_sub_u32" | "s_add_i32" | "s_sub_i32" | "s_mul_i32"
                | "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" | "s_bfe_u32" | "s_bfe_i32"
                    if src(0, 1) && src(1, 1) => 1,
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" if src(0, 2) && src(1, 2) => 2,
                // Extend only the ordinary compute path, where these
                // scalar operations are not pixel-quad or mesh substitutes.
                "s_min_i32" | "s_min_u32" | "s_max_i32" | "s_max_u32"
                    if compute && src(0,1) && src(1,1) => 1,
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64"
                    if compute && src(0,2) && src(1,1) => 2,
                "s_bitset0_b32" | "s_bitset1_b32"
                    if compute && src(0,1) && ins.dst.first().is_some_and(|o| ordinary(&known,o,1)) => 1,
                "s_addc_u32" | "s_subb_u32" | "s_cselect_b32"
                    if compute && known_scc && src(0,1) && src(1,1) => 1,
                "s_cselect_b64" if compute && known_scc && src(0,2) && src(1,2) => 2,
                "s_cmov_b32" if compute && known_scc && src(0,1)
                    && ins.dst.first().is_some_and(|o| ordinary(&known,o,1)) => 1,
                "s_cmov_b64" if compute && known_scc && src(0,2)
                    && ins.dst.first().is_some_and(|o| ordinary(&known,o,2)) => 2,
                // Row-permute lane selectors; the vector result defines no SGPR.
                "v_permlane16_b32" | "v_permlanex16_b32" if src(1,1) && src(2,1) => {
                    sites.insert(at);
                    0
                }
                _ if ins.enc == Enc::Smem
                    && matches!(em.res.get(&at).map(Vec::as_slice), Some([Res::Cb(_) | Res::CodeTbl]))
                    && (ins.op <= 4 || (8..=12).contains(&ins.op)) => 1usize << (ins.op & 7),
                _ => 0,
            }
        };
        // SCC is a separate reaching definition. Never trust a condition
        // derived from EXEC/VCC or an unrecognized write. An explicit list
        // of flag-preserving forms is safer than assuming unknown ops preserve SCC.
        known_scc = compute && ins.is_complete() && match m {
            "s_cmp_eq_i32" | "s_cmp_lg_i32" | "s_cmp_gt_i32" | "s_cmp_ge_i32"
            | "s_cmp_lt_i32" | "s_cmp_le_i32" | "s_cmp_eq_u32" | "s_cmp_lg_u32"
            | "s_cmp_gt_u32" | "s_cmp_ge_u32" | "s_cmp_lt_u32" | "s_cmp_le_u32"
            | "s_bitcmp0_b32" | "s_bitcmp1_b32" => src(0,1) && src(1,1),
            "s_bitcmp0_b64" | "s_bitcmp1_b64" => sites.contains(&at),
            "s_add_u32" | "s_sub_u32" | "s_add_i32" | "s_sub_i32"
            | "s_addc_u32" | "s_subb_u32" | "s_min_i32" | "s_max_i32"
            | "s_min_u32" | "s_max_u32"
            | "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32"
            | "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
            | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64"
            | "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" | "s_lshl_b64" | "s_lshr_b64"
            | "s_ashr_i64" | "s_not_b32" | "s_not_b64" | "s_bcnt1_i32_b32"
            | "s_bcnt1_i32_b64" | "s_bcnt0_i32_b32" | "s_bcnt0_i32_b64"
            | "s_abs_i32" | "s_bfe_u32" | "s_bfe_i32"
            | "s_quadmask_b32" | "s_quadmask_b64" =>
                width>0 && ins.dst.first().and_then(|d| register(d,width)).is_some(),
            "s_mov_b32" | "s_mov_b64" | "s_movk_i32" | "s_cmov_b32" | "s_cmov_b64"
            | "s_cselect_b32" | "s_cselect_b64" | "s_brev_b32" | "s_brev_b64"
            | "s_ff0_i32_b32" | "s_ff0_i32_b64" | "s_ff1_i32_b32"
            | "s_flbit_i32_b32" | "s_flbit_i32_b64" | "s_flbit_i32" | "s_flbit_i32_i64"
            | "s_sext_i32_i8" | "s_sext_i32_i16" | "s_bitset0_b32" | "s_bitset1_b32"
            | "s_bitset0_b64" | "s_bitset1_b64" | "s_mul_i32" | "s_bitreplicate_b64_b32" =>
                known_scc && width>0 && ins.dst.first().and_then(|d| register(d,width)).is_some(),
            _ => false,
        };
        // Compute provenance before invalidating destinations, including in-place ops.
        let dest = if width > 0 {
            ins.dst.first().and_then(|d| register(d, width))
        } else { None };
        for (reg, value) in known.iter_mut().enumerate() {
            if *value && clobbers_scalar(ins, reg as u8) { *value = false; }
        }
        if let Some(d) = dest { known[d..d+width].fill(true); }
    }
    sites
}

/// Whether both lane selectors of the row permute at `idx` are proven
/// ordinary scalar values, not lane masks or unknown words.
pub(super) fn ordinary_selectors(em: &Emitter<'_>, idx: usize) -> bool {
    em.ordinary_bit_sites.get_or_init(|| ordinary_sites(em)).contains(&idx)
}

pub(super) fn emit(em: &mut Emitter<'_>, idx: usize, ins: &Insn) -> Result<bool, String> {
    let m = ins.mnemonic.as_str();
    let Some(form) = shape(m) else { return Ok(false) };
    // One linear provenance pass for the whole program, only if needed.
    if !em.ordinary_bit_sites.get_or_init(|| ordinary_sites(em)).contains(&idx) {
        return Err(format!("{m} needs proven ordinary scalar operands (not a wave mask)"));
    }
    let source = &ins.src[0];
    em.touch(source);
    let a = r(source)?;
    let upper = if form.src == 2 {
        let high = hi(source);
        em.touch(&high);
        r(&high)?
    } else { String::new() };
    let dst = &ins.dst[0];
    let t = em.tmp;
    em.tmp += 1;
    let low = format!("tb_bit{t}l");
    let high = format!("tb_bit{t}h");
    let index = format!("tb_bit{t}i");
    let clz = |lo: &str, hi: &str| format!(
        "({hi} != 0u ? 31u - firstbithigh({hi}) : ({lo} != 0u ? 63u - firstbithigh({lo}) : 0xffffffffu))");
    match m {
        "s_brev_b64" => {
            em.line(&format!("uint {low} = reversebits({upper}), {high} = reversebits({a});"));
            em.ws(dst, &low)?;
            em.ws(&hi(dst), &high)?;
        }
        "s_bitset0_b64" | "s_bitset1_b64" => {
            em.touch(dst);
            em.touch(&hi(dst));
            let dl = r(dst)?;
            let dh = r(&hi(dst))?;
            let op = if m == "s_bitset0_b64" { "& ~" } else { "|" };
            em.line(&format!("uint {index} = {a} & 63u;"));
            em.line(&format!("uint {low} = {dl} {op} ({index} < 32u ? (1u << ({index} & 31u)) : 0u);"));
            em.line(&format!("uint {high} = {dh} {op} ({index} >= 32u ? (1u << ({index} & 31u)) : 0u);"));
            em.ws(dst, &low)?;
            em.ws(&hi(dst), &high)?;
        }
        // LLVM folds s_quadmask(0x85003092) to 0xcb and the b64 form of
        // 0x67d000fc85f00a90 to 0xe3e6 (high word 0): result bit i is set
        // when source nibble i is nonzero. LLVM main lists SCC as an implicit
        // def; its value, D != 0, follows the RDNA ISA pseudocode.
        "s_quadmask_b32" | "s_quadmask_b64" => {
            let quad = |word: &str, var: &str| [
                format!("uint {var} = ({word} | ({word} >> 1) | ({word} >> 2) | ({word} >> 3)) & 0x11111111u;"),
                format!("{var} = ({var} | ({var} >> 3)) & 0x03030303u;"),
                format!("{var} = ({var} | ({var} >> 6)) & 0x000f000fu;"),
                format!("{var} = ({var} | ({var} >> 12)) & 0xffu;"),
            ];
            for line in quad(&a, &low) { em.line(&line); }
            if form.src == 2 {
                for line in quad(&upper, &high) { em.line(&line); }
                em.line(&format!("{low} |= {high} << 8;"));
            }
            em.line(&format!("scc = uint({low} != 0u);"));
            em.ws(dst, &low)?;
            if form.dst == 2 { em.ws(&hi(dst), "0u")?; }
        }
        // Source bit i becomes result bits 2i and 2i+1; LLVM folds
        // 0x85fe3a92 to 0xc033fffc_0fccc30c. SCC is not written.
        "s_bitreplicate_b64_b32" => {
            let spread = |half: String, var: &str| [
                format!("uint {var} = {half};"),
                format!("{var} = ({var} | ({var} << 8)) & 0x00ff00ffu;"),
                format!("{var} = ({var} | ({var} << 4)) & 0x0f0f0f0fu;"),
                format!("{var} = ({var} | ({var} << 2)) & 0x33333333u;"),
                format!("{var} = (({var} | ({var} << 1)) & 0x55555555u) * 3u;"),
            ];
            for line in spread(format!("{a} & 0xffffu"), &low) { em.line(&line); }
            for line in spread(format!("{a} >> 16"), &high) { em.line(&line); }
            em.ws(dst, &low)?;
            em.ws(&hi(dst), &high)?;
        }
        "s_bitcmp0_b64" | "s_bitcmp1_b64" => {
            em.touch(&ins.src[1]);
            let b = r(&ins.src[1])?;
            let expect = if m == "s_bitcmp0_b64" { 0 } else { 1 };
            em.line(&format!("uint {index} = {b} & 63u;"));
            em.line(&format!("scc = uint((((({index} < 32u ? {a} : {upper}) >> ({index} & 31u)) & 1u) == {expect}u));"));
        }
        _ => {
            let expr = match m {
                "s_bcnt0_i32_b32" => format!("32u - countbits({a})"),
                "s_bcnt0_i32_b64" => format!("64u - countbits({a}) - countbits({upper})"),
                "s_ff0_i32_b32" => format!("firstbitlow(~{a})"),
                "s_ff0_i32_b64" => format!("({a} != 0xffffffffu ? firstbitlow(~{a}) : ({upper} != 0xffffffffu ? 32u + firstbitlow(~{upper}) : 0xffffffffu))"),
                "s_flbit_i32_b64" => clz(&a, &upper),
                "s_flbit_i32" => {
                    em.line(&format!("uint {high} = {a} ^ (0u - ({a} >> 31u));"));
                    format!("({high} != 0u ? 31u - firstbithigh({high}) : 0xffffffffu)")
                }
                "s_flbit_i32_i64" => {
                    em.line(&format!("uint {index} = 0u - ({upper} >> 31u);"));
                    em.line(&format!("uint {low} = {a} ^ {index}, {high} = {upper} ^ {index};"));
                    clz(&low, &high)
                }
                _ => unreachable!(),
            };
            let out = format!("tb_bit{t}result");
            em.line(&format!("uint {out} = {expr};"));
            if m.starts_with("s_bcnt0") {
                em.line(&format!("scc = uint({out} != 0u);"));
            }
            em.ws(dst, &out)?;
        }
    }
    Ok(true)
}


#[cfg(test)]
#[path = "../tests/hlsl/scalar_bits.rs"]
mod tests;
