//! Cross-lane `v_readlane_b32` broadcasts and full-wave elections in compute shaders.
//!
//! RDNA2 `v_readlane_b32 sDst, vSrc, lane_sel` reads `vSrc` from lane `lane_sel`
//! regardless of `EXEC`. On Wave32 compute groups where every lane is proven active
//! in `EXEC` and control flow is uniform, `WaveReadLaneAt` is exact with zero
//! groupshared overhead. Otherwise (when `EXEC` may have disabled the target lane,
//! on 2D/3D or partial groups, or for portable Wave64 running on Wave32 hosts),
//! values are staged through `groupshared` `tb_swizzle` with explicit barrier
//! uniformity checks. Unproven dynamic SGPR selectors, out-of-range lanes, and
//! non-uniform loops are refused with explicit diagnostics.
use super::{writes_scalar, Emitter, Enc, Insn, Opnd};
use crate::rdna2::clobbers_scalar;
use std::collections::{BTreeSet, HashMap, VecDeque};

/// Synthetic opcode/mnemonic for native Wave32 `WaveReadLaneAt` lowering.
pub(super) const WAVE32_READ_LANE: &str = "tb_wave32_readlane";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ConstantReadPlan {
    /// At least one `v_readlane_b32` was lowered to groupshared `tb_indexed_readlane`.
    pub(super) uses_shared: bool,
    /// At least one `v_readlane_b32` was lowered to native `tb_wave32_readlane`.
    pub(super) uses_native_wave32: bool,
    /// Logical wave width (`32` or `64`) for groupshared `tb_indexed_readlane` sites.
    pub(super) shared_wave_width: Option<u32>,
    /// Logical wave width (`32` or `64`) when unproven dynamic SGPR selectors
    /// fall back to bounds-checked `GROUP_READ` (`tb_swizzle`) exchange.
    pub(super) group_read_width: Option<u32>,
}

/// A launched compute wave enters with an active low lane. The compiler emits an
/// out-of-line lane-number fallback behind a guard that always bypasses it.
pub(super) fn skip_empty_wave_fallback(insns: &mut Vec<Insn>) -> bool {
    let branch_target = |op: &Insn| op.offset as i64 + 4 + op.aux.imm * 4;
    let branch = |op: &Insn| op.enc == Enc::Sopp && Emitter::branch_cond(op.op).is_some();
    let Some(guard) = insns.iter().position(branch) else {
        return false;
    };
    if guard == 0
        || insns[guard - 1].raw != [0xbeea037e]
        || insns[guard].mnemonic != "s_cbranch_vccnz"
    {
        return false;
    }
    // The initial copy of EXEC_LO to VCC_LO must still see the launch mask.
    if insns[..guard - 1].iter().any(|op| {
        op.mnemonic.contains("exec")
            || op.mnemonic.starts_with("v_cmpx")
            || op.mnemonic == "s_endpgm"
            || op
                .dst
                .iter()
                .any(|v| matches!(v, Opnd::ExecLo | Opnd::ExecHi))
    }) {
        return false;
    }
    // A dynamic transfer or a later jump into the prefix could enter the
    // fallback or revisit its guard with a different EXEC mask.
    let target = branch_target(&insns[guard]);
    let Some(join) = insns.iter().position(|op| op.offset as i64 == target) else {
        return false;
    };
    if join <= guard + 1
        || insns.iter().any(|op| {
            matches!(
                op.mnemonic.as_str(),
                "s_setpc_b64" | "s_swappc_b64" | "s_call_b64"
            ) || (op.mnemonic.contains("branch") && !branch(op))
        })
        || insns[join..]
            .iter()
            .any(|op| branch(op) && branch_target(op) < target)
    {
        return false;
    }
    if !insns[guard + 1..join]
        .iter()
        .any(|op| op.mnemonic == "v_readlane_b32" && matches!(op.src.get(1), Some(Opnd::Sgpr(_))))
    {
        return false;
    }
    insns.drain(guard..join);
    true
}

struct CfgInfo {
    target: HashMap<usize, usize>,
    back: HashMap<usize, usize>,
    loops: Vec<(usize, usize)>,
}

fn build_cfg(insns: &[Insn], first_read_offset: usize) -> Result<CfgInfo, String> {
    if let Some(bad) = insns.iter().find(|op| {
        matches!(
            op.mnemonic.as_str(),
            "s_setpc_b64" | "s_swappc_b64" | "s_call_b64" | "s_rfe_b64" | "s_trap"
        ) || (op.mnemonic.contains("branch")
            && !(op.enc == Enc::Sopp && Emitter::branch_cond(op.op).is_some()))
    }) {
        return Err(format!(
            "v_readlane_b32 at {first_read_offset:#x}: cannot prove convergence across {} at {:#x}",
            bad.mnemonic, bad.offset
        ));
    }
    let mut off2idx: HashMap<usize, usize> =
        insns.iter().enumerate().map(|(i, x)| (x.offset, i)).collect();
    if let Some(last) = insns.last() {
        off2idx.insert(last.offset + last.dwords * 4, insns.len());
    }
    let prog_end = insns.last().map_or(0, |l| l.offset + l.dwords * 4);
    let mut target: HashMap<usize, usize> = HashMap::new();
    let mut back: HashMap<usize, usize> = HashMap::new();
    for (i, ins) in insns.iter().enumerate() {
        if ins.enc == Enc::Sopp && Emitter::branch_cond(ins.op).is_some() {
            let t = ins.offset as i64 + 4 + ins.aux.imm * 4;
            let ti = if t >= 0 && t as usize >= prog_end {
                insns.len()
            } else {
                *off2idx.get(&(t as usize)).ok_or_else(|| {
                    format!(
                        "v_readlane_b32 at {first_read_offset:#x}: branch at {:#x} targets mid-instruction {t:#x}",
                        ins.offset
                    )
                })?
            };
            if ti <= i {
                back.insert(i, ti);
            } else {
                target.insert(i, ti);
            }
        }
    }
    let mut heads: HashMap<usize, usize> = HashMap::new();
    for (&b, &l) in &back {
        let e = heads.entry(l).or_insert(b);
        *e = (*e).max(b);
    }
    let continues: Vec<(usize, usize)> = back
        .iter()
        .filter(|(&b, &l)| heads[&l] != b)
        .map(|(&b, &l)| (b, heads[&l]))
        .collect();
    for (b, closing) in continues {
        back.remove(&b);
        target.insert(b, closing);
    }
    let mut loops: Vec<(usize, usize)> = heads.into_iter().collect();
    loops.sort_unstable();
    for &(l1, b1) in &loops {
        for &(l2, b2) in &loops {
            let disjoint = b1 < l2 || b2 < l1;
            let nested = (l1 <= l2 && b2 <= b1) || (l2 <= l1 && b1 <= b2);
            if !(disjoint || nested) {
                return Err(format!(
                    "v_readlane_b32 at {first_read_offset:#x}: cannot lower across interleaved loops"
                ));
            }
        }
    }
    Ok(CfgInfo { target, back, loops })
}

#[derive(Clone, PartialEq, Eq)]
struct UniformState {
    sgpr: [bool; 106],
    scc: bool,
}

impl UniformState {
    fn entry() -> Self {
        Self { sgpr: [true; 106], scc: false }
    }

    fn all_tainted() -> Self {
        Self { sgpr: [false; 106], scc: false }
    }

    fn opnd(&self, op: &Opnd, wide: bool) -> bool {
        match *op {
            Opnd::ConstInt(_) | Opnd::ConstF32(_) | Opnd::Literal(_) => true,
            Opnd::Scc => self.scc,
            Opnd::Sgpr(r @ 0..=105) => {
                self.sgpr[r as usize]
                    && (!wide || (r < 105 && self.sgpr[r as usize + 1]))
            }
            _ => false,
        }
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (a, &b) in self.sgpr.iter_mut().zip(&other.sgpr) {
            if *a && !b {
                *a = false;
                changed = true;
            }
        }
        if self.scc && !other.scc {
            self.scc = false;
            changed = true;
        }
        changed
    }
}

/// Explicit allowlist of scalar instructions known to leave `SCC` untouched.
fn scalar_preserves_scc(ins: &Insn) -> bool {
    matches!(
        ins.mnemonic.as_str(),
        "s_mov_b32"
            | "s_mov_b64"
            | "s_movk_i32"
            | "s_cselect_b32"
            | "s_cselect_b64"
            | "s_cmov_b32"
            | "s_cmov_b64"
            | "s_cmovk_i32"
            | "s_getpc_b64"
            | "s_brev_b32"
            | "s_brev_b64"
            | "s_bitset0_b32"
            | "s_bitset0_b64"
            | "s_bitset1_b32"
            | "s_bitset1_b64"
            | "s_bfm_b32"
            | "s_bfm_b64"
            | "s_mul_i32"
            | "s_mulk_i32"
            | "s_mul_hi_u32"
            | "s_mul_hi_i32"
            | "s_ff1_i32_b32"
            | "s_ff1_i32_b64"
            | "s_ff0_i32_b32"
            | "s_ff0_i32_b64"
            | "s_flbit_i32_b32"
            | "s_flbit_i32_b64"
            | "s_flbit_i32"
            | "s_flbit_i32_i64"
            | "s_sext_i32_i8"
            | "s_sext_i32_i16"
            | "s_pack_ll_b32_b16"
            | "s_pack_lh_b32_b16"
            | "s_pack_hh_b32_b16"
            | "s_setreg_b32"
            | "s_setreg_imm32_b32"
            | "s_getreg_b32"
    )
}

/// Explicit allowlist of scalar instructions known to write `SCC` solely from their
/// inputs (defaulting every unrecognized scalar instruction to `scc = false`).
fn scalar_writes_scc(ins: &Insn) -> bool {
    let m = ins.mnemonic.as_str();
    match ins.enc {
        Enc::Sopc => m.starts_with("s_cmp_") || m.starts_with("s_bitcmp"),
        Enc::Sopk => m.starts_with("s_cmpk_") || m == "s_addk_i32",
        Enc::Sop2 => matches!(
            m,
            "s_add_u32"
                | "s_sub_u32"
                | "s_add_i32"
                | "s_sub_i32"
                | "s_addc_u32"
                | "s_subb_u32"
                | "s_min_i32"
                | "s_min_u32"
                | "s_max_i32"
                | "s_max_u32"
                | "s_and_b32"
                | "s_and_b64"
                | "s_or_b32"
                | "s_or_b64"
                | "s_xor_b32"
                | "s_xor_b64"
                | "s_andn2_b32"
                | "s_andn2_b64"
                | "s_orn2_b32"
                | "s_orn2_b64"
                | "s_nand_b32"
                | "s_nand_b64"
                | "s_nor_b32"
                | "s_nor_b64"
                | "s_xnor_b32"
                | "s_xnor_b64"
                | "s_lshl_b32"
                | "s_lshl_b64"
                | "s_lshr_b32"
                | "s_lshr_b64"
                | "s_ashr_i32"
                | "s_ashr_i64"
                | "s_bfe_u32"
                | "s_bfe_i32"
                | "s_bfe_u64"
                | "s_bfe_i64"
                | "s_absdiff_i32"
                | "s_lshl1_add_u32"
                | "s_lshl2_add_u32"
                | "s_lshl3_add_u32"
                | "s_lshl4_add_u32"
        ),
        Enc::Sop1 => matches!(
            m,
            "s_not_b32"
                | "s_not_b64"
                | "s_bcnt0_i32_b32"
                | "s_bcnt0_i32_b64"
                | "s_bcnt1_i32_b32"
                | "s_bcnt1_i32_b64"
                | "s_abs_i32"
        ),
        _ => false,
    }
}

fn scalar_reads_dst(ins: &Insn) -> bool {
    let m = ins.mnemonic.as_str();
    (ins.enc == Enc::Sopk && !m.starts_with("s_movk") && !m.starts_with("s_getreg"))
        || matches!(
            m,
            "s_bitset0_b32"
                | "s_bitset0_b64"
                | "s_bitset1_b32"
                | "s_bitset1_b64"
                | "s_cmov_b32"
                | "s_cmov_b64"
        )
}

fn analyze_uniformity_pass(
    insns: &[Insn],
    cfg: &CfgInfo,
    non_uniform_ctl: &[bool],
) -> Vec<UniformState> {
    let n = insns.len();
    let mut states: Vec<Option<UniformState>> = vec![None; n];
    if n == 0 {
        return Vec::new();
    }
    states[0] = Some(UniformState::entry());
    let mut work = VecDeque::from([0usize]);
    let mut iters = 0usize;
    while let Some(i) = work.pop_front() {
        iters += 1;
        if iters > n.saturating_mul(64) {
            return vec![UniformState::all_tainted(); n];
        }
        let ins = &insns[i];
        let incoming = states[i].as_ref().expect("reachable");
        let mut out = incoming.clone();
        let m = ins.mnemonic.as_str();
        let ctl_ok = !non_uniform_ctl[i];
        if m != "v_nop" {
            match ins.enc {
                Enc::Smem => {
                    // Conservatively treat scalar memory loads (s_load_*, s_buffer_load_*)
                    // as non-uniform unless proven read-only immutable: mutable/UAV/coherent
                    // scalar loads can vary across waves in the same workgroup and deadlock
                    // barriers.
                    for r in 0..106u8 {
                        if clobbers_scalar(ins, r) {
                            out.sgpr[r as usize] = false;
                        }
                    }
                }
                Enc::Sop1 | Enc::Sop2 | Enc::Sopk | Enc::Sopc => {
                    let wide = m.ends_with("_b64") || m.ends_with("_u64") || m.ends_with("_i64");
                    let reads_scc = matches!(
                        m,
                        "s_cselect_b32"
                            | "s_cselect_b64"
                            | "s_cmov_b32"
                            | "s_cmov_b64"
                            | "s_cmovk_i32"
                            | "s_addc_u32"
                            | "s_subb_u32"
                    );
                    let uniform = if !ctl_ok {
                        false
                    } else if m == "s_getpc_b64" {
                        true
                    } else if m.contains("exec")
                        || m.starts_with("s_wqm")
                        || m.starts_with("s_movrel")
                        || m.starts_with("s_getreg")
                    {
                        false
                    } else {
                        let src_ok = ins.src.iter().all(|o| incoming.opnd(o, wide));
                        let dst_in_ok = !scalar_reads_dst(ins)
                            || ins.dst.iter().all(|o| incoming.opnd(o, wide));
                        src_ok && dst_in_ok && (!reads_scc || incoming.scc)
                    };
                    for r in 0..106u8 {
                        if clobbers_scalar(ins, r) {
                            out.sgpr[r as usize] = uniform;
                        }
                    }
                    if !scalar_preserves_scc(ins) {
                        out.scc = if scalar_writes_scc(ins) { uniform } else { false };
                    }
                }
                Enc::Sopp => {}
                _ => {
                    for r in 0..106u8 {
                        if clobbers_scalar(ins, r) {
                            out.sgpr[r as usize] = false;
                        }
                    }
                    if m == "v_readlane_b32" {
                        if let Some(Opnd::Vgpr(d @ 0..=105) | Opnd::Sgpr(d @ 0..=105)) =
                            ins.dst.first()
                        {
                            out.sgpr[*d as usize] = false;
                        }
                    }
                }
            }
        }
        let mut push = |next: usize| {
            if next >= n {
                return;
            }
            let changed = match &mut states[next] {
                Some(existing) => existing.join(&out),
                slot => {
                    *slot = Some(out.clone());
                    true
                }
            };
            if changed {
                work.push_back(next);
            }
        };
        if let Some(&t) = cfg.target.get(&i).or_else(|| cfg.back.get(&i)) {
            push(t);
            if ins.enc == Enc::Sopp && ins.op == 2 {
                continue;
            }
        }
        if ins.mnemonic != "s_endpgm" {
            push(i + 1);
        }
    }
    states
        .into_iter()
        .map(|s| s.unwrap_or_else(UniformState::all_tainted))
        .collect()
}

fn is_uniform_branch(ins: &Insn, state: &UniformState, non_uniform_ctl: bool) -> bool {
    if non_uniform_ctl || ins.enc != Enc::Sopp {
        return false;
    }
    match ins.op {
        2 => true,
        4 | 5 => state.scc,
        _ => false,
    }
}

fn is_loop_uniform(
    insns: &[Insn],
    cfg: &CfgInfo,
    l: usize,
    b: usize,
    states: &[UniformState],
    non_uniform_ctl: &[bool],
) -> bool {
    if !is_uniform_branch(&insns[b], &states[b], non_uniform_ctl[b]) {
        return false;
    }
    let bad_body = insns[l..=b].iter().enumerate().any(|(off, op)| {
        let j = l + off;
        op.mnemonic == "s_endpgm"
            || cfg.target.get(&j).is_some_and(|&t| {
                t >= b && !is_uniform_branch(op, &states[j], non_uniform_ctl[j])
            })
    });
    if bad_body {
        return false;
    }
    // Reject any non-uniform forward branch entering the loop body OR skipping the whole loop.
    !cfg.target.iter().any(|(&from, &to)| {
        from < l && to > l && !is_uniform_branch(&insns[from], &states[from], non_uniform_ctl[from])
    })
}

/// Coupled dataflow + control-dependence fixpoint for workgroup uniformity.
/// Every SGPR/SCC written inside the span of a non-uniform branch or loop is tainted.
fn analyze_uniformity(insns: &[Insn], cfg: &CfgInfo) -> (Vec<UniformState>, Vec<bool>) {
    let n = insns.len();
    let mut non_uniform_ctl = vec![false; n];
    loop {
        let states = analyze_uniformity_pass(insns, cfg, &non_uniform_ctl);
        let mut next_ctl = non_uniform_ctl.clone();
        for (&from, &to) in &cfg.target {
            if !is_uniform_branch(&insns[from], &states[from], non_uniform_ctl[from]) {
                let end = to.min(n);
                for slot in next_ctl.iter_mut().take(end).skip(from + 1) {
                    *slot = true;
                }
            }
        }
        for &(l, b) in &cfg.loops {
            if !is_loop_uniform(insns, cfg, l, b, &states, &non_uniform_ctl) {
                for slot in &mut next_ctl[l..=b] {
                    *slot = true;
                }
            }
        }
        if next_ctl == non_uniform_ctl {
            return (states, non_uniform_ctl);
        }
        non_uniform_ctl = next_ctl;
    }
}

#[derive(Clone, PartialEq, Eq)]
struct ExecState {
    full: bool,
    /// `(base_sgpr, is_64bit_pair)` snapshots taken when `full` was true.
    saved_full: BTreeSet<(u8, bool)>,
}

impl ExecState {
    fn entry() -> Self {
        Self { full: true, saved_full: BTreeSet::new() }
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        if self.full && !other.full {
            self.full = false;
            changed = true;
        }
        let before = self.saved_full.len();
        self.saved_full.retain(|entry| other.saved_full.contains(entry));
        if self.saved_full.len() != before {
            changed = true;
        }
        changed
    }
}

fn analyze_exec_full(insns: &[Insn], cfg: &CfgInfo, wave64: bool) -> Vec<bool> {
    let n = insns.len();
    if n == 0 {
        return Vec::new();
    }
    let mut states: Vec<Option<ExecState>> = vec![None; n];
    states[0] = Some(ExecState::entry());
    let mut work = VecDeque::from([0usize]);
    let mut iters = 0usize;
    while let Some(i) = work.pop_front() {
        iters += 1;
        if iters > n.saturating_mul(64) {
            return vec![false; n];
        }
        let ins = &insns[i];
        let incoming = states[i].as_ref().expect("reachable");
        let mut out = incoming.clone();
        let m = ins.mnemonic.as_str();
        out.saved_full.retain(|&(r, pair)| {
            !clobbers_scalar(ins, r) && (!pair || !clobbers_scalar(ins, r.saturating_add(1)))
        });
        // Track SGPRs that capture a full EXEC mask before EXEC itself changes.
        let is_pair = m.ends_with("_b64");
        let width_matches = is_pair == wave64;
        let saves_exec = (matches!(m, "s_mov_b32" | "s_mov_b64")
            && ins.src.first() == Some(&Opnd::ExecLo))
            || m.contains("saveexec");
        if saves_exec && width_matches && incoming.full {
            if let [Opnd::Sgpr(dst @ 0..=105)] = ins.dst.as_slice() {
                if !is_pair || *dst < 105 {
                    out.saved_full.insert((*dst, is_pair));
                }
            }
        } else if matches!(m, "s_mov_b32" | "s_mov_b64") {
            if let ([Opnd::Sgpr(dst @ 0..=105)], [Opnd::Sgpr(src)]) =
                (ins.dst.as_slice(), ins.src.as_slice())
            {
                if incoming.saved_full.contains(&(*src, is_pair)) && (!is_pair || *dst < 105) {
                    out.saved_full.insert((*dst, is_pair));
                }
            }
        }
        // Update whether every lane in the wave is active in EXEC.
        let sets_all = width_matches
            && ((matches!(m, "s_mov_b32" | "s_mov_b64")
                && ins.dst.first() == Some(&Opnd::ExecLo)
                && matches!(ins.src.first(), Some(Opnd::ConstInt(-1) | Opnd::Literal(u32::MAX))))
                || (matches!(m, "s_or_saveexec_b32" | "s_or_saveexec_b64")
                    && matches!(ins.src.first(), Some(Opnd::ConstInt(-1) | Opnd::Literal(u32::MAX))))
                || (matches!(m, "s_nand_saveexec_b32" | "s_nand_saveexec_b64")
                    && ins.src.first() == Some(&Opnd::ConstInt(0))));
        let restores_full = width_matches
            && matches!(m, "s_mov_b32" | "s_mov_b64")
            && ins.dst.first() == Some(&Opnd::ExecLo)
            && matches!(ins.src.first(), Some(Opnd::Sgpr(s)) if incoming.saved_full.contains(&(*s, is_pair)));
        let writes_exec = ins.dst.iter().any(|o| matches!(o, Opnd::ExecLo | Opnd::ExecHi))
            || m.contains("saveexec")
            || m.contains("wrexec")
            || m.starts_with("v_cmpx_");
        if sets_all || restores_full {
            out.full = true;
        } else if writes_exec {
            out.full = false;
        }
        let mut push = |next: usize| {
            if next >= n {
                return;
            }
            let changed = match &mut states[next] {
                Some(existing) => existing.join(&out),
                slot => {
                    *slot = Some(out.clone());
                    true
                }
            };
            if changed {
                work.push_back(next);
            }
        };
        if let Some(&t) = cfg.target.get(&i).or_else(|| cfg.back.get(&i)) {
            push(t);
            if ins.enc == Enc::Sopp && ins.op == 2 {
                continue;
            }
        }
        if m != "s_endpgm" {
            push(i + 1);
        }
    }
    states.into_iter().map(|s| s.is_some_and(|e| e.full)).collect()
}

fn touches_exec(op: &Insn) -> bool {
    op.dst
        .iter()
        .chain(&op.src)
        .any(|o| matches!(o, Opnd::ExecLo | Opnd::ExecHi))
}

fn has_wave32_markers(insns: &[Insn], required_wave_size: Option<u32>, fspx_wave32: bool) -> bool {
    if required_wave_size == Some(32) || fspx_wave32 {
        return true;
    }
    insns.iter().any(|op| {
        let m = op.mnemonic.as_str();
        m.ends_with("saveexec_b32")
            || m.ends_with("wrexec_b32")
            || (op.enc == Enc::Vop3
                && m.starts_with("v_cmp_")
                && matches!(op.dst.as_slice(), [Opnd::Sgpr(n)] if n & 1 != 0))
    })
}

fn has_wave64_markers(
    insns: &[Insn],
    read_sites: &[usize],
    required_wave_size: Option<u32>,
) -> bool {
    if required_wave_size == Some(64) {
        return true;
    }
    if read_sites
        .iter()
        .any(|&i| matches!(insns[i].src.get(1), Some(Opnd::ConstInt(32..=63))))
    {
        return true;
    }
    insns.iter().any(|op| {
        let m = op.mnemonic.as_str();
        op.dst.iter().chain(&op.src).any(|o| matches!(o, Opnd::ExecHi))
            || m.ends_with("saveexec_b64")
            || m.ends_with("wrexec_b64")
            || (matches!(op.enc, Enc::Sop1 | Enc::Sop2 | Enc::Sopc)
                && (m.ends_with("_b64") || m.ends_with("_u64") || m.ends_with("_i64"))
                && touches_exec(op))
            || m == "v_mbcnt_hi_u32_b32"
    })
}

fn has_ambiguous_exec_lo_b32(insns: &[Insn]) -> bool {
    insns.iter().any(|op| {
        let m = op.mnemonic.as_str();
        let is_64 = m.ends_with("_b64") || m.ends_with("_u64") || m.ends_with("_i64");
        matches!(op.enc, Enc::Sop1 | Enc::Sop2 | Enc::Sopc | Enc::Sopk)
            && !is_64
            && op.dst.iter().chain(&op.src).any(|o| matches!(o, Opnd::ExecLo))
    })
}

fn non_spill_read_sites(insns: &[Insn]) -> Vec<usize> {
    let spill_vregs: BTreeSet<u8> = insns
        .iter()
        .filter(|op| op.enc == Enc::Vop3 && op.op == 0x361)
        .filter_map(|op| match op.dst.first() {
            Some(Opnd::Vgpr(v)) => Some(*v),
            _ => None,
        })
        .collect();
    insns
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            op.enc == Enc::Vop3
                && op.op == 0x360
                && op.mnemonic == "v_readlane_b32"
                && matches!(op.src.first(), Some(Opnd::Vgpr(v)) if !spill_vregs.contains(v))
        })
        .map(|(i, _)| i)
        .collect()
}

/// Lower constant-lane and upper-bound-proven SGPR `v_readlane_b32` instructions
/// (excluding SGPR spill VGPRs written by `v_writelane_b32`).
///
/// Validates every non-spill `v_readlane_b32` transactionally before mutating
/// `insns`, refusing out-of-range lanes, partial-group unproven dynamic SGPR
/// lanes, and non-uniform loop/barrier placements with explicit error reasons.
/// When a complete-wave compute group (`count % guest_width == 0`) contains an
/// unproven dynamic SGPR selector, returns `group_read_width = Some(guest_width)`
/// so `lower_lane_spills` + `emit_compute_swizzle(GROUP_READ)` lowers it cleanly
/// with masked `(tb_lane & ~mask) | (sel & mask)` bounds safety.
pub(super) fn lower_constant_reads(
    group: [u32; 3],
    insns: &mut [Insn],
    required_wave_size: Option<u32>,
    fspx_wave32: bool,
    allow_small_group: bool,
    allow_group_read_fallback: bool,
) -> Result<ConstantReadPlan, String> {
    let read_sites = non_spill_read_sites(insns);
    if read_sites.is_empty() {
        return Ok(ConstantReadPlan::default());
    }
    let first_off = insns[read_sites[0]].offset;
    let count = group.into_iter().fold(1u32, u32::saturating_mul);
    if count == 0 || count > 1024 {
        return Err(format!(
            "v_readlane_b32 at {first_off:#x}: compute group size {count} outside 1..=1024"
        ));
    }
    let wave32 = has_wave32_markers(insns, required_wave_size, fspx_wave32);
    let wave64 = has_wave64_markers(insns, &read_sites, required_wave_size);
    if wave32 && wave64 {
        return Err(format!(
            "v_readlane_b32 at {first_off:#x}: conflicting wave32 and wave64 instructions"
        ));
    }
    if !wave32 && !wave64 && has_ambiguous_exec_lo_b32(insns) {
        return Err(format!(
            "v_readlane_b32 at {first_off:#x}: ambiguous wave width (32-bit exec_lo operation without conclusive wave32 or wave64 marker)"
        ));
    }
    let (guest_width, max_lane) = if count <= 32 {
        if !allow_small_group {
            return Ok(ConstantReadPlan::default());
        }
        (if wave64 { 64u32 } else { 32u32 }, count)
    } else if wave32 {
        if count % 32 != 0 {
            return Err(format!(
                "v_readlane_b32 at {first_off:#x}: wave32 compute group ({count} threads) must be a multiple of 32"
            ));
        }
        (32u32, 32u32)
    } else {
        if count % 64 != 0 {
            return Err(format!(
                "v_readlane_b32 at {first_off:#x}: wave64 compute group ({count} threads) must be a multiple of 64"
            ));
        }
        (64u32, 64u32)
    };

    let needs_sgpr_proof = read_sites
        .iter()
        .any(|&i| matches!(insns[i].src.get(1), Some(Opnd::Sgpr(_))));
    let selectors = if needs_sgpr_proof {
        super::scalar_values::at_reads(insns, super::scalar_values::Proof::UpperBound, |op| {
            match (op.mnemonic.as_str(), op.src.get(1)) {
                ("v_readlane_b32", Some(Opnd::Sgpr(reg))) => Some(*reg),
                _ => None,
            }
        })
    } else {
        Default::default()
    };

    let can_group_read_fallback = allow_group_read_fallback && count % guest_width == 0;
    let mut has_unproven_group_read = false;
    let mut dst_regs = Vec::with_capacity(read_sites.len());
    for &idx in &read_sites {
        let op = &insns[idx];
        let dst = match op.dst.as_slice() {
            [Opnd::Vgpr(d @ 0..=105) | Opnd::Sgpr(d @ 0..=105)] => *d,
            other => {
                return Err(format!(
                    "v_readlane_b32 at {:#x}: invalid scalar destination {other:?}",
                    op.offset
                ));
            }
        };
        if !op.is_complete()
            || op.aux.sdwa.is_some()
            || op.aux.imm != 0
            || op.aux.imm2 != 0
            || (op.raw.len() == 2 && op.aux.mask != 0)
        {
            return Err(format!(
                "v_readlane_b32 at {:#x}: unsupported VOP3 modifiers or truncated encoding",
                op.offset
            ));
        }
        match op.src.get(1) {
            Some(Opnd::ConstInt(k)) => {
                if !(0..i64::from(max_lane)).contains(k) {
                    return Err(format!(
                        "v_readlane_b32 at {:#x}: constant lane {k} out of range 0..{max_lane} for wave{guest_width} group",
                        op.offset
                    ));
                }
            }
            Some(Opnd::Sgpr(reg)) => match selectors.get(&(op.offset, *reg)).copied() {
                Some(upper) if upper < max_lane => {}
                Some(upper) => {
                    return Err(format!(
                        "v_readlane_b32 at {:#x}: SGPR lane s{reg} upper bound {upper} exceeds limit {max_lane}",
                        op.offset
                    ));
                }
                None if can_group_read_fallback => {
                    has_unproven_group_read = true;
                }
                None => {
                    return Err(format!(
                        "v_readlane_b32 at {:#x}: dynamic SGPR lane s{reg} lacks an upper-bound proof (< {max_lane})",
                        op.offset
                    ));
                }
            },
            other => {
                return Err(format!(
                    "v_readlane_b32 at {:#x}: unsupported lane selector {other:?}",
                    op.offset
                ));
            }
        }
        dst_regs.push(dst);
    }

    if has_unproven_group_read {
        return Ok(ConstantReadPlan {
            group_read_width: Some(guest_width),
            ..ConstantReadPlan::default()
        });
    }
    let cfg = build_cfg(insns, first_off)?;
    let last_read = *read_sites.last().expect("non-empty");
    if let Some(early) = insns[..last_read].iter().find(|op| op.mnemonic == "s_endpgm") {
        return Err(format!(
            "v_readlane_b32 at {:#x}: follows early s_endpgm at {:#x}",
            insns[last_read].offset, early.offset
        ));
    }
    let (uniformity, non_uniform_ctl) = analyze_uniformity(insns, &cfg);
    // Check that every enclosing loop (and any earlier loop that shares `tb_guard`)
    // is proven workgroup-uniform.
    for &idx in &read_sites {
        for &(l, b) in &cfg.loops {
            if !(l <= idx && idx <= b) {
                continue;
            }
            if !is_loop_uniform(insns, &cfg, l, b, &uniformity, &non_uniform_ctl) {
                return Err(format!(
                    "v_readlane_b32 at {:#x}: inside non-uniform loop ({:#x}..={:#x}); barrier/wave read cannot be proven convergent",
                    insns[idx].offset, insns[l].offset, insns[b].offset
                ));
            }
            // A preceding or nested non-uniform loop would desynchronize the
            // function-wide `tb_guard` counter across lanes in this barrier loop.
            if let Some(&(pl, pb)) = cfg.loops.iter().find(|&&(pl, pb)| {
                (pl, pb) != (l, b)
                    && pb <= b
                    && !is_loop_uniform(insns, &cfg, pl, pb, &uniformity, &non_uniform_ctl)
            }) {
                return Err(format!(
                    "v_readlane_b32 at {:#x}: loop ({:#x}..={:#x}) follows or contains non-uniform loop ({:#x}..={:#x})",
                    insns[idx].offset, insns[l].offset, insns[b].offset, insns[pl].offset, insns[pb].offset
                ));
            }
        }
    }

    let exec_full = analyze_exec_full(insns, &cfg, guest_width == 64);
    let can_wave32_group = guest_width == 32 && count % 32 == 0 && group[1] == 1 && group[2] == 1;
    let mut plan = ConstantReadPlan::default();
    let mut modes = Vec::with_capacity(read_sites.len());
    for &idx in &read_sites {
        let uniform_selector = match insns[idx].src.get(1) {
            Some(Opnd::ConstInt(_)) => true,
            Some(Opnd::Sgpr(reg @ 0..=105)) => uniformity[idx].sgpr[*reg as usize],
            _ => false,
        };
        let native_wave32 = can_wave32_group
            && exec_full[idx]
            && !non_uniform_ctl[idx]
            && uniform_selector;
        modes.push(native_wave32);
    }

    for ((&idx, dst), native_wave32) in read_sites.iter().zip(dst_regs).zip(modes) {
        let op = &mut insns[idx];
        op.enc = Enc::Sop1;
        op.dst[0] = Opnd::Sgpr(dst);
        op.src.truncate(2);
        if native_wave32 {
            op.mnemonic = WAVE32_READ_LANE.into();
            plan.uses_native_wave32 = true;
        } else {
            op.mnemonic = "tb_indexed_readlane".into();
            op.aux.imm = i64::from(guest_width - 1);
            plan.uses_shared = true;
            plan.shared_wave_width = Some(guest_width);
        }
    }
    Ok(plan)
}

/// Backward-compatible helper for direct unit tests of constant/bounded reads.
#[cfg(test)]
pub(super) fn constant_reads(group: [u32; 3], insns: &mut [Insn], allow_small_group: bool) -> bool {
    matches!(
        lower_constant_reads(group, insns, None, false, allow_small_group, false),
        Ok(plan) if plan.uses_shared || plan.uses_native_wave32
    )
}

pub(super) fn prepare(group: [u32; 3], insns: &mut [Insn]) -> bool {
    if !insns
        .iter()
        .any(|op| op.mnemonic == "v_readlane_b32" && matches!(op.src.get(1), Some(Opnd::Sgpr(_))))
        || !insns.iter().any(|op| op.mnemonic == "v_mbcnt_hi_u32_b32")
    {
        return false;
    }
    let count = group.into_iter().fold(1u32, u32::saturating_mul);
    if count == 0 || count > 1024 || count % 64 != 0 {
        return false;
    }
    // Every invocation in every guest wave must participate in the group
    // exchange. Keep arbitrary EXEC, LDS and branching programs on their
    // existing translation paths until their convergence is established.
    if insns.iter().enumerate().any(|(i, op)| {
        op.enc == Enc::Ds
            || op.mnemonic.contains("branch")
            || op.mnemonic.contains("exec")
            || op.mnemonic.starts_with("v_cmpx")
            || matches!(
                op.mnemonic.as_str(),
                "s_setpc_b64" | "s_swappc_b64" | "s_call_b64" | "v_writelane_b32"
            )
            || (op.mnemonic == "s_endpgm" && i + 1 < insns.len())
            || op
                .dst
                .iter()
                .any(|r| matches!(r, Opnd::ExecLo | Opnd::ExecHi))
            || (op.mnemonic.starts_with("v_mbcnt_") && op.src.first() != Some(&Opnd::ConstInt(-1)))
    }) {
        return false;
    }
    let mut reads = Vec::new();
    for i in 0..insns.len() {
        let op = &insns[i];
        let ([Opnd::Vgpr(dst)], [Opnd::Vgpr(source), Opnd::Sgpr(reg), ..]) =
            (&op.dst[..], &op.src[..])
        else {
            continue;
        };
        let (dst, value, selector) = (*dst, Opnd::Vgpr(*source), Opnd::Sgpr(*reg));
        if op.mnemonic != "v_readlane_b32"
            || op.raw
                != [
                    0xd7600000 | u32::from(dst),
                    (u32::from(*reg) << 9) | 256 | u32::from(*source),
                ]
        {
            continue;
        }
        let Some(elected) = insns[..i].iter().rposition(|op| writes_scalar(op, *reg)) else {
            continue;
        };
        let select = &insns[elected];
        let [Opnd::Sgpr(mask)] = select.src[..] else {
            continue;
        };
        if select.mnemonic != "s_ff1_i32_b64" || select.dst != [selector] {
            continue;
        }
        let Some(compare) = insns[..elected]
            .iter()
            .rev()
            .find(|op| writes_scalar(op, mask) || writes_scalar(op, mask.saturating_add(1)))
        else {
            continue;
        };
        // A comparison of unequal literal integers sets all 64 predicate bits.
        // FF1 of that complete mask selects guest lane zero, even on wave32 hosts.
        if compare.dst != [Opnd::Sgpr(mask)]
            || compare.mnemonic != "v_cmp_ne_i32"
            || !matches!(compare.src[..], [Opnd::ConstInt(a), Opnd::ConstInt(b), ..] if a != b)
            || compare.raw[0] & 0xff00 != 0
            || compare.raw[1] >> 27 != 0
        {
            continue;
        }
        reads.push((elected, i, dst, value, selector));
    }
    let changed = !reads.is_empty();
    for (elected, i, dst, value, selector) in reads {
        let select = &mut insns[elected];
        select.mnemonic = "s_mov_b32".into();
        select.op = 3;
        select.src = vec![Opnd::ConstInt(0)];
        let read = &mut insns[i];
        read.mnemonic = "tb_indexed_readlane".into();
        read.enc = Enc::Sop1;
        read.dst = vec![Opnd::Sgpr(dst)];
        read.src = vec![value, selector];
    }
    changed
}

#[cfg(test)]
#[path = "../tests/hlsl/indexed_lane.rs"]
mod tests;
