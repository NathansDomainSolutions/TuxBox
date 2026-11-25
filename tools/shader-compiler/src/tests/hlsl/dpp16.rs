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
        "v5 = asuint(ceil(asfloat(tb_s0)));",
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
    // The quad binary lowering takes this one; it is not marked here.
    let mut insns = crate::rdna2::disassemble(&[0xfa, 0x04, 0x08, 0x10, 0x01, 0xb1, 0x4c, 0xff]);
    crate::hlsl::lower_wave_reductions(&mut insns).unwrap();
    super::prepare(&mut insns).unwrap();
    assert!(!super::marked(&insns[0]));
    assert!(translate(Stage::Compute, 64, &[0x1008_04fa, 0xff4c_b101]).is_ok());
    // LLVM gfx1030: v_and_b32_dpp v4, v1, v2 row_xmask:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
    let hlsl = compute(&[0x3608_04fa, 0xff09_6101]);
    has(&hlsl, &["q2src = q2lane ^ 1u;", "uint tb_s0 = (q2ok ? q2 : 0u);", "v4 = (tb_s0 & v2);"]);
    assert!(!hlsl.contains("if (q2ok"), "{hlsl}");
}

#[test]
fn moves_outside_the_reduction_shapes_and_branch_skips() {
    // LLVM gfx1030: v_not_b32_dpp v4, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
    // (row shifts on v_mov_b32 itself are the row-move lowering's.)
    let hlsl = compute(&[0x7e08_6efa, 0xff09_1401]);
    has(&hlsl, &["q2src = (q2lane & ~15u) | ((q2lane - 4u) & 15u);", "uint tb_s0 = (q2ok ? q2 : 0u);", "v4 = (~tb_s0);"]);
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
    refused(&[0x7e04_6efa, 0x2f01_0203], "guest wave width");
    // LLVM gfx900: v_mov_b32_dpp v4, v1 row_bcast:15 row_mask:0xf bank_mask:0xf
    refused(&[0x7e08_02fa, 0xff01_4201], "unsupported DPP16 move");
    // LLVM gfx900: v_mov_b32_dpp v4, v1 wave_shl:1 row_mask:0xf bank_mask:0xf
    refused(&[0x7e08_02fa, 0xff01_3001], "unsupported DPP16 move");
    // Hand-built: dpp_ctrl 0x100 (row_shl by zero) and 0x170 are reserved.
    // On v_mov_b32 these four are refused by the move lowering.
    refused(&[0x7e08_02fa, 0xff01_0001], "unsupported DPP16 move");
    refused(&[0x7e08_02fa, 0xff01_7001], "unsupported DPP16 move");
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
    assert!(err.contains("guest wave width"), "{err}");
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
        "v4 = tb_s0;",
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
    // The group exchange lowering reads this one; the sum is the same.
    let hlsl = compute(&[0x4a0a_04ea, 0x0539_7701]);
    has(&hlsl, &["((0x53977u >> (3u * (gtid.x & 7u))) & 7u)", "v5 = t0 + v2;"]);
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
