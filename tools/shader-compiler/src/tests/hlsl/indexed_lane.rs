use super::*;

fn recognizes(words: &[u32], group: [u32; 3]) -> bool {
    let isa: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    prepare(group, &mut crate::rdna2::disassemble(&isa))
}

#[test]
fn partial_group_reads_only_proven_launched_lanes() {
    // In a 7x3 group, registers and inline operands can select lanes 0..20.
    const SET: u32 = 0xbe98_0394; // s_mov_b32 s24,20
    const READ: [u32; 2] = [0xd760_001c, 0x0000_3109]; // s28,v9,s24
    const END: u32 = 0xbf81_0000;
    let check = |words: &[u32]| {
        let isa: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut ops = crate::rdna2::disassemble(&isa);
        let before = ops.iter().map(|op| op.mnemonic.clone()).collect::<Vec<_>>();
        let accepted = constant_reads([7, 3, 1], &mut ops, true);
        if accepted {
            assert!(ops.iter().any(|op| op.mnemonic == "tb_indexed_readlane"));
            assert!(!ops.iter().any(|op| op.mnemonic == "v_readlane_b32"));
        } else {
            assert!(ops.iter().map(|op| &op.mnemonic).eq(&before));
        }
        accepted
    };
    assert!(check(&[SET, READ[0], READ[1], END]));
    assert!(!check(&[SET + 1, READ[0], READ[1], END]));
    assert!(check(&[READ[0], 0x0001_2909, END])); // inline lane 20
    assert!(!check(&[READ[0], 0x0001_2b09, END])); // inline lane 21
    // Equal incoming values at a forward branch join prove the selector.
    assert!(check(&[SET, 0xbf84_0001, SET, READ[0], READ[1], END]));
    assert!(!check(&[SET, 0xbf84_0001, SET + 1, READ[0], READ[1], END]));
    assert!(check(&[SET, 0xbf84_0001, SET - 1, READ[0], READ[1], END]));
    // Keep the runtime selector when both CSELECT operands are in range.
    assert!(check(&[0x8518_8a94, 0xbe99_0318, READ[0], 0x0000_3309, END]));
    assert!(!check(&[0x8518_9594, READ[0], READ[1], END])); // 20 or 21
    assert!(!check(&[0x8518_0594, READ[0], READ[1], END])); // 20 or unknown
    // CMOV keeps the previous destination when SCC is clear.
    const CMOV_20: u32 = 0xbe98_0594; // s_cmov_b32 s24,20
    assert!(check(&[SET - 10, CMOV_20, READ[0], READ[1], END]));
    assert!(!check(&[CMOV_20, READ[0], READ[1], END]));
    assert!(!check(&[SET + 1, CMOV_20, READ[0], READ[1], END]));
    assert!(!check(&[SET, CMOV_20 + 1, READ[0], READ[1], END]));
    assert!(!check(&[SET, 0xbe98_0505, READ[0], READ[1], END])); // source s5 unknown
    assert!(check(&[0x8718_8f05, READ[0], READ[1], END])); // s5 & 15
    assert!(!check(&[0x8718_9f05, READ[0], READ[1], END])); // s5 & 31
    assert!(!check(&[0x8718_8f05, 0x8018_a018, READ[0], READ[1], END])); // add 32
    // The first read overwrites the selector with the selected VGPR value.
    assert!(!check(&[SET, 0xd760_0018, READ[1], READ[0], READ[1], END]));
}

#[test]
fn partial_group_selector_tracks_each_word_of_a_scalar_pair() {
    const OLD: u32 = 0xbe99_038a; // s_mov_b32 s25,10
    const NEW: u32 = 0xbe9b_0394; // s_mov_b32 s27,20
    const MOV: u32 = 0xbe98_041a; // s_mov_b64 s[24:25],s[26:27]
    const CMOV: u32 = 0xbe98_061a; // conditional copy of the same pair
    const SELECT: u32 = 0x8598_181a; // select s[26:27] or s[24:25]
    let reads_high = |setup: &[u32]| {
        let words = [setup, &[0xd760_001c, 0x0000_3309, 0xbf81_0000]].concat();
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        constant_reads([7, 3, 1], &mut crate::rdna2::disassemble(&bytes), true)
    };
    assert!(reads_high(&[NEW, MOV]));
    assert!(!reads_high(&[OLD, MOV])); // unknown source overwrites known high word
    for op in [CMOV, SELECT] {
        assert!(reads_high(&[OLD, NEW, op])); // low words are unknown
        assert!(!reads_high(&[NEW, op]));
        assert!(!reads_high(&[OLD, op]));
        assert!(!reads_high(&[OLD, NEW + 1, op]));
        assert!(!reads_high(&[OLD + 11, NEW, op]));
    }
    assert!(reads_high(&[0xbe98_0487])); // high word of inline 7 is zero
    assert!(!reads_high(&[OLD, 0xbe98_04c1])); // high word of -1 is out of range
}

#[test]
fn empty_wave_fallback_cannot_be_entered_from_another_edge() {
    let pruned = |words: &[u32]| {
        let isa: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut ops = crate::rdna2::disassemble(&isa);
        skip_empty_wave_fallback(&mut ops)
            && !ops.iter().any(|op| op.mnemonic == "v_readlane_b32")
    };
    let words = [
        0xbf800000, 0xbeea037e, 0xbf870002, 0xd760000b, 0x00000d01, 0xbf800000, 0xbf810000,
    ];
    assert!(pruned(&words));
    for (index, word) in [
        (0, 0xbefe0380),
        (1, 0xbeea0380),
        (2, 0xbf860002),
        (5, 0xbf82fffd),
        (5, 0xbf82fffb),
        (5, 0xbe802000),
    ] {
        let mut changed = words;
        changed[index] = word;
        assert!(!pruned(&changed));
    }
}

#[test]
fn lane_election_requires_an_intact_full_wave_predicate() {
    let words = [
        0xd4850004, 0x00010081, // v_cmp_ne_i32 s[4:5], 1, 0
        0xbf800000, // s_nop
        0xbed01404, // s_ff1_i32_b64 s80, s[4:5]
        0xd766000c, 0x00020cc1, // v_mbcnt_hi_u32_b32
        0xd7600000, 0x0000a104, // v_readlane_b32 s0, v4, s80
        0xbf810000,
    ];
    assert!(recognizes(&words, [16, 16, 1]));
    assert!(!recognizes(&words, [32, 1, 1]));
    for change in [0xbe850380, 0xbf820001, 0xbefe047e] {
        let mut changed = words;
        changed[2] = change; // clobber mask high, branch, write EXEC
        assert!(!recognizes(&changed, [16, 16, 1]));
    }
    let mut equal = words;
    equal[1] = 0x00010080; // empty mask
    assert!(!recognizes(&equal, [16, 16, 1]));
}

fn compute_shader(group: [u32; 3], words: &[u32], fspx_config: Option<[u32; 7]>) -> crate::ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::ShaderExtract {
        stage: crate::Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config,
        ps09_pixel_inputs: None,
        fssm: Vec::new(),
        entry: Some("main".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        psv_threads: Some(group),
        flat_inputs: Vec::new(),
        flat_outputs: Vec::new(),
        isa_preamble: [0, group[0], group[1], group[2]],
        has_input_sig: false,
        has_output_sig: false,
        has_psv0: false,
        has_root_sig: false,
        input_sig: Vec::new(),
        output_sig: Vec::new(),
        resources: Vec::new(),
        gs_info: None,
        mesh_info: None,
        primitive_sig: Vec::new(),
        root_sig: None,
    }
}

const fn readlane_const(dst: u8, vsrc: u8, lane: u32) -> [u32; 2] {
    [
        0xd760_0000 | dst as u32,
        ((128 + lane) << 9) | 256 | vsrc as u32,
    ]
}

const fn readlane_sgpr(dst: u8, vsrc: u8, s_lane: u8) -> [u32; 2] {
    [
        0xd760_0000 | dst as u32,
        ((s_lane as u32) << 9) | 256 | vsrc as u32,
    ]
}

const fn writelane_const(vdst: u8, ssrc: u8, lane: u32) -> [u32; 2] {
    [
        0xd761_0000 | vdst as u32,
        ((128 + lane) << 9) | ssrc as u32,
    ]
}

#[test]
fn native_wave32_constant_reads_use_wave_read_lane_at_when_all_lanes_active() {
    for lane in [0u32, 15, 31] {
        let [w0, w1] = readlane_const(7, 1, lane);
        let words = [
            0x7e02_0285, // v_mov_b32 v1, 5
            w0, w1,      // v_readlane_b32 s7, v1, lane
            0x7e04_0207, // v_mov_b32 v2, s7
            0xbf81_0000, // s_endpgm
        ];
        let ex = compute_shader([32, 1, 1], &words, None);
        let tr = crate::hlsl::translate(&ex).expect("native wave32 constant readlane");
        assert_eq!(tr.required_wave_size, Some(32));
        assert!(tr.hlsl.contains("[WaveSize(32)]"), "{}", tr.hlsl);
        assert!(
            tr.hlsl.contains(&format!("s7 = WaveReadLaneAt(v1, {lane}u);")),
            "{}",
            tr.hlsl
        );
        assert!(!tr.hlsl.contains("groupshared uint tb_swizzle"), "{}", tr.hlsl);
        assert!(!tr.hlsl.contains("GroupMemoryBarrierWithGroupSync"), "{}", tr.hlsl);
    }
}

#[test]
fn wave32_masked_exec_readlane_stages_through_groupshared_and_ignores_exec() {
    // Lane 31 writes v1, then v_cmpx_lt_u32 disables lane 31 in EXEC,
    // remaining lanes overwrite v1, then v_readlane_b32 reads lane 31 of v1.
    let [r0, r1] = readlane_const(7, 1, 31);
    let words = [
        0x7e02_0300, // v_mov_b32 v1, v0
        0x7e04_029f, // v_mov_b32 v2, 31
        0x7da2_0500, // v_cmpx_lt_u32 exec, v0, v2 (lane 31 -> EXEC = 0)
        0x7e02_0280, // v_mov_b32 v1, 0 (active lanes overwrite v1; lane 31 keeps 31)
        0xbf88_0002, // s_cbranch_execz +2 (lane 31 skips past readlane)
        r0, r1,      // v_readlane_b32 s7, v1, 31
        0x7e06_0207, // v_mov_b32 v3, s7
        0xbf81_0000, // s_endpgm
    ];
    let ex = compute_shader([32, 1, 1], &words, None);
    let tr = crate::hlsl::translate(&ex).expect("masked wave32 readlane");
    assert_eq!(tr.required_wave_size, None);
    assert!(tr.hlsl.contains("#define TB_SWIZZLE_WORDS 32u\ngroupshared uint tb_swizzle[TB_SWIZZLE_WORDS];"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle[tb_lane] = v1;"), "{}", tr.hlsl);
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~31u) | (0x1fu & 31u))"),
        "{}",
        tr.hlsl
    );
    assert!(!tr.hlsl.contains("WaveReadLaneAt("), "{}", tr.hlsl);
    // Staging must be hoisted BEFORE the skip guard `if (!(j...))`.
    let stage_pos = tr.hlsl.find("tb_swizzle[tb_lane] = v1;").unwrap();
    let guard_pos = tr.hlsl.find("if (!(j").unwrap();
    assert!(stage_pos < guard_pos, "staging must precede skip guard:\n{}", tr.hlsl);
}

#[test]
fn wave32_multiwave_group_uses_32_lane_stride_and_restored_exec_uses_native() {
    // 64 threads with s_and_saveexec_b32 (two 32-lane waves):
    // First readlane happens while EXEC is narrowed -> groupshared stride 32.
    // Second readlane happens after s_mov_b32 exec_lo, s4 restores full EXEC -> native WaveReadLaneAt.
    let [r_masked0, r_masked1] = readlane_const(7, 1, 31);
    let [r_full0, r_full1] = readlane_const(8, 1, 15);
    let words = [
        0x7e02_0300, // v_mov_b32 v1, v0
        0x7e04_029f, // v_mov_b32 v2, 31
        0x7d82_0500, // v_cmp_lt_u32 vcc, v0, v2
        0xbe84_3c6a, // s_and_saveexec_b32 s4, vcc_lo
        r_masked0, r_masked1, // v_readlane_b32 s7, v1, 31 (EXEC narrowed)
        0xbefe_0304, // s_mov_b32 exec_lo, s4 (restores full EXEC)
        r_full0, r_full1,     // v_readlane_b32 s8, v1, 15 (EXEC full again)
        0xbf81_0000,
    ];
    let ex = compute_shader([64, 1, 1], &words, None);
    let tr = crate::hlsl::translate(&ex).expect("multi-wave wave32 readlane");
    assert_eq!(tr.required_wave_size, Some(32));
    assert!(tr.hlsl.contains("[WaveSize(32)]"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("#define TB_SWIZZLE_WORDS 64u\ngroupshared uint tb_swizzle[TB_SWIZZLE_WORDS];"), "{}", tr.hlsl);
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~31u) | (0x1fu & 31u))"),
        "{}",
        tr.hlsl
    );
    assert!(tr.hlsl.contains("s8 = WaveReadLaneAt(v1, 15u);"), "{}", tr.hlsl);
}

#[test]
fn portable_wave64_constant_reads_cross_wave32_boundary_without_wavesize64() {
    for group in [[64, 1, 1], [128, 1, 1]] {
        let [a0, a1] = readlane_const(6, 1, 31);
        let [b0, b1] = readlane_const(7, 1, 47);
        let [c0, c1] = readlane_const(8, 1, 63);
        let words = [
            0x7e02_0300, // v_mov_b32 v1, v0
            a0, a1,
            b0, b1,
            c0, c1,
            0xbf81_0000,
        ];
        let ex = compute_shader(group, &words, None);
        let tr = crate::hlsl::translate(&ex).expect("portable wave64 readlane");
        assert_eq!(tr.required_wave_size, None);
        assert!(!tr.hlsl.contains("[WaveSize(64)]"), "{}", tr.hlsl);
        assert!(
            tr.hlsl.contains(&format!("#define TB_SWIZZLE_WORDS {}u\ngroupshared uint tb_swizzle[TB_SWIZZLE_WORDS];", group[0])),
            "{}",
            tr.hlsl
        );
        assert!(tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (0x1fu & 63u))"), "{}", tr.hlsl);
        assert!(tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (0x2fu & 63u))"), "{}", tr.hlsl);
        assert!(tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (0x3fu & 63u))"), "{}", tr.hlsl);
    }
}

#[test]
fn constant_readlane_coexists_with_sgpr_spills_and_uniform_loops() {
    // Models a shader with SGPR spills on v72, a uniform loop containing
    // v_readlane_b32 s7, v1, 31, and spill restore from v72.
    let [wl0, wl1] = writelane_const(72, 8, 0);
    let [rl_spill0, rl_spill1] = readlane_const(9, 72, 0);
    let [rl_lane0, rl_lane1] = readlane_const(7, 1, 31);
    let words = [
        0xbe88_0383,          // s_mov_b32 s8, 3
        wl0, wl1,             // v_writelane_b32 v72, s8, 0
        0xbe8a_0384,          // s_mov_b32 s10, 4 (uniform loop counter)
        0x7e02_0300,          // L_loop: v_mov_b32 v1, v0
        rl_lane0, rl_lane1,   // v_readlane_b32 s7, v1, 31
        0x818a_810a,          // s_sub_i32 s10, s10, 1
        0xbf07_800a,          // s_cmp_lg_u32 s10, 0
        0xbf85_fffa,          // s_cbranch_scc1 L_loop (-6 dwords = 24 bytes back to 0x10)
        rl_spill0, rl_spill1, // v_readlane_b32 s9, v72, 0
        0xbf81_0000,
    ];
    let ex = compute_shader([64, 1, 1], &words, None);
    let tr = crate::hlsl::translate(&ex).expect("spill + uniform loop + readlane");
    assert!(tr.hlsl.contains("tb_s10000"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (0x1fu & 63u))"), "{}", tr.hlsl);
}

#[test]
fn refuses_out_of_range_unproven_dynamic_and_non_uniform_loop_readlanes() {
    // 1. Out-of-range constant lane 32 in a 32-thread group.
    let [o32_0, o32_1] = readlane_const(7, 1, 32);
    let err = crate::hlsl::translate(&compute_shader([32, 1, 1], &[o32_0, o32_1, 0xbf81_0000], None))
        .err()
        .expect("lane 32 on 32-thread group must be refused");
    assert!(err.contains("out of range"), "{err}");

    // 2. Out-of-range constant lane 64 in a 64-thread group.
    let [o64_0, o64_1] = readlane_const(7, 1, 64);
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &[o64_0, o64_1, 0xbf81_0000], None))
        .err()
        .expect("lane 64 on 64-thread group must be refused");
    assert!(err.contains("out of range"), "{err}");

    // 3. Lane 32 in a Wave32 64-thread group (s_and_saveexec_b32 present).
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[0xbe84_3c6a, o32_0, o32_1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("lane 32 in wave32 shader must be refused");
    assert!(err.contains("conflicting wave32 and wave64"), "{err}");

    // 4. Unproven dynamic SGPR lane selector on partial / non-wave-multiple groups
    //    must stay refused, while complete-wave groups fall back to masked GROUP_READ.
    let [dyn0, dyn1] = readlane_sgpr(7, 1, 5);
    for (group, words) in [
        ([7, 3, 1], vec![dyn0, dyn1, 0xbf81_0000]),
        ([32, 1, 1], vec![0xbefe_04c1, dyn0, dyn1, 0xbf81_0000]),
        ([96, 1, 1], vec![dyn0, dyn1, 0xbf81_0000]),
    ] {
        let err = crate::hlsl::translate(&compute_shader(group, &words, None))
            .err()
            .expect("unproven dynamic SGPR selector on partial wave must be refused");
        assert!(
            err.contains("upper-bound proof") || err.contains("multiple of 64"),
            "{group:?}: {err}"
        );
    }

    // 5. Dynamic SGPR selector masked above limit (`s5 & 63` in 32-thread group).
    let err = crate::hlsl::translate(&compute_shader(
        [32, 1, 1],
        &[0x8705_bf06, dyn0, dyn1, 0xbf81_0000], // s_and_b32 s5, s6, 63
        None,
    ))
    .err()
    .expect("upper bound 63 on 32-thread group must be refused");
    assert!(err.contains("exceeds limit 32"), "{err}");

    // 6. Non-uniform loop (s_cbranch_vccnz back-edge) enclosing v_readlane_b32.
    let [r0, r1] = readlane_const(7, 1, 31);
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[
            0x7e02_0300, // L_loop: v_mov_b32 v1, v0
            r0, r1,      // v_readlane_b32 s7, v1, 31
            0xbf87_fffc, // s_cbranch_vccnz L_loop (-4 dwords)
            0xbf81_0000,
        ],
        None,
    ))
    .err()
    .expect("readlane inside non-uniform VCC loop must be refused");
    assert!(err.contains("non-uniform loop"), "{err}");

    // 7. Non-uniform loop whose SCC depends on v_readlane_b32 itself.
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[
            0x7e02_0300, // L_loop: v_mov_b32 v1, v0
            r0, r1,      // v_readlane_b32 s7, v1, 31
            0xbf07_8007, // s_cmp_lg_u32 s7, 0 (tainted SCC!)
            0xbf85_fffb, // s_cbranch_scc1 L_loop (-5 dwords)
            0xbf81_0000,
        ],
        None,
    ))
    .err()
    .expect("readlane inside lane-dependent SCC loop must be refused");
    assert!(err.contains("non-uniform loop"), "{err}");

    // 8. Early s_endpgm before v_readlane_b32.
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[0xbf88_0001, 0xbf81_0000, r0, r1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("readlane after early s_endpgm must be refused");
    assert!(err.contains("early s_endpgm"), "{err}");

    // 9. Spill VGPR read of unwritten lane still refused by lower_lane_spills.
    let [wl0, wl1] = writelane_const(72, 8, 0);
    let [bad_rl0, bad_rl1] = readlane_const(9, 72, 1);
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[wl0, wl1, bad_rl0, bad_rl1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("unwritten spill lane must stay refused");
    assert!(err.contains("v72 lane 1 read before any v_writelane"), "{err}");
}

#[test]
fn f1_refuses_non_uniform_branch_skipping_entire_readlane_loop_and_prior_non_uniform_loop() {
    let [r0, r1] = readlane_const(7, 1, 31);
    // F1a: `if (exec != 0) { for (s10 = 4; s10 != 0; --s10) v_readlane_b32 ... }`
    let skip_loop = [
        0xbe8a_0384, // 0x00: s_mov_b32 s10, 4
        0xbe84_3c6a, // 0x04: s_and_saveexec_b32 s4, vcc_lo
        0xbf88_0005, // 0x08: s_cbranch_execz +5 -> 0x20 (past loop end!)
        r0, r1,      // 0x0c: L_loop: v_readlane_b32 s7, v1, 31
        0x818a_810a, // 0x14: s_sub_i32 s10, s10, 1
        0xbf07_800a, // 0x18: s_cmp_lg_u32 s10, 0
        0xbf85_fffb, // 0x1c: s_cbranch_scc1 L_loop
        0xbf81_0000, // 0x20: s_endpgm
    ];
    let err = crate::hlsl::translate(&compute_shader([32, 1, 1], &skip_loop, None))
        .err()
        .expect("non-uniform branch skipping entire barrier loop must be refused");
    assert!(err.contains("non-uniform loop"), "{err}");

    // F1b: Preceding non-uniform loop would desync function-wide `tb_guard` before barrier loop.
    let prior_non_uniform = [
        0x7e02_0300, // 0x00: L0: v_mov_b32 v1, v0
        0xbf87_fffe, // 0x04: s_cbranch_vccnz L0 (non-uniform!)
        0xbe8a_0384, // 0x08: s_mov_b32 s10, 4
        r0, r1,      // 0x0c: L1: v_readlane_b32 s7, v1, 31
        0x818a_810a, // 0x14: s_sub_i32 s10, s10, 1
        0xbf07_800a, // 0x18: s_cmp_lg_u32 s10, 0
        0xbf85_fffb, // 0x1c: s_cbranch_scc1 L1
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &prior_non_uniform, None))
        .err()
        .expect("barrier loop after non-uniform loop must be refused");
    assert!(err.contains("follows or contains non-uniform loop"), "{err}");

    // F1b-nested: Uniform outer barrier loop containing a non-uniform inner loop
    // would also desync function-wide `tb_guard` across lanes.
    let nested_non_uniform = [
        0xbe8a_0384, // 0x00: s_mov_b32 s10, 4
        r0, r1,      // 0x04: L_outer: v_readlane_b32 s7, v1, 31
        0x7e02_0300, // 0x0c: L_inner: v_mov_b32 v1, v0
        0xbf87_fffe, // 0x10: s_cbranch_vccnz L_inner (non-uniform inner loop!)
        0x818a_810a, // 0x14: s_sub_i32 s10, s10, 1
        0xbf07_800a, // 0x18: s_cmp_lg_u32 s10, 0
        0xbf85_fff9, // 0x1c: s_cbranch_scc1 L_outer (-7 dwords -> 0x04)
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &nested_non_uniform, None))
        .err()
        .expect("barrier loop containing non-uniform inner loop must be refused");
    assert!(err.contains("follows or contains non-uniform loop"), "{err}");
}

#[test]
fn f2_control_dependence_taints_sgpr_and_scc_under_non_uniform_branch() {
    let [r0, r1] = readlane_const(7, 1, 31);
    // F2a: `s10 = 4; if (v0 != 0) s10 = 1;` then loop on s10 around v_readlane_b32 -> refused!
    let divergent_trip = [
        0xbe8a_0384, // s_mov_b32 s10, 4
        0x7d84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xbf87_0001, // s_cbranch_vccnz +1 (lane 0 keeps s10=4, others set s10=1)
        0xbe8a_0381, // s_mov_b32 s10, 1
        r0, r1,      // L_loop: v_readlane_b32 s7, v1, 31
        0x818a_810a, // s_sub_i32 s10, s10, 1
        0xbf07_800a, // s_cmp_lg_u32 s10, 0
        0xbf85_fffb, // s_cbranch_scc1 L_loop
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &divergent_trip, None))
        .err()
        .expect("control-dependent trip count must be refused");
    assert!(err.contains("non-uniform loop"), "{err}");

    // F2b: Control-dependent guard over v_readlane_b32 on [32,1,1] MUST NOT use WaveReadLaneAt.
    let divergent_guard = [
        0xbe8a_0380,             // s_mov_b32 s10, 0
        0x7d84_009f,             // v_cmp_eq_u32 vcc, 31, v0 (lane 31 sets vcc=1)
        0xbf87_0001,             // s_cbranch_vccnz +1 (lane 31 keeps s10=0)
        0xbe8a_0381,             // s_mov_b32 s10, 1 (lanes 0..30 get s10=1)
        0xbf06_800a,             // s_cmp_eq_u32 s10, 0 (lane 31 sets scc=1)
        0xbf85_0004,             // s_cbranch_scc1 +4 (lane 31 skips overwrite+readlane!)
        0x7e02_02ff, 0xdead0000, // v_mov_b32 v1, 0xdead0000
        r0, r1,                  // v_readlane_b32 s7, v1, 31
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([32, 1, 1], &divergent_guard, None))
        .expect("control-dependent guard must fall back to hoisted groupshared");
    assert!(!tr.hlsl.contains("WaveReadLaneAt("), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle[tb_lane] = v1;"), "{}", tr.hlsl);

    // F2c: Control-dependent bounded SGPR selector MUST NOT use WaveReadLaneAt.
    let [rs0, rs1] = readlane_sgpr(7, 1, 10);
    let divergent_sel = [
        0xbe8a_0385, // s_mov_b32 s10, 5
        0x7d84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xbf87_0001, // s_cbranch_vccnz +1
        0xbe8a_0383, // s_mov_b32 s10, 3
        rs0, rs1,    // v_readlane_b32 s7, v1, s10
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([32, 1, 1], &divergent_sel, None))
        .expect("control-dependent bounded selector falls back to groupshared");
    assert!(!tr.hlsl.contains("WaveReadLaneAt("), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle["), "{}", tr.hlsl);
}

#[test]
fn f3_uniformity_tracks_sop1_rmw_scc_preserving_ops_and_smem_soffset() {
    let [r0, r1] = readlane_const(7, 1, 31);
    // F3a-1: SOP1 RMW `s_bitset0_b32 s6, 31` must keep tainted s6 tainted.
    let bitset_poc = [
        0xd4c2_0006, 0x0002_0080, // v_cmp_eq_u32 s[6:7], 0, v0
        0xbe86_1b9f,              // s_bitset0_b32 s6, 31
        r0, r1,                   // L_loop: v_readlane_b32 s7, v1, 31
        0xbf07_8006,              // s_cmp_lg_u32 s6, 0
        0xbf85_fffc,              // s_cbranch_scc1 L_loop
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &bitset_poc, None))
        .err()
        .expect("s_bitset0 on tainted SGPR must stay non-uniform");
    assert!(err.contains("non-uniform loop"), "{err}");

    // F3a-2: SOP1 RMW `s_cmov_b32 s6, 0` must keep tainted s6 tainted.
    let cmov_poc = [
        0xd4c2_0006, 0x0002_0080, // v_cmp_eq_u32 s[6:7], 0, v0
        0xbf06_8180,              // s_cmp_eq_u32 0, 1 (uniform scc = 0)
        0xbe86_0580,              // s_cmov_b32 s6, 0 (preserves tainted s6!)
        r0, r1,                   // L_loop: v_readlane_b32 s7, v1, 31
        0xbf07_8006,              // s_cmp_lg_u32 s6, 0
        0xbf85_fffc,              // s_cbranch_scc1 L_loop
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &cmov_poc, None))
        .err()
        .expect("s_cmov on tainted SGPR must stay non-uniform");
    assert!(err.contains("non-uniform loop"), "{err}");

    // F3b: SCC-preserving `s_mulk_i32 s10, 2` after non-uniform `s_cmp_lg_u32` must keep scc tainted.
    let mulk_poc = [
        0xd4c2_0006, 0x0002_0080, // v_cmp_eq_u32 s[6:7], 0, v0
        0xbe8a_0384,              // s_mov_b32 s10, 4
        r0, r1,                   // L_loop: v_readlane_b32 s7, v1, 31
        0xbf07_8006,              // s_cmp_lg_u32 s6, 0 (tainted scc)
        0xb80a_0002,              // s_mulk_i32 s10, 2 (preserves scc!)
        0xbf85_fffb,              // s_cbranch_scc1 L_loop
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &mulk_poc, None))
        .err()
        .expect("s_mulk_i32 must not clear tainted SCC");
    assert!(err.contains("non-uniform loop"), "{err}");

    // F3c: SMEM with non-NULL special soffset (106 = VCC_LO) must be non-uniform.
    let smem_vcc_soffset = [
        0xf400_0180, 0xd400_0000, // s_load_dword s6, s[0:1], vcc_lo (soffset=106)
        r0, r1,                   // L_loop: v_readlane_b32 s7, v1, 31
        0xbf07_8006,              // s_cmp_lg_u32 s6, 0
        0xbf85_fffc,              // s_cbranch_scc1 L_loop
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &smem_vcc_soffset, None))
        .err()
        .expect("SMEM with VCC soffset must be non-uniform");
    assert!(err.contains("non-uniform loop"), "{err}");
}

#[test]
fn f4_detects_all_exec_width_ops_and_checks_conflicts_on_small_groups() {
    let [r0, r1] = readlane_const(7, 1, 31);
    // F4a: [64, 1, 1] using ONLY `s_and_b32 exec_lo` + `s_or_b32 exec_lo` without a
    // conclusive Wave32 or Wave64 marker has ambiguous wave width and MUST be refused.
    let ambiguous_logic_exec = [
        0x877e_6a7e, // s_and_b32 exec_lo, exec_lo, vcc_lo
        r0, r1,      // v_readlane_b32 s7, v1, 31
        0x887e_c17e, // s_or_b32 exec_lo, exec_lo, -1
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &ambiguous_logic_exec, None))
        .err()
        .expect("ambiguous 32-bit exec_lo ops without conclusive marker must be refused");
    assert!(err.contains("ambiguous wave width"), "{err}");

    // With a conclusive Wave32 marker (odd SGPR VOP3 v_cmp_lt_u32 s5), [64, 1, 1]
    // classifies as Wave32 (stride 31u, NOT 63u).
    let conclusive_wave32 = [
        0xd4c1_0005, 0x0002_0702, // v_cmp_lt_u32 s5, v2, v3 (odd SGPR -> Wave32)
        0x877e_057e,              // s_and_b32 exec_lo, exec_lo, s5
        r0, r1,                   // v_readlane_b32 s7, v1, 31
        0x887e_c17e,              // s_or_b32 exec_lo, exec_lo, -1
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([64, 1, 1], &conclusive_wave32, None))
        .expect("conclusive wave32 readlane");
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~31u) | (0x1fu & 31u))"),
        "{}",
        tr.hlsl
    );
    assert!(!tr.hlsl.contains("~63u"), "{}", tr.hlsl);

    // F4b: [32, 1, 1] with conflicting 32-bit and 64-bit EXEC saves MUST be refused.
    let conflict_small = [
        0xbe84_3c6a, // s_and_saveexec_b32 s4, vcc_lo
        0xbe86_246a, // s_and_saveexec_b64 s[6:7], vcc
        r0, r1,
        0xbf81_0000,
    ];
    let err = crate::hlsl::translate(&compute_shader([32, 1, 1], &conflict_small, None))
        .err()
        .expect("conflicting wave32/wave64 on 32-thread group must be refused");
    assert!(err.contains("conflicting wave32 and wave64"), "{err}");

    // F4c: [32, 1, 1] with Wave64 EXEC marker stays portable (no [WaveSize(32)]).
    let wave64_small = [
        0xbefe_04c1, // s_mov_b64 exec, -1
        r0, r1,
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([32, 1, 1], &wave64_small, None))
        .expect("wave64 32-thread group uses portable groupshared");
    assert_eq!(tr.required_wave_size, None);
    assert!(!tr.hlsl.contains("[WaveSize(32)]"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle["), "{}", tr.hlsl);
}

#[test]
fn f5_indexed_lane_preserves_exec_ballot_for_v_mbcnt_lo() {
    let [r0, r1] = readlane_const(7, 1, 31);
    let words = [
        0xbe84_3c6a,              // s_and_saveexec_b32 s4, vcc_lo
        0xd765_0002, 0x0001_007e, // v_mbcnt_lo_u32_b32 v2, exec_lo, 0
        r0, r1,                   // v_readlane_b32 s7, v1, 31
        0xbefe_0304,              // s_mov_b32 exec_lo, s4
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([32, 1, 1], &words, None))
        .expect("wave32 indexed_lane with v_mbcnt_lo(exec_lo)");
    assert_eq!(tr.required_wave_size, Some(32));
    assert!(
        tr.hlsl.contains("WaveActiveBallot((exec_lo & 1u) != 0u).x"),
        "{}",
        tr.hlsl
    );
    assert!(!tr.hlsl.contains("countbits(exec_lo &"), "{}", tr.hlsl);
}

#[test]
fn f6_hoisted_indexed_read_guards_load_against_oob_on_partial_groups() {
    // [7, 3, 1] -> tb_swizzle[21]. Lane 0 sets s24 = 30 and skips past s24 = 5 + readlane.
    // The hoisted load MUST guard tb_swizzle[...] on `!j...` while keeping both barriers unconditional.
    let [rs0, rs1] = readlane_sgpr(7, 1, 24);
    let words = [
        0xbe98_039e, // s_mov_b32 s24, 30
        0x7d84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xbf87_0003, // s_cbranch_vccnz +3 (lane 0 skips s24=5 and readlane)
        0xbe98_0385, // s_mov_b32 s24, 5
        rs0, rs1,    // v_readlane_b32 s7, v1, s24
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([7, 3, 1], &words, None))
        .expect("partial group guarded hoisted readlane");
    assert!(tr.hlsl.contains("#define TB_SWIZZLE_WORDS 21u\ngroupshared uint tb_swizzle[TB_SWIZZLE_WORDS];"), "{}", tr.hlsl);
    assert!(
        tr.hlsl.contains("(!j5) ? tb_swizzle_read((tb_lane & ~31u) | (s24 & 31u)) : 0u;"),
        "{}",
        tr.hlsl
    );
}

#[test]
fn minor_refuses_flagged_vop3_readlane_encoding() {
    // Set CLAMP bit (bit 15) on v_readlane_b32 VOP3 word 0.
    let [r0, r1] = readlane_const(7, 1, 31);
    let err = crate::hlsl::translate(&compute_shader(
        [32, 1, 1],
        &[r0 | (1 << 15), r1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("CLAMP-flagged v_readlane_b32 must be refused");
    assert!(err.contains("modifiers"), "{err}");
}

#[test]
fn wave64_exec_lo_b32_inspect_keeps_wave64_and_ambiguous_b32_exec_refuses() {
    let [r31_0, r31_1] = readlane_const(7, 1, 31);
    // 1. Wave64 shader (ExecHi referenced via s_mov_b32 s5, exec_hi) that also inspects
    // exec_lo via s_mov_b32 MUST stay Wave64 (64-lane stride ~63u, never Wave32).
    let wave64_inspect_exec_lo = [
        0xbe84_037e, // s_mov_b32 s4, exec_lo
        0xbe85_037f, // s_mov_b32 s5, exec_hi
        r31_0, r31_1,
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([64, 1, 1], &wave64_inspect_exec_lo, None))
        .expect("wave64 reading exec_lo and exec_hi via s_mov_b32 must keep wave64");
    assert_eq!(tr.required_wave_size, None);
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (0x1fu & 63u))"),
        "{}",
        tr.hlsl
    );
    assert!(!tr.hlsl.contains("~31u"), "{}", tr.hlsl);

    // 2. Wave64 shader (s_mov_b64 exec, -1) inspecting exec_lo with s_and_b32 MUST keep Wave64.
    let wave64_and_exec_lo = [
        0xbefe_04c1, // s_mov_b64 exec, -1
        0x8704_057e, // s_and_b32 s4, exec_lo, s5
        r31_0, r31_1,
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([64, 1, 1], &wave64_and_exec_lo, None))
        .expect("wave64 s_and_b32 reading exec_lo must keep wave64");
    assert_eq!(tr.required_wave_size, None);
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (0x1fu & 63u))"),
        "{}",
        tr.hlsl
    );

    // 3. Ambiguous s_mov_b32 exec_lo, s0 without conclusive wave32/wave64 markers MUST refuse.
    let ambiguous_mov_exec_lo = [
        0xbefe_0300, // s_mov_b32 exec_lo, s0
        r31_0, r31_1,
        0xbf81_0000,
    ];
    for group in [[64, 1, 1], [32, 1, 1]] {
        let err = crate::hlsl::translate(&compute_shader(group, &ambiguous_mov_exec_lo, None))
            .err()
            .expect("ambiguous s_mov_b32 exec_lo must be refused");
        assert!(err.contains("ambiguous wave width"), "{err}");
    }
}

#[test]
fn unproven_swizzle_fallback_refuses_partial_groups_and_preserves_skipped_lanes() {
    let [dyn0, dyn1] = readlane_sgpr(7, 1, 5);
    // 1. Partial/non-64-multiple groups with ds_swizzle_b32 + s_mov_b64 + unproven dynamic
    // v_readlane_b32 MUST refuse instead of emitting unbounded tb_swizzle[(tb_lane & ~63u) | s5].
    let swizzle_unproven = [
        0xbefe_04c1,             // s_mov_b64 exec, -1
        0xd8d4_041f, 0x0000_0001, // ds_swizzle_b32 v0, v1 offset:0x041f
        dyn0, dyn1,              // v_readlane_b32 s7, v1, s5 (unproven!)
        0xbf81_0000,
    ];
    for group in [[32, 1, 1], [7, 3, 1], [96, 1, 1]] {
        let err = crate::hlsl::translate(&compute_shader(group, &swizzle_unproven, None))
            .err()
            .expect("unproven swizzle fallback must be refused when count % 64 != 0");
        assert!(
            err.contains("upper-bound proof") || err.contains("multiple of 64"),
            "{group:?}: {err}"
        );
    }

    // 2. Wave32 [64, 1, 1] shader with ds_swizzle_b32 + non-exec s_mov_b64 + unproven selector
    // falls back to Wave32-masked GROUP_READ (~31u / & 31u), NEVER hardcoded 64-lane base (~63u).
    let wave32_swizzle_unproven = [
        0xbe84_3c6a,             // s_and_saveexec_b32 s4, vcc_lo (Wave32)
        0xbe8a_040c,             // s_mov_b64 s[10:11], s[12:13]
        0xd8d4_041f, 0x0000_0001, // ds_swizzle_b32 v0, v1 offset:0x041f
        dyn0, dyn1,              // v_readlane_b32 s7, v1, s5
        0xbefe_0304,             // s_mov_b32 exec_lo, s4
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([64, 1, 1], &wave32_swizzle_unproven, None))
        .expect("wave32 complete-wave unproven selector falls back to 32-lane GROUP_READ");
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~31u) | (s5 & 31u))"),
        "{}",
        tr.hlsl
    );
    assert!(!tr.hlsl.contains("~63u"), "{}", tr.hlsl);

    // 2b. Standalone (no ds_swizzle_b32) unproven dynamic SGPR v_readlane_b32 on complete
    // wave groups ([32,1,1] Wave32, [64,1,1] Wave32, [64,1,1] Wave64) falls back cleanly
    // with exact wave-width masking and unconditional staging.
    let tr_w32_32 = crate::hlsl::translate(&compute_shader([32, 1, 1], &[dyn0, dyn1, 0xbf81_0000], None))
        .expect("standalone [32,1,1] unproven dynamic SGPR falls back to 32-lane GROUP_READ");
    assert!(tr_w32_32.hlsl.contains("tb_swizzle_read((tb_lane & ~31u) | (s5 & 31u))"), "{}", tr_w32_32.hlsl);
    assert!(!tr_w32_32.hlsl.contains("~63u"), "{}", tr_w32_32.hlsl);

    let tr_w64_64 = crate::hlsl::translate(&compute_shader([64, 1, 1], &[dyn0, dyn1, 0xbf81_0000], None))
        .expect("standalone [64,1,1] unproven dynamic SGPR falls back to 64-lane GROUP_READ");
    assert!(tr_w64_64.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (s5 & 63u))"), "{}", tr_w64_64.hlsl);

    let tr_w32_64 = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[0xbe84_3c6a, dyn0, dyn1, 0xbefe_0304, 0xbf81_0000],
        None,
    ))
    .expect("standalone wave32 [64,1,1] unproven dynamic SGPR falls back to 32-lane GROUP_READ");
    assert!(tr_w32_64.hlsl.contains("tb_swizzle_read((tb_lane & ~31u) | (s5 & 31u))"), "{}", tr_w32_64.hlsl);
    assert!(!tr_w32_64.hlsl.contains("~63u"), "{}", tr_w32_64.hlsl);

    // 3. Valid [64, 1, 1] Wave64 ds_swizzle_b32 + GROUP_READ inside a conditional skip branch
    // MUST publish each lane's actual VGPR unconditionally (preserving skipped lanes' values),
    // guard ONLY the load on the skip flag, and mask the selector with `& 63u`.
    let swizzle_branch_read = [
        0x7e02_0300,             // v_mov_b32 v1, v0
        0x7e04_02bf,             // v_mov_b32 v2, 63
        0x7d82_0500,             // v_cmp_lt_u32 vcc, v0, v2
        0xbe84_246a,             // s_and_saveexec_b64 s[4:5], vcc (lane 63 EXEC = 0)
        0xd8d4_041f, 0x0000_0001, // ds_swizzle_b32 v0, v1 offset:0x041f
        0xbf88_0002,             // s_cbranch_execz +2 (lane 63 skips readlane)
        dyn0, dyn1,              // v_readlane_b32 s7, v1, s5 (lowered via GROUP_READ)
        0xbefe_0404,             // s_mov_b64 exec, s[4:5]
        0xbf81_0000,
    ];
    let tr = crate::hlsl::translate(&compute_shader([64, 1, 1], &swizzle_branch_read, None))
        .expect("wave64 64-lane swizzle GROUP_READ inside forward branch");
    assert!(tr.hlsl.contains("#define TB_SWIZZLE_WORDS 64u\ngroupshared uint tb_swizzle[TB_SWIZZLE_WORDS];"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle[tb_lane] = v1;"), "{}", tr.hlsl);
    assert!(
        tr.hlsl.contains("tb_swizzle_read((tb_lane & ~63u) | (s5 & 63u)) : 0u;"),
        "{}",
        tr.hlsl
    );
    assert!(
        !tr.hlsl.contains("tb_swizzle[tb_lane] = (!j"),
        "GROUP_READ must not zero branch-skipped lanes:\n{}",
        tr.hlsl
    );
}

#[test]
fn refused_readlane_is_not_swallowed_by_lone_lane_retry() {
    // Election + LDS atomic prefix that `lone_lane::prepare` accepts and translates cleanly
    // (rewriting `v_readlane_b32` to `v_readfirstlane_b32`), while on the primary path it
    // reaches `lower_constant_reads` so every cross-lane refusal must propagate as `Err`.
    const ELECT_ATOMIC: [u32; 4] = [
        0xbe88_137e,             // s_ff1_i32_b32 s8, exec_lo
        0x7da4_0008,             // v_cmpx_eq_u32 exec, s8, v0
        0xd800_0000, 0x0000_0403, // ds_add_u32 v3, v4
    ];
    let prefix_ops = crate::rdna2::disassemble(
        &ELECT_ATOMIC.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
    );
    assert_eq!(prefix_ops[0].mnemonic, "s_ff1_i32_b32");
    assert_eq!(prefix_ops[1].mnemonic, "v_cmpx_eq_u32");
    assert_eq!(prefix_ops[2].mnemonic, "ds_add_u32");

    let assert_refused_despite_lone_lane = |group: [u32; 3], words: &[u32], expected: &str| {
        let ex = compute_shader(group, words, None);
        // Prove the shader really would have been swallowed by `lone_lane::retry`.
        let retry_ok = crate::hlsl::lone_lane::retry(|| {
            crate::hlsl::translate_outputs_once(&ex, None, &[], None)
        })
        .and_then(|t| crate::hlsl::lone_lane::check(&t.hlsl));
        assert!(
            retry_ok.is_ok(),
            "test fixture must be accepted by lone_lane::retry (got {retry_ok:?})"
        );
        let err = crate::hlsl::translate(&ex)
            .err()
            .unwrap_or_else(|| panic!("expected refusal containing {expected:?}, got Ok"));
        assert!(err.contains(expected), "expected {expected:?}, got: {err}");
    };

    // 1. Out-of-range lane 63 on [32, 1, 1] (`lone_lane::prepare` accepts `ConstInt(63)`).
    let [o63_0, o63_1] = readlane_const(7, 1, 63);
    assert_refused_despite_lone_lane(
        [32, 1, 1],
        &[
            ELECT_ATOMIC[0], ELECT_ATOMIC[1], ELECT_ATOMIC[2], ELECT_ATOMIC[3],
            0xbefe_04c1, // s_mov_b64 exec, -1
            o63_0, o63_1,
            0xbf81_0000,
        ],
        "out of range",
    );

    // 2. Unproven dynamic SGPR selector on Wave64 [32, 1, 1] (`count % 64 != 0`, so GROUP_READ
    //    fallback is unsafe and `lower_constant_reads` refuses it; `lone_lane::retry` must not swallow it).
    let [dyn0, dyn1] = readlane_sgpr(7, 1, 5);
    assert_refused_despite_lone_lane(
        [32, 1, 1],
        &[
            ELECT_ATOMIC[0], ELECT_ATOMIC[1], ELECT_ATOMIC[2], ELECT_ATOMIC[3],
            0xbefe_04c1, // s_mov_b64 exec, -1
            dyn0, dyn1,
            0xbf81_0000,
        ],
        "upper-bound proof",
    );

    // 3. Non-uniform loop enclosing v_readlane_b32 lane 31 on [64, 1, 1].
    let [r31_0, r31_1] = readlane_const(7, 1, 31);
    assert_refused_despite_lone_lane(
        [64, 1, 1],
        &[
            ELECT_ATOMIC[0], ELECT_ATOMIC[1], ELECT_ATOMIC[2], ELECT_ATOMIC[3],
            0xbefe_04c1, // s_mov_b64 exec, -1
            0x7e02_0300, // L_loop: v_mov_b32 v1, v0
            r31_0, r31_1,
            0xbf87_fffc, // s_cbranch_vccnz L_loop
            0xbf81_0000,
        ],
        "non-uniform loop",
    );

    // 4. Ambiguous 32-bit exec_lo op with v_readlane_b32 on [64, 1, 1].
    assert_refused_despite_lone_lane(
        [64, 1, 1],
        &[
            ELECT_ATOMIC[0], ELECT_ATOMIC[1], ELECT_ATOMIC[2], ELECT_ATOMIC[3],
            0xbefe_0300, // s_mov_b32 exec_lo, s0
            r31_0, r31_1,
            0xbf81_0000,
        ],
        "ambiguous wave width",
    );
}

#[test]
fn smem_loads_are_not_treated_as_workgroup_uniform_for_readlane_loops() {
    let [r0, r1] = readlane_const(7, 1, 31);
    for (label, load_words, expected_mnemonic) in [
        ("s_load_dword", [0xf400_0180u32, 0xfa00_0000], "s_load_dword"),
        ("s_buffer_load_dword", [0xf420_0180u32, 0xfa00_0000], "s_buffer_load_dword"),
    ] {
        let isa: Vec<u8> = load_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let ops = crate::rdna2::disassemble(&isa);
        assert_eq!(ops[0].mnemonic, expected_mnemonic, "{label}");

        let program = [
            load_words[0], load_words[1], // s_[buffer_]load_dword s6, s[0:1/3], 0 (NULL soffset)
            r0, r1,                       // L_loop: v_readlane_b32 s7, v1, 31
            0xbf07_8006,                  // s_cmp_lg_u32 s6, 0
            0xbf85_fffc,                  // s_cbranch_scc1 L_loop
            0xbf81_0000,
        ];
        let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &program, None))
            .err()
            .unwrap_or_else(|| panic!("{label}-conditioned loop must be refused as non-uniform"));
        assert!(err.contains("non-uniform loop"), "{label}: {err}");
    }
}

#[test]
fn non_readlane_failure_still_allows_lone_lane_retry_when_readlane_present() {
    // Election + atomic shader using `s_ff1_i32_b64 exec_lo` (whose primary path fails in
    // `mark_wave_allocations`, NOT in `indexed_lane::lower_constant_reads`) AND containing a
    // non-spill `v_readlane_b32 s7, v1, 0`. Disabling the blanket `has_non_spill_readlane`
    // early return lets `lone_lane::retry` rescue this shader just like on `main`.
    let [r0, r1] = readlane_const(7, 1, 0);
    let words = [
        0xbe88_147e,              // s_ff1_i32_b64 s8, exec_lo (fails verify_lane_election on primary)
        0x7da4_0008,              // v_cmpx_eq_u32 exec, s8, v0
        0xd800_0000, 0x0000_0403, // ds_add_u32 v3, v4
        0xbefe_04c1,              // s_mov_b64 exec, -1
        r0, r1,                   // v_readlane_b32 s7, v1, 0
        0xbf81_0000,
    ];
    let ex = compute_shader([64, 1, 1], &words, None);
    let primary = crate::hlsl::translate_outputs_once(&ex, None, &[], None);
    assert!(
        primary.is_err(),
        "primary path must fail on unverified election before lower_constant_reads"
    );
    let tr = crate::hlsl::translate(&ex)
        .expect("non-readlane primary failure must still fall back to lone_lane::retry");
    assert!(!tr.hlsl.contains("Wave"), "{}", tr.hlsl);
}
