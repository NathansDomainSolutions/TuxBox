use super::group;
use super::super::translate;
use crate::{ShaderExtract, Stage};

fn synthetic_resolve_wave32(words: &[u32]) -> ShaderExtract {
    let mut fssm = vec![0u8; 0x100];
    fssm[..8].copy_from_slice(b"FSSM\x01\x01\0\0");
    let rows = [
        [3u32, 0, 1, 0, 4],
        [3, 0, 1, 0, 0x105],
        [0, 0, 4, 1, 0x206],
        [5, 0, 1, 0, 10],
        [6, 0, 1, 0, 11],
    ];
    fssm[0xe8..0xec].copy_from_slice(&0xf0u32.to_le_bytes());
    fssm[0xf8..0xfc].copy_from_slice(&(rows.len() as u32).to_le_bytes());
    fssm.extend(rows.into_iter().flatten().flat_map(u32::to_le_bytes));
    let payload = fssm.len() as u32 - 16;
    fssm[8..12].copy_from_slice(&payload.to_le_bytes());
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        stage: Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config: None,
        ps09_pixel_inputs: None,
        fssm,
        entry: Some("CS_RaytracingResolve_Synthetic".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        psv_threads: None,
        flat_inputs: Vec::new(),
        flat_outputs: Vec::new(),
        isa_preamble: [0, 32, 1, 1],
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

fn synthetic_wave64(words: &[u32]) -> ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        stage: Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config: None,
        ps09_pixel_inputs: None,
        fssm: Vec::new(),
        entry: Some("main".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        psv_threads: None,
        flat_inputs: Vec::new(),
        flat_outputs: Vec::new(),
        isa_preamble: [0, 64, 1, 1],
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

#[test]
fn wave32_lane_bit_guards_coalesce_and_never_mix_with_1u() {
    let input = r#"    if (exec_lo & tb_lane_bit) v1 = asuint(exp2(asfloat(v0)));
    if (exec_lo & tb_lane_bit) v2 = asuint(cos(asfloat(v1)));
    if (exec_lo & 1u) v3 = asuint(log2(asfloat(v2)));
    if (exec_lo & 1u) v4 = countbits(v3);
"#;
    let expected = r#"    if (exec_lo & tb_lane_bit) {
        v1 = asuint(exp2(asfloat(v0)));
        v2 = asuint(cos(asfloat(v1)));
    }
    if (exec_lo & 1u) {
        v3 = asuint(log2(asfloat(v2)));
        v4 = countbits(v3);
    }
"#;
    assert_eq!(group(input), expected);

    // Interleaved 1u / tb_lane_bit must stay separate.
    let interleaved = r#"    if (exec_lo & 1u) v1 = v0 + 1u;
    if (exec_lo & tb_lane_bit) v2 = v1 + 2u;
    if (exec_lo & 1u) v3 = v2 + 3u;
"#;
    let interleaved_out = group(interleaved);
    assert_eq!(interleaved_out.matches("if (exec_lo & 1u) {").count(), 2);
    assert_eq!(interleaved_out.matches("if (exec_lo & tb_lane_bit) {").count(), 1);
}

#[test]
fn pure_intrinsics_and_pure_tb_helpers_coalesce() {
    let input = r#"    if (exec_lo & tb_lane_bit) v1 = asuint(exp2(asfloat(v0)));
    if (exec_lo & tb_lane_bit) v2 = asuint(log2(asfloat(v1)));
    if (exec_lo & tb_lane_bit) v3 = asuint(round(asfloat(v2)));
    if (exec_lo & tb_lane_bit) v4 = asuint(cos(asfloat(v3)));
    if (exec_lo & tb_lane_bit) v5 = asuint(clamp(asfloat(v4), 0.0, 1.0));
    if (exec_lo & tb_lane_bit) v6 = countbits(reversebits(v5));
    if (exec_lo & tb_lane_bit) v7 = (firstbitlow(v6) + (31u - firstbithigh(v6)));
    if (exec_lo & tb_lane_bit) v8 = f32tof16(asfloat(v5));
    if (exec_lo & tb_lane_bit) v9 = tb_perm(v7, v8, v6);
    if (exec_lo & tb_lane_bit) v10 = tb_mulhi_i32(asint(v9), asint(v8));
    if (exec_lo & tb_lane_bit) v11 = asuint(tb_cubema(asfloat(v1), asfloat(v2), asfloat(v3)));
    if (exec_lo & tb_lane_bit) v12 = asuint(tb_fract_f32(asfloat(v11)));
    if (exec_lo & tb_lane_bit) v13 = asuint(tb_ldexp_f32(asfloat(v12), asint(v6)));
    if (exec_lo & tb_lane_bit) v14 = tb_f16_rtz(asfloat(v13));
    if (exec_lo & tb_lane_bit) v15 = tb_quad_extrema(v14, v13, true, tb_fp_mode);
    if (exec_lo & tb_lane_bit) v16 = tb_dot4_i8(v15, v14, v13);
    if (exec_lo & tb_lane_bit) v17 = select(tb_bit(uint2(vcc_lo, vcc_hi), tb_lane), v15, v16);
    if (exec_lo & tb_lane_bit) v18 = uint((v1.x == 0u) ? 1u : 2u);
    if (exec_lo & tb_lane_bit) tb_s0 = (v18 & 0xffffu);
"#;
    let out = group(input);
    assert_eq!(out.matches("if (exec_lo & tb_lane_bit) {").count(), 1, "{out}");
    assert!(out.contains("tb_s0 = (v18 & 0xffffu);"), "{out}");
}

#[test]
fn unsafe_intrinsics_wave_ops_barriers_and_memory_never_coalesce() {
    let unsafe_lines = [
        "    if (exec_lo & tb_lane_bit) v3 = WaveActiveSum(v2);",
        "    if (exec_lo & tb_lane_bit) v3 = WaveReadLaneFirst(v2);",
        "    if (exec_lo & tb_lane_bit) v3 = WaveReadLaneAt(v2, 1u);",
        "    if (exec_lo & tb_lane_bit) v3 = QuadReadLaneAt(v2, 1u);",
        "    if (exec_lo & tb_lane_bit) v3 = QuadReadAcrossX(v2);",
        "    if (exec_lo & tb_lane_bit) v3 = tb_reduction_read(v2, 1u);",
        "    if (exec_lo & tb_lane_bit) v3 = tb_page_ballot(v2 != 0u);",
        "    if (exec_lo & tb_lane_bit) v3 = tb_page_read(v2, 0u);",
        "    if (exec_lo & tb_lane_bit) v3 = tb_group_ballot(v2 != 0u, tb_lane);",
        "    if (exec_lo & tb_lane_bit) v3 = tb_group_read(v2, 0u, tb_lane);",
        "    if (exec_lo & tb_lane_bit) v3 = tb_heap_index(s4, v2);",
        "    if (exec_lo & tb_lane_bit) v3 = raw_ubuf0.Load(v2);",
        "    if (exec_lo & tb_lane_bit) v3 = tex0.SampleLevel(samp0, float2(0.0, 0.0), 0.0);",
        "    if (exec_lo & tb_lane_bit) v3 = asuint(ddx(asfloat(v2)));",
        "    if (exec_lo & tb_lane_bit) v3 = asuint(ddy(asfloat(v2)));",
        "    if (exec_lo & tb_lane_bit) v3 = asuint(sincos(asfloat(v2), s0, s1));",
        "    if (exec_lo & tb_lane_bit) v3 = asuint(frexp(asfloat(v2), s0));",
        "    if (exec_lo & tb_lane_bit) v3 = asuint(modf(asfloat(v2), s0));",
        "    if (exec_lo & tb_lane_bit) v3 = lds[v2];",
        "    if (exec_lo & tb_lane_bit) InterlockedAdd(lds[0], v2);",
        "    if (exec_lo & tb_lane_bit) GroupMemoryBarrierWithGroupSync();",
        "    if (exec_lo & tb_lane_bit) AllMemoryBarrierWithGroupSync();",
        "    if (exec_lo & tb_lane_bit) DeviceMemoryBarrier();",
        "    if (exec_lo & tb_lane_bit) v3 = (v0.Load(0));",
        "    if (exec_lo & tb_lane_bit) v3 = v0.5;",
    ];
    for line in unsafe_lines {
        let snippet = format!(
            "    if (exec_lo & tb_lane_bit) v1 = v0 + 1u;\n{line}\n    if (exec_lo & tb_lane_bit) v2 = v1 + 2u;\n"
        );
        let out = group(&snippet);
        assert_eq!(out.matches("if (exec_lo & tb_lane_bit) {").count(), 2, "coalesced past unsafe line `{line}`:\n{out}");
        assert!(out.contains(&format!("{line}\n")), "unsafe line altered `{line}`:\n{out}");
    }
}

#[test]
fn guard_variable_mutations_break_the_group() {
    let mutating_lines = [
        "    exec_lo = s0;",
        "    exec_hi = s1;",
        "    tb_lane_bit = 1u << (v0 & 31u);",
        "    vcc_lo = (vcc_lo & ~exec_lo) | (exec_lo & 1u);",
        "    vcc_hi = 0u;",
        "    vcc = 0u;",
        "    scc = uint(s0 == 0u);",
        "    if (exec_lo & 1u) exec_lo = s0;",
        "    if (exec_lo & 1u) tb_lane_bit = 0u;",
        "    if (exec_lo & 1u) vcc_lo = s0;",
        "    if (exec_lo & 1u) vcc_hi = s0;",
        "    if (exec_lo & 1u) scc = 1u;",
        "    if (exec_lo & 1u) v3 = (exec_lo = 0u);",
        "    if (exec_lo & 1u) v3 = (vcc_lo += 1u);",
        "    if (exec_lo & 1u) v3 = (exec_lo >>= 1u);",
        "    if (exec_lo & 1u) v3 = (tb_lane_bit <<= 1u);",
        "    if (exec_lo & 1u) v3 = tb_lane_bit++;",
        "    uint t0 = (exec_lo = 1u);",
        "    uint t0 = (exec_lo >>= 1u);",
    ];
    for mutator in mutating_lines {
        let snippet = format!(
            "    if (exec_lo & 1u) v1 = v0 + 1u;\n{mutator}\n    if (exec_lo & 1u) v2 = v1 + 2u;\n"
        );
        let out = group(&snippet);
        assert_eq!(out.matches("if (exec_lo & 1u) {").count(), 2, "failed to break on `{mutator}`:\n{out}");
        assert!(out.contains(&format!("{mutator}\n")), "mutator altered `{mutator}`:\n{out}");
    }
    // Wave32 shift-assign repros must also refuse coalescing.
    let wave32_shift = r#"    if (exec_lo & tb_lane_bit) v1 = v0 + 1u;
    if (exec_lo & tb_lane_bit) v2 = (tb_lane_bit <<= 1u);
    if (exec_lo & tb_lane_bit) v3 = v1 + 2u;
"#;
    assert_eq!(group(wave32_shift).matches("if (exec_lo & tb_lane_bit) {").count(), 2);

    // Comparisons and shifts (`==`, `!=`, `<=`, `>=`, `<<`, `>>`) reading guard vars are pure and DO coalesce.
    let pure_cmp = r#"    if (exec_lo & tb_lane_bit) v1 = ((vcc_lo & tb_lane_bit) != 0u ? (v0 << 1u) : (v0 >> 1u));
    if (exec_lo & tb_lane_bit) v2 = (((v1 << 1u) == v0) ? 1u : 0u);
    if (exec_lo & tb_lane_bit) v3 = ((v1 <= v2 && v2 >= 0u) ? v1 : v2);
"#;
    assert_eq!(group(pure_cmp).matches("if (exec_lo & tb_lane_bit) {").count(), 1);
}

#[test]
fn block_comments_do_not_trap_or_corrupt_braces() {
    let masked_comment = r#"    /*
    if (exec_lo & 1u) v1 = v0 + 1u;
    // */
    if (exec_lo & 1u) v2 = v1 + 2u;
"#;
    assert_eq!(
        group(masked_comment),
        r#"    /*
    if (exec_lo & 1u) v1 = v0 + 1u;
    // */
    if (exec_lo & 1u) {
        v2 = v1 + 2u;
    }
"#
    );

    let split_comment = r#"    if (exec_lo & 1u) v1 = v0 + 1u;
    /* block
       comment */
    if (exec_lo & 1u) v2 = v1 + 2u;
"#;
    assert_eq!(group(split_comment).matches("if (exec_lo & 1u) {").count(), 2);

    // Single-line `// /*` must remain a regular comment and allow coalescing.
    let line_comment_with_slash_star = r#"    if (exec_lo & 1u) v1 = v0 + 1u;
    // /* ignored block opener inside line comment
    if (exec_lo & 1u) v2 = v1 + 2u;
"#;
    assert_eq!(
        group(line_comment_with_slash_star).matches("if (exec_lo & 1u) {").count(),
        1
    );

    // `precise` qualified outer declaration tracks shadowing cleanly.
    let precise_shadow = r#"    precise float t0 = raw_ubuf0.Load(0u);
    if (s0 != 0u) {
        float t0 = round(asfloat(v1));
        if (exec_lo & 1u) v2 = asuint(t0);
        if (exec_lo & 1u) v3 = v2 + 1u;
    }
"#;
    let out = group(precise_shadow);
    assert!(out.contains("        float t0 = round(asfloat(v1));\n        if (exec_lo & 1u) {\n"), "{out}");
}

#[test]
fn dead_outside_temps_coalesce_across_guarded_chains() {
    let input = r#"    float t0 = round(asfloat(v1));
    if (exec_lo & tb_lane_bit) v2 = asuint(t0);
    uint t1o = (0x3u) & 31u, t1w = (0x7u) & 31u;
    if (exec_lo & tb_lane_bit) v3 = (v2 >> t1o) & t1w;
    uint diff2 = v2 ^ v3;
    if (exec_lo & tb_lane_bit) v4 = diff2 + v3;
    uint t3l = v4 << 1u; uint t3h = v4 >> 31u;
    if (exec_lo & tb_lane_bit) v5 = t3l ^ t3h;
    uint tb_s4 = v5 & 0xffffu;
    if (exec_lo & tb_lane_bit) v6 = tb_s4;
    uint2 t5f64 = uint2(v5, v6);
    if (exec_lo & tb_lane_bit) v7 = t5f64.x + t5f64.y;
    // Comment mentioning t0 t1o diff2 t5f64 after the block is not a live use.
    raw_ubuf0.Store(v0, v7);
"#;
    let out = group(input);
    assert_eq!(out.matches("if (exec_lo & tb_lane_bit) {").count(), 1, "{out}");
    for temp in ["float t0 =", "uint t1o =", "uint diff2 =", "uint t3l =", "uint tb_s4 =", "uint2 t5f64 ="] {
        assert!(out.contains(&format!("        {temp}")), "missing absorbed `{temp}`:\n{out}");
    }
}

#[test]
fn sibling_blocks_may_reuse_dead_temp_name_independently() {
    let input = r#"    if (vcc_lo != 0u) {
        float t0 = round(asfloat(v1));
        if (exec_lo & 1u) v2 = asuint(t0);
        if (exec_lo & 1u) v3 = v2 + 1u;
    }
    if (scc != 0u) {
        float t0 = round(asfloat(v4));
        if (exec_lo & 1u) v5 = asuint(t0);
        if (exec_lo & 1u) v6 = v5 + 1u;
    }
"#;
    let out = group(input);
    assert_eq!(out.matches("if (exec_lo & 1u) {").count(), 2, "{out}");
    assert_eq!(out.matches("            float t0 =").count(), 2, "{out}");
}

#[test]
fn live_outside_temps_never_trapped_in_guarded_block() {
    // 1. Live on immediate next line.
    let immediate = r#"    if (exec_lo & 1u) v1 = v0 + 1u;
    float t0 = round(asfloat(v1));
    if (exec_lo & 1u) v2 = asuint(t0);
    v3 = asuint(t0);
"#;
    let immediate_out = group(immediate);
    assert_eq!(
        immediate_out,
        r#"    if (exec_lo & 1u) {
        v1 = v0 + 1u;
    }
    float t0 = round(asfloat(v1));
    if (exec_lo & 1u) {
        v2 = asuint(t0);
    }
    v3 = asuint(t0);
"#
    );

    // 2. Leading TempDecl that is live outside emits verbatim while following guarded ops coalesce.
    let leading_live = r#"    float t0 = round(asfloat(v1));
    if (exec_lo & 1u) v2 = asuint(t0);
    if (exec_lo & 1u) v3 = v2 + 1u;
    raw_ubuf0.Store(0u, asuint(t0));
"#;
    assert_eq!(
        group(leading_live),
        r#"    float t0 = round(asfloat(v1));
    if (exec_lo & 1u) {
        v2 = asuint(t0);
        v3 = v2 + 1u;
    }
    raw_ubuf0.Store(0u, asuint(t0));
"#
    );

    // 3. Live inside a later inner `{ ... }` child block of the same enclosing scope.
    let inner_use = r#"    float t0 = round(asfloat(v1));
    if (exec_lo & 1u) v2 = asuint(t0);
    if (s0 != 0u) {
        v3 = asuint(t0);
    }
"#;
    assert_eq!(
        group(inner_use),
        r#"    float t0 = round(asfloat(v1));
    if (exec_lo & 1u) {
        v2 = asuint(t0);
    }
    if (s0 != 0u) {
        v3 = asuint(t0);
    }
"#
    );

    // 4. Live after an early return / inside return.
    let early_return = r#"    uint t0 = v1 + 1u;
    if (exec_lo & 1u) v2 = t0;
    if (s0 == 0u) return;
    v3 = t0;
"#;
    assert_eq!(
        group(early_return),
        r#"    uint t0 = v1 + 1u;
    if (exec_lo & 1u) {
        v2 = t0;
    }
    if (s0 == 0u) return;
    v3 = t0;
"#
    );

    // 5. Transitive shrink: t1 is live outside -> shrinks end before t1 -> exposes t0 used after t1!
    let transitive = r#"    uint t0 = v0 + 1u;
    if (exec_lo & 1u) v1 = t0;
    uint t1 = v1 + 2u;
    if (exec_lo & 1u) v2 = t1 + t0;
    raw_ubuf0.Store(0u, t1);
"#;
    assert_eq!(
        group(transitive),
        r#"    uint t0 = v0 + 1u;
    if (exec_lo & 1u) {
        v1 = t0;
    }
    uint t1 = v1 + 2u;
    if (exec_lo & 1u) {
        v2 = t1 + t0;
    }
    raw_ubuf0.Store(0u, t1);
"#
    );

    // 6. Shadowing outer decl, use-before-decl, and uninitialized decl never absorbed.
    let shadowing = r#"    uint t0 = raw_sbuf0.Load(0u);
    if (s0 != 0u) {
        uint t0 = v1 + 1u;
        if (exec_lo & 1u) v2 = t0;
        if (exec_lo & 1u) v3 = v2 + 1u;
    }
"#;
    let shadow_out = group(shadowing);
    assert!(shadow_out.contains("        uint t0 = v1 + 1u;\n        if (exec_lo & 1u) {\n"), "{shadow_out}");

    let use_before_decl = r#"    if (exec_lo & 1u) v1 = t0;
    uint t0 = v2 + 1u;
    if (exec_lo & 1u) v3 = t0;
"#;
    assert_eq!(group(use_before_decl).matches("if (exec_lo & 1u) {").count(), 2);

    let uninit = r#"    uint t0, t1;
    if (exec_lo & 1u) v1 = v0 + 1u;
    if (exec_lo & 1u) v2 = v1 + 2u;
"#;
    assert!(group(uninit).starts_with("    uint t0, t1;\n    if (exec_lo & 1u) {\n"));
}

#[test]
fn real_translate_coalesces_wave32_and_wave64_intrinsics_and_temps() {
    let mut wave32_words = vec![
        0xd4c4_0003, 0x0201_0100, // v_cmp_gt_u32 s3, v0, 0
        0x7e02_0281,              // v_mov_b32 v1, 1
        0x7e04_02f2,              // v_mov_b32 v2, 1.0
    ];
    let body_ops = [
        vec![0x7e06_6b02],             // v_sin_f32 v3, v2
        vec![0x7e08_6d03],             // v_cos_f32 v4, v3
        vec![0x7e0a_4b04],             // v_exp_f32 v5, v4
        vec![0x7e0c_4f05],             // v_log_f32 v6, v5
        vec![0x7e0e_5d02],             // v_rsq_f32 v7, v2
        vec![0x7e10_6702],             // v_sqrt_f32 v8, v2
        vec![0x7e12_4707],             // v_rndne_f32 v9, v7
        vec![0x7e14_1908],             // v_cvt_rpi_i32_f32 v10, v8 (temp float tN)
        vec![0xd549_000b, 0x021d_0701], // v_bfe_i32 v11, v1, 3, 7 (temp uint tNo, tNw)
        vec![0xd744_000c, 0x042e_1909], // v_perm_b32 v12, v9, v12, v11 (tb_perm)
        vec![0xd54d_0001, 0x0432_1901], // v_lerp_u8 v1, v1, v12, v12 (temp uint diffN)
        vec![0xd764_000d, 0x0202_0301], // v_bcnt_u32_b32 v13, v1, v1 (countbits)
        vec![0x7e1c_750d],             // v_ffbl_b32 v14, v13 (firstbitlow)
        vec![0x7e1e_730d],             // v_ffbh_u32 v15, v13 (firstbithigh)
    ];
    for _ in 0..4 {
        for op in &body_ops {
            wave32_words.extend_from_slice(op);
        }
    }
    wave32_words.push(0xbf81_0000); // s_endpgm

    let t32 = translate(&synthetic_resolve_wave32(&wave32_words)).expect("translate wave32");
    assert_eq!(t32.required_wave_size, Some(32));
    assert_eq!(
        t32.hlsl.matches("if (exec_lo & tb_lane_bit) {").count(),
        1,
        "expected all wave32 intrinsics and dead temps in 1 block:\n{}",
        t32.hlsl
    );
    assert_eq!(t32.hlsl.matches("if (exec_lo & tb_lane_bit)").count(), 1);
    assert_eq!(t32.hlsl.matches("if (exec_lo & 1u)").count(), 0);

    let mut wave64_words = vec![0x7e02_0281, 0x7e04_02f2];
    for _ in 0..4 {
        for op in &body_ops {
            wave64_words.extend_from_slice(op);
        }
    }
    wave64_words.push(0xbf81_0000);
    let t64 = translate(&synthetic_wave64(&wave64_words)).expect("translate wave64");
    assert_eq!(
        t64.hlsl.matches("if (exec_lo & 1u) {").count(),
        1,
        "expected all wave64 intrinsics and dead temps in 1 block:\n{}",
        t64.hlsl
    );
    assert_eq!(t64.hlsl.matches("if (exec_lo & 1u)").count(), 1);
}
