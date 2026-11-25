use crate::{hlsl, ShaderExtract, Stage};

fn extract(words: &[u32], group: [u32; 3]) -> ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract { flat_inputs: Vec::new(), flat_outputs: Vec::new(),
        stage: Stage::Compute, shader_model: (6, 0), required_features: None,
        bitcode_stripped: true, fspx_config: None, ps09_pixel_inputs: None,
        fssm: Vec::new(), entry: Some("main".into()), isa_data: isa.clone(),
        isa, isa_offset_in_fssm: 0, psv_threads: None, isa_preamble: [0, group[0], group[1], group[2]],
        has_input_sig: false, has_output_sig: false, has_psv0: false, has_root_sig: false,
        input_sig: Vec::new(), output_sig: Vec::new(), resources: Vec::new(),
        gs_info: None, mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
    }
}

// s_mov_b32 s2, 0x76543210; s_mov_b32 s3, 0xfedcba98; <body>; s_endpgm
fn program(body: &[u32]) -> Vec<u32> {
    let mut words = vec![0xbe82_03ff, 0x7654_3210, 0xbe83_03ff, 0xfedc_ba98];
    words.extend(body);
    words.push(0xbf81_0000);
    words
}

fn translate(words: &[u32], group: [u32; 3]) -> Result<hlsl::Translated, String> {
    hlsl::translate(&extract(words, group))
}

// LLVM gfx1030 words.
pub(super) const FORMS: [(&str, [u32; 2]); 8] = [
    ("v_permlane16_b32 v5, v4, s2, s3", [0xd777_0005, 0x000c_0504]),
    ("v_permlane16_b32 v5, v4, s2, s3 op_sel:[1,0]", [0xd777_0805, 0x000c_0504]),
    ("v_permlane16_b32 v5, v4, s2, s3 op_sel:[0,1]", [0xd777_1005, 0x000c_0504]),
    ("v_permlane16_b32 v5, v4, s2, s3 op_sel:[1,1]", [0xd777_1805, 0x000c_0504]),
    ("v_permlanex16_b32 v5, v4, s2, s3", [0xd778_0005, 0x000c_0504]),
    ("v_permlanex16_b32 v5, v4, s2, s3 op_sel:[1,0]", [0xd778_0805, 0x000c_0504]),
    ("v_permlanex16_b32 v5, v4, s2, s3 op_sel:[0,1]", [0xd778_1005, 0x000c_0504]),
    ("v_permlanex16_b32 v5, v4, s2, s3 op_sel:[1,1]", [0xd778_1805, 0x000c_0504]),
];

#[test]
fn permlane_forms_translate_in_complete_wave32_compute() {
    for (asm, words) in FORMS {
        let out = translate(&program(&words), [64, 1, 1]).unwrap_or_else(|e| panic!("{asm}: {e}"));
        let fi = asm.contains("op_sel:[1,");
        let bc = asm.ends_with(",1]");
        let legacy = asm == "v_permlanex16_b32 v5, v4, s2, s3";
        assert!(out.hlsl.contains("WaveReadLaneAt(v4, "), "{asm}");
        let row = if asm.starts_with("v_permlanex16") { "& 16u) ^ 16u)" } else { "src = (t" };
        if !legacy {
            assert_eq!(out.required_wave_size, Some(32), "{asm}");
            assert!(out.hlsl.contains(row), "{asm}\n{}", out.hlsl);
            assert_eq!(out.hlsl.contains("WaveReadLaneAt(exec_lo & 1u"), !fi, "{asm}");
            assert_eq!(out.hlsl.contains("value : 0u"), !fi && bc, "{asm}");
        }
    }
}

#[test]
fn plain_permlanex16_keeps_its_established_lowering() {
    // No new wave-size requirement for the existing op_sel 0 form.
    let out = translate(&program(&[0xd778_0005, 0x000c_0504]), [64, 1, 1]).unwrap();
    assert_eq!(out.required_wave_size, None);
    assert!(out.hlsl.contains("WaveActiveBallot((exec_lo & 1u) != 0u)"));
}

#[test]
fn permlane_accepts_literal_and_inline_selectors() {
    for (asm, words) in [
        ("v_permlane16_b32 v5, v4, 0x76543210, s3", vec![0xd777_0005, 0x000d_ff04, 0x7654_3210]),
        ("v_permlane16_b32 v5, v4, -1, 0", vec![0xd777_0005, 0x0201_8304]),
        ("v_permlane16_b32 v4, v4, s2, s3", vec![0xd777_0004, 0x000c_0504]),
    ] {
        let out = translate(&program(&words), [32, 1, 1]).unwrap_or_else(|e| panic!("{asm}: {e}"));
        assert!(out.hlsl.contains("WaveReadLaneAt(v4, "), "{asm}");
    }
}

#[test]
fn permlane_refuses_unverified_shapes() {
    let plain = [0xd777_0005, 0x000c_0504];
    for (what, words, group) in [
        // Lane masks as selectors: EXEC and VCC hold one bit per invocation here.
        ("exec selector", program(&[0xd777_0005, 0x000c_fd04]), [64, 1, 1]), // v_permlane16_b32 v5, v4, exec_lo, s3
        ("vcc selector", program(&[0xd777_0005, 0x000c_d504]), [64, 1, 1]),  // v_permlane16_b32 v5, v4, vcc_lo, s3
        ("m0 selector", program(&[0xd777_0005, 0x000c_f904]), [64, 1, 1]),   // v_permlane16_b32 v5, v4, m0, s3
        // No reaching definition of the selector registers.
        ("unknown selectors", [plain.as_slice(), &[0xbf81_0000]].concat(), [64, 1, 1]),
        // A compare result copied into s2 is a lane mask.
        ("compare selector", program(&[0x7d84_0880, 0xbe82_036a, 0xd777_0005, 0x000c_0504]), [64, 1, 1]), // v_cmp_eq_u32 vcc_lo, 0, v4; s_mov_b32 s2, vcc_lo
        // Incomplete 32-lane waves.
        ("partial wave", program(&plain), [16, 1, 1]),
        ("partial wave 48", program(&plain), [48, 1, 1]),
        // Divergent control flow around the exchange.
        ("branch", program(&[0xbf88_0002, 0xd777_0005, 0x000c_0504]), [64, 1, 1]), // s_cbranch_execz
        // op_sel bits other than [fi, bound_ctrl].
        ("op_sel dst", program(&[0xd777_4005, 0x000c_0504]), [64, 1, 1]),
        ("op_sel src2", program(&[0xd777_2005, 0x000c_0504]), [64, 1, 1]),
        ("op_sel x16 dst", program(&[0xd778_4005, 0x000c_0504]), [64, 1, 1]),
        // op_sel[0] and op_sel[3], seen in natively compiled cull shaders.
        ("op_sel x16 fi dst", program(&[0xd778_4805, 0x000c_0504]), [64, 1, 1]),
        // Source modifiers are not part of the instruction.
        ("neg src0", program(&[0xd777_0005, 0x200c_0504]), [64, 1, 1]),
    ] {
        let err = translate(&words, group).err().unwrap_or_else(|| panic!("{what}"));
        println!("REFUSED {what}: {err}");
    }
}

#[test]
fn permlane_lane_model_fixture() {
    // Consumed by the external 32-lane Python model, which evaluates these
    // statements per lane with random EXEC masks and compares them with
    // the ISA rules.
    for (asm, words) in FORMS {
        let out = translate(&program(&words), [64, 1, 1]).unwrap();
        let start = out.hlsl.find("// 0x0010").expect("permlane body");
        let end = out.hlsl[start..].find("// 0x0018").map_or(out.hlsl.len(), |e| start + e);
        println!("BEGIN_PERMLANE {asm}\n{}END_PERMLANE", &out.hlsl[start..end]);
    }
}
