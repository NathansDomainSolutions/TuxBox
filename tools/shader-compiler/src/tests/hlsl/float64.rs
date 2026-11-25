use crate::{hlsl, ShaderExtract, Stage};

const V4: [u32; 2] = [0x7e08_02ff, 0x1357_2468];
const V5: [u32; 2] = [0x7e0a_02ff, 0x2468_1357];
const ENDPGM: u32 = 0xbf81_0000;

/// Translate `v_mov_b32 v4/v5, literal`, then `body`, then `s_endpgm`.
pub(in crate::hlsl) fn translate(stage: Stage, body: &[u32]) -> Result<String, String> {
    let words: Vec<u32> = V4.iter().chain(&V5).chain(body).chain(&[ENDPGM]).copied().collect();
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let ex = ShaderExtract { flat_inputs: Vec::new(), flat_outputs: Vec::new(),
        stage, shader_model: (6, 0), required_features: None, bitcode_stripped: true,
        fspx_config: None, ps09_pixel_inputs: None, fssm: Vec::new(), entry: Some("main".into()),
        isa_data: isa.clone(), isa, isa_offset_in_fssm: 0, psv_threads: None, isa_preamble: [0, 1, 1, 1],
        has_input_sig: false, has_output_sig: false, has_psv0: false, has_root_sig: false,
        input_sig: Vec::new(), output_sig: Vec::new(), resources: Vec::new(), gs_info: None,
        mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
    };
    hlsl::translate(&ex).map(|t| t.hlsl)
}

#[test]
fn pair_forms_write_both_destination_words() {
    // LLVM gfx1030: v_floor_f64_e32 v[8:9], v[4:5]
    let hlsl = translate(Stage::Compute, &[0x7e10_3504]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(v4, v5), 1u, true)"), "{hlsl}");
    assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
    // v_rndne_f64_e64 v[8:9], v[4:5]
    assert!(translate(Stage::Compute, &[0xd599_0008, 0x0201_0104]).unwrap().contains(", 3u, true)"));
    // v_cvt_i32_f64_e32 v8, v[4:5]
    assert!(translate(Stage::Compute, &[0x7e10_0704]).unwrap().contains("tb_f64_to_i32(uint2(v4, v5), true)"));
    // v_cvt_f64_u32_e32 v[8:9], v4
    assert!(translate(Stage::Compute, &[0x7e10_2d04]).unwrap().contains("tb_i32_to_f64(v4, false)"));
}

#[test]
fn f32_conversions_round_and_follow_the_f32_denormal_mode() {
    // LLVM gfx1030: v_cvt_f64_f32_e32 v[8:9], v4
    let hlsl = translate(Stage::Compute, &[0x7e10_2104]).unwrap();
    assert!(hlsl.contains("= tb_f32_to_f64(v4, false);"), "{hlsl}");
    assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
    // v_cvt_f32_f64_e32 v8, v[4:5]
    assert!(translate(Stage::Compute, &[0x7e10_1f04]).unwrap().contains("v8 = tb_f64_to_f32(uint2(v4, v5), false)"));
    // s_denorm_mode 0 makes the mode dynamic.
    let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0x7e10_2104, 0x7e14_1f04]).unwrap();
    assert!(hlsl.contains("tb_f32_to_f64(v4, (tb_fp_mode & 0x10u) != 0u)"), "{hlsl}");
    assert!(hlsl.contains("v10 = tb_f64_to_f32(uint2(v4, v5), (tb_fp_mode & 0x20u) != 0u)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_f64_f32_e64 v[8:9], -v4 (neg flips the f32 sign bit)
    let hlsl = translate(Stage::Compute, &[0xd590_0008, 0x2000_0104]).unwrap();
    assert!(hlsl.contains("= tb_f32_to_f64((v4 ^ 0x80000000u), false);"), "{hlsl}");
    // Clamp stays refused: v_cvt_f32_f64_e64 v8, v[4:5] clamp
    assert!(translate(Stage::Compute, &[0xd58f_8008, 0x0000_0104]).is_err());
}

#[test]
fn denormal_mode_controls_input_flushing() {
    // s_denorm_mode 0, then v_floor_f64_e32 v[8:9], v[4:5]
    let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0x7e10_3504]).unwrap();
    assert!(hlsl.contains("(tb_fp_mode & 0x40u) != 0u"), "{hlsl}");
}

#[test]
fn arithmetic_forms_call_soft_float_helpers() {
    // LLVM gfx1030: v_add_f64 v[8:9], v[4:5], v[6:7]
    let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x0002_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_add(uint2(v4, v5), uint2(v6, v7), true, true);"), "{hlsl}");
    assert!(hlsl.contains("uint2 tb_f64_add(uint2 a, uint2 b, bool keepIn, bool keepOut)"));
    assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
    assert!(!hlsl.contains("double"));
    // Both halves of the unwritten second pair are read on entry.
    assert!(hlsl.lines().next().unwrap().contains(" v6 v7"), "{hlsl}");
    // LLVM gfx1030: v_mul_f64 v[8:9], v[4:5], v[6:7]
    let hlsl = translate(Stage::Compute, &[0xd565_0008, 0x0002_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_mul(uint2(v4, v5), uint2(v6, v7), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_fma_f64 v[8:9], v[4:5], v[6:7], v[4:5]
    let hlsl = translate(Stage::Compute, &[0xd54c_0008, 0x0412_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_fma(uint2(v4, v5), uint2(v6, v7), uint2(v4, v5), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_ldexp_f64 v[8:9], v[4:5], v6
    let hlsl = translate(Stage::Compute, &[0xd568_0008, 0x0002_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_ldexp(uint2(v4, v5), asint(v6), true, true);"), "{hlsl}");
    // The rounding and conversion helpers alone do not pull in soft float.
    assert!(!translate(Stage::Compute, &[0x7e10_3504]).unwrap().contains("tb_f64_add"));
}

#[test]
fn arithmetic_source_modifiers_flip_or_clear_the_sign_bit() {
    // LLVM gfx1030: v_add_f64 v[8:9], v[4:5], -v[6:7]
    let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x4002_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_add(uint2(v4, v5), uint2(v6, (v7 ^ 0x80000000u)), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_add_f64 v[8:9], |v[4:5]|, -|v[6:7]|
    let hlsl = translate(Stage::Compute, &[0xd564_0308, 0x4002_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_add(uint2(v4, (v5 & 0x7fffffffu)), uint2(v6, ((v7 & 0x7fffffffu) ^ 0x80000000u)), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_fma_f64 v[8:9], -v[4:5], v[6:7], -v[2:3]
    let hlsl = translate(Stage::Compute, &[0xd54c_0008, 0xa40a_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_fma(uint2(v4, (v5 ^ 0x80000000u)), uint2(v6, v7), uint2(v2, (v3 ^ 0x80000000u)), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_ldexp_f64 v[8:9], -v[4:5], s6
    let hlsl = translate(Stage::Compute, &[0xd568_0008, 0x2000_0d04]).unwrap();
    assert!(hlsl.contains("tb_f64_ldexp(uint2(v4, (v5 ^ 0x80000000u)), asint(s6), true, true);"), "{hlsl}");
}

#[test]
fn arithmetic_accepts_scalar_pairs_and_constants() {
    // LLVM gfx1030: v_add_f64 v[8:9], s[4:5], v[6:7]
    let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x0002_0c04]).unwrap();
    assert!(hlsl.contains("tb_f64_add(uint2(s4, s5), uint2(v6, v7), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_mul_f64 v[8:9], 2.0, v[4:5]
    let hlsl = translate(Stage::Compute, &[0xd565_0008, 0x0002_08f4]).unwrap();
    assert!(hlsl.contains("tb_f64_mul(uint2(0x0u, 0x40000000u), uint2(v4, v5), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_mul_f64 v[8:9], 0x40080000, v[4:5] (the literal is the high word)
    let hlsl = translate(Stage::Compute, &[0xd565_0008, 0x0002_08ff, 0x4008_0000]).unwrap();
    assert!(hlsl.contains("tb_f64_mul(uint2(0x0u, 0x40080000u), uint2(v4, v5), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_add_f64 v[8:9], -1, v[4:5] (integer constants are 64-bit integers)
    let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x0002_08c1]).unwrap();
    assert!(hlsl.contains("tb_f64_add(uint2(0xffffffffu, 0xffffffffu), uint2(v4, v5), true, true);"), "{hlsl}");
    // LLVM gfx1030: v_fma_f64 v[8:9], v[4:5], v[6:7], 1.0
    let hlsl = translate(Stage::Compute, &[0xd54c_0008, 0x03ca_0d04]).unwrap();
    assert!(hlsl.contains("uint2(0x0u, 0x3ff00000u), true, true);"), "{hlsl}");
}

#[test]
fn arithmetic_follows_the_f64_denormal_mode() {
    // s_denorm_mode 0, then v_add_f64 v[8:9], v[4:5], v[6:7]
    let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0xd564_0008, 0x0002_0d04]).unwrap();
    assert!(hlsl.contains("uint2(v6, v7), (tb_fp_mode & 0x40u) != 0u, (tb_fp_mode & 0x80u) != 0u);"), "{hlsl}");
}

#[test]
fn arithmetic_output_modifiers_and_unmodelled_forms_are_refused() {
    // LLVM gfx1030: v_add_f64 v[8:9], v[4:5], v[6:7] clamp
    assert!(translate(Stage::Compute, &[0xd564_8008, 0x0002_0d04]).is_err());
    // LLVM gfx1030: v_mul_f64 v[8:9], v[4:5], v[6:7] mul:2
    assert!(translate(Stage::Compute, &[0xd565_0008, 0x0802_0d04]).is_err());
    // v_add_f64 v[8:9], v[4:5], v[6:7] with op_sel bit 11 set by hand
    assert!(translate(Stage::Compute, &[0xd564_0808, 0x0002_0d04]).is_err());
    // v_ldexp_f64 v[8:9], v[4:5], -v6 (llvm-mc rejects it; neg bit set by hand)
    assert!(translate(Stage::Compute, &[0xd568_0008, 0x4002_0d04]).is_err());
    // LLVM gfx1030: v_ldexp_f64 v[8:9], v[4:5], 2.0 (an inline float as the integer exponent)
    assert!(translate(Stage::Compute, &[0xd568_0008, 0x0001_e904]).is_err());
    // LLVM gfx1030: v_min_f64 v[8:9], v[4:5], v[6:7] and v_max_f64 (not modelled)
    assert!(translate(Stage::Compute, &[0xd566_0008, 0x0002_0d04]).is_err());
    assert!(translate(Stage::Compute, &[0xd567_0008, 0x0002_0d04]).is_err());
    assert!(translate(Stage::Pixel, &[0xd564_0008, 0x0002_0d04]).is_err());
}

#[test]
fn rounding_and_converts_accept_scalar_pairs_constants_and_literals() {
    // LLVM gfx1030: v_floor_f64_e32 v[8:9], s[4:5]  ; encoding: [0x04,0x34,0x10,0x7e]
    let hlsl = translate(Stage::Compute, &[0x7e10_3404]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(s4, s5), 1u, true)"), "{hlsl}");
    // LLVM gfx1030: v_ceil_f64_e32 v[8:9], 0.5  ; encoding: [0xf0,0x30,0x10,0x7e]
    let hlsl = translate(Stage::Compute, &[0x7e10_30f0]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(0x0u, 0x3fe00000u), 2u, true)"), "{hlsl}");
    // LLVM gfx1030: v_rndne_f64_e32 v[8:9], 0x3ff80000 (the literal is the high word)
    let hlsl = translate(Stage::Compute, &[0x7e10_32ff, 0x3ff8_0000]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(0x0u, 0x3ff80000u), 3u, true)"), "{hlsl}");
    // LLVM gfx1030: v_trunc_f64_e32 v[8:9], -1 (a 64-bit integer pattern)
    let hlsl = translate(Stage::Compute, &[0x7e10_2ec1]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(0xffffffffu, 0xffffffffu), 0u, true)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_u32_f64_e32 v8, vcc  ; encoding: [0x6a,0x2a,0x10,0x7e]
    let hlsl = translate(Stage::Compute, &[0x7e10_2a6a]).unwrap();
    assert!(hlsl.contains("tb_f64_to_i32(uint2(vcc_lo, vcc_hi), false)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_f64_i32_e32 v[8:9], s4  ; encoding: [0x04,0x08,0x10,0x7e]
    let hlsl = translate(Stage::Compute, &[0x7e10_0804]).unwrap();
    assert!(hlsl.contains("tb_i32_to_f64(s4, true)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_f64_u32_e32 v[8:9], 0xaf123456
    let hlsl = translate(Stage::Compute, &[0x7e10_2cff, 0xaf12_3456]).unwrap();
    assert!(hlsl.contains("tb_i32_to_f64(0xaf123456u, false)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_f64_i32_e32 v[8:9], -4.0 (the f32 bit pattern as an integer)
    let hlsl = translate(Stage::Compute, &[0x7e10_08f7]).unwrap();
    assert!(hlsl.contains("tb_i32_to_f64(0xc0800000u, true)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_f64_f32_e32 v[8:9], m0  ; encoding: [0x7c,0x20,0x10,0x7e]
    let hlsl = translate(Stage::Compute, &[0x7e10_207c]).unwrap();
    assert!(hlsl.contains("tb_f32_to_f64(m0, false)"), "{hlsl}");
}

#[test]
fn rounding_and_converts_apply_source_abs_and_neg() {
    // LLVM gfx1030: v_floor_f64_e64 v[8:9], -v[4:5]
    let hlsl = translate(Stage::Compute, &[0xd59a_0008, 0x2000_0104]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(v4, (v5 ^ 0x80000000u)), 1u, true)"), "{hlsl}");
    // LLVM gfx1030: v_trunc_f64_e64 v[8:9], |v[4:5]|
    let hlsl = translate(Stage::Compute, &[0xd597_0108, 0x0000_0104]).unwrap();
    assert!(hlsl.contains("tb_f64_round(uint2(v4, (v5 & 0x7fffffffu)), 0u, true)"), "{hlsl}");
    // LLVM gfx1030: v_cvt_i32_f64_e64 v8, -|v[4:5]|
    let hlsl = translate(Stage::Compute, &[0xd583_0108, 0x2000_0104]).unwrap();
    assert!(hlsl.contains("tb_f64_to_i32(uint2(v4, ((v5 & 0x7fffffffu) ^ 0x80000000u)), true)"), "{hlsl}");
}

#[test]
fn modifiers_and_other_stages_are_refused() {
    // LLVM gfx1030: v_floor_f64_e64 v[8:9], v[4:5] clamp
    assert!(translate(Stage::Compute, &[0xd59a_8008, 0x0000_0104]).is_err());
    // LLVM gfx1030: v_floor_f64_e64 v[8:9], v[4:5] mul:2
    assert!(translate(Stage::Compute, &[0xd59a_0008, 0x0800_0104]).is_err());
    // LLVM gfx1030: v_cvt_f64_i32_e64 v[8:9], v4 clamp
    assert!(translate(Stage::Compute, &[0xd584_8008, 0x0000_0104]).is_err());
    // v_cvt_f64_i32_e64 v[8:9], -v4 (llvm-mc rejects it; neg bit set by hand on an integer source)
    assert!(translate(Stage::Compute, &[0xd584_0008, 0x2000_0104]).is_err());
    // v_floor_f64_e64 v[8:9], v[4:5] with op_sel bit 11 set by hand
    assert!(translate(Stage::Compute, &[0xd59a_0808, 0x0000_0104]).is_err());
    assert!(translate(Stage::Pixel, &[0x7e10_3504]).is_err());
}

#[test]
fn fract_and_frexp_call_soft_float_helpers() {
    // LLVM gfx1030: v_fract_f64_e32 v[8:9], v[4:5]  ; encoding: [0x04,0x7d,0x10,0x7e]
    let hlsl = translate(Stage::Compute, &[0x7e10_7d04]).unwrap();
    assert!(hlsl.contains("= tb_f64_fract(uint2(v4, v5), true, true);"), "{hlsl}");
    assert!(hlsl.contains("uint2 tb_f64_fract(uint2 x, bool keepIn, bool keepOut)"), "{hlsl}");
    assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
    // LLVM gfx1030: v_fract_f64_e64 v[8:9], -|v[4:5]|
    let hlsl = translate(Stage::Compute, &[0xd5be_0108, 0x2000_0104]).unwrap();
    assert!(hlsl.contains("tb_f64_fract(uint2(v4, ((v5 & 0x7fffffffu) ^ 0x80000000u)), true, true)"), "{hlsl}");
    // LLVM gfx1030: v_fract_f64_e32 v[8:9], s[4:5]
    let hlsl = translate(Stage::Compute, &[0x7e10_7c04]).unwrap();
    assert!(hlsl.contains("tb_f64_fract(uint2(s4, s5), true, true)"), "{hlsl}");
    // LLVM gfx1030: v_frexp_mant_f64_e32 v[8:9], v[4:5]
    let hlsl = translate(Stage::Compute, &[0x7e10_7b04]).unwrap();
    assert!(hlsl.contains("= tb_f64_frexp_mant(uint2(v4, v5), true);"), "{hlsl}");
    assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
    // LLVM gfx1030: v_frexp_mant_f64_e64 v[8:9], -v[4:5]
    let hlsl = translate(Stage::Compute, &[0xd5bd_0008, 0x2000_0104]).unwrap();
    assert!(hlsl.contains("tb_f64_frexp_mant(uint2(v4, (v5 ^ 0x80000000u)), true)"), "{hlsl}");
    // LLVM gfx1030: v_frexp_exp_i32_f64_e32 v8, v[4:5] (one result register)
    let hlsl = translate(Stage::Compute, &[0x7e10_7904]).unwrap();
    assert!(hlsl.contains("v8 = tb_f64_frexp_exp(uint2(v4, v5), true)"), "{hlsl}");
    assert!(!hlsl.contains("v9 ="), "{hlsl}");
    // LLVM gfx1030: v_frexp_exp_i32_f64_e64 v8, |v[4:5]|
    let hlsl = translate(Stage::Compute, &[0xd5bc_0108, 0x0000_0104]).unwrap();
    assert!(hlsl.contains("v8 = tb_f64_frexp_exp(uint2(v4, (v5 & 0x7fffffffu)), true)"), "{hlsl}");
    // LLVM gfx1030: v_frexp_exp_i32_f64_e32 v8, 1.0
    let hlsl = translate(Stage::Compute, &[0x7e10_78f2]).unwrap();
    assert!(hlsl.contains("v8 = tb_f64_frexp_exp(uint2(0x0u, 0x3ff00000u), true)"), "{hlsl}");
    // s_denorm_mode 0, then v_fract_f64_e32 v[8:9], v[4:5] and v_frexp_exp_i32_f64_e32 v10, v[4:5]
    let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0x7e10_7d04, 0x7e14_7904]).unwrap();
    assert!(hlsl.contains("tb_f64_fract(uint2(v4, v5), (tb_fp_mode & 0x40u) != 0u, (tb_fp_mode & 0x80u) != 0u)"), "{hlsl}");
    assert!(hlsl.contains("tb_f64_frexp_exp(uint2(v4, v5), (tb_fp_mode & 0x40u) != 0u)"), "{hlsl}");
}

#[test]
fn fract_frexp_modifiers_and_unpinned_f64_ops_are_refused() {
    // LLVM gfx1030: v_fract_f64_e64 v[8:9], v[4:5] clamp
    assert!(translate(Stage::Compute, &[0xd5be_8008, 0x0000_0104]).is_err());
    // LLVM gfx1030: v_frexp_mant_f64_e64 v[8:9], v[4:5] mul:2
    assert!(translate(Stage::Compute, &[0xd5bd_0008, 0x0800_0104]).is_err());
    assert!(translate(Stage::Pixel, &[0x7e10_7d04]).is_err());
    // LLVM gfx1030: v_rcp_f64_e32 v[8:9], v[4:5] (hardware result not exactly specified)
    assert!(translate(Stage::Compute, &[0x7e10_5f04]).is_err());
    // LLVM gfx1030: v_min_f64 v[8:9], v[4:5], v[6:7] (NaN and signed-zero results depend on MODE.IEEE)
    let err = translate(Stage::Compute, &[0xd566_0008, 0x0002_0d04]).unwrap_err();
    assert!(err.contains("MODE.IEEE"), "{err}");
}
