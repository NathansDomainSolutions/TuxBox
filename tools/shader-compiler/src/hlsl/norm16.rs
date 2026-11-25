//! Two-channel 16-bit normalized packs: `v_cvt_pknorm_u16_f32` (UNORM16) and
//! `v_cvt_pknorm_i16_f32` (SNORM16), and their half-float source forms
//! `v_cvt_pknorm_u16_f16` and `v_cvt_pknorm_i16_f16`. Source 0 fills the low
//! half and source 1 the high half. The conversions follow the D3D float-to-normalized rules:
//! UNORM clamps to [0, 1] and scales by 65535; SNORM clamps to [-1, 1] and
//! scales by 32767. Both round to nearest even, and NaN converts to 0.

/// Whether `mnemonic` is a normalized 16-bit pack, and if so whether it is signed.
pub(super) fn signed(mnemonic: &str) -> Option<bool> {
    match mnemonic {
        "v_cvt_pknorm_u16_f32" | "v_cvt_pknorm_u16_f16" => Some(false),
        "v_cvt_pknorm_i16_f32" | "v_cvt_pknorm_i16_f16" => Some(true),
        _ => None,
    }
}

/// The packed 32-bit result for float expressions `lo` and `hi`. Each is
/// evaluated once per use, so callers pass temporaries for long expressions.
pub(super) fn pack(signed: bool, lo: &str, hi: &str) -> String {
    let half = |x: &str| if signed {
        // clamp() would turn NaN into a bound, so NaN is tested on the bits.
        format!("(tb_isnan({x}) ? 0u : (asuint((int)round(clamp({x}, -1.0, 1.0) * 32767.0)) & 0xffffu))")
    } else {
        // saturate() returns 0 for NaN.
        format!("(uint)round(saturate({x}) * 65535.0)")
    };
    format!("({} | ({} << 16))", half(lo), half(hi))
}

/// The float a pixel export passes to the render target for a 16-bit `word`
/// (the bits in the low half) written by a pack. SNORM -32768 maps to -1.0.
pub(super) fn export(signed: bool, word: &str) -> String {
    if signed {
        format!("max((float)(asint(({word}) << 16) >> 16) / 32767.0, -1.0)")
    } else {
        format!("((float){word} / 65535.0)")
    }
}

#[cfg(test)]
mod tests {
    use super::{export, pack, signed};
    use crate::hlsl::float64::tests::translate;
    use crate::{hlsl, ShaderExtract, SigElement, Stage};

    /// A pixel program with one four-component float SV_Target0: `v4`/`v5`
    /// literals, then `body`, then `s_endpgm`.
    fn pixel(body: &[u32]) -> Result<String, String> {
        let words: Vec<u32> = [0x7e08_02ff, 0x1357_2468, 0x7e0a_02ff, 0x2468_1357].iter()
            .chain(body).chain(&[0xbf81_0000]).copied().collect();
        let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let target = SigElement { name: "SV_Target".into(), index: 0, sysval: 64, comp_type: 3,
            register: 0, mask: 0xf, rw_mask: 0xf };
        let ex = ShaderExtract { flat_inputs: Vec::new(), flat_outputs: Vec::new(),
            stage: Stage::Pixel, shader_model: (6, 0), required_features: None, bitcode_stripped: true,
            fspx_config: None, ps09_pixel_inputs: None, fssm: Vec::new(), entry: Some("main".into()),
            isa_data: isa.clone(), isa, isa_offset_in_fssm: 0, psv_threads: None, isa_preamble: [0, 1, 1, 1],
            has_input_sig: true, has_output_sig: true, has_psv0: false, has_root_sig: false,
            input_sig: Vec::new(), output_sig: vec![target], resources: Vec::new(), gs_info: None,
            mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
        };
        hlsl::translate(&ex).map(|t| t.hlsl)
    }

    // LLVM gfx1030 words.
    const I16: [u32; 2] = [0xd768_0008, 0x0002_0b04]; // v_cvt_pknorm_i16_f32 v8, v4, v5
    const U16: [u32; 2] = [0xd769_0008, 0x0002_0b04]; // v_cvt_pknorm_u16_f32 v8, v4, v5

    #[test]
    fn mnemonics_select_the_signedness() {
        assert_eq!(signed("v_cvt_pknorm_i16_f32"), Some(true));
        assert_eq!(signed("v_cvt_pknorm_u16_f32"), Some(false));
        assert_eq!(signed("v_cvt_pk_u16_u32"), None);
    }

    #[test]
    fn snorm_pack_clamps_scales_and_zeroes_nan() {
        let e = pack(true, "a", "b");
        assert!(e.contains("tb_isnan(a) ? 0u : (asuint((int)round(clamp(a, -1.0, 1.0) * 32767.0)) & 0xffffu)"), "{e}");
        assert!(e.ends_with("<< 16))"), "{e}");
        // The unsigned form keeps the expression the emitter used before.
        assert_eq!(pack(false, "a", "b"), "((uint)round(saturate(a) * 65535.0) | ((uint)round(saturate(b) * 65535.0) << 16))");
    }

    #[test]
    fn snorm_export_sign_extends_and_maps_the_minimum_to_minus_one() {
        assert_eq!(export(true, "w"), "max((float)(asint((w) << 16) >> 16) / 32767.0, -1.0)");
        assert_eq!(export(false, "w"), "((float)w / 65535.0)");
    }

    #[test]
    fn compute_programs_translate_both_packs() {
        let hlsl = translate(Stage::Compute, &I16).unwrap();
        assert!(hlsl.contains("clamp(") && hlsl.contains("* 32767.0"), "{hlsl}");
        assert!(hlsl.contains("v8 = "), "{hlsl}");
        let hlsl = translate(Stage::Compute, &U16).unwrap();
        assert!(hlsl.contains("saturate(") && hlsl.contains("* 65535.0"), "{hlsl}");
        // v_cvt_pknorm_i16_f32 v8, -v4, |v5|
        let hlsl = translate(Stage::Compute, &[0xd768_0208, 0x2002_0b04]).unwrap();
        assert!(hlsl.contains("(-asfloat(v4))") && hlsl.contains("abs(asfloat(v5))"), "{hlsl}");
    }

    #[test]
    fn modifiers_and_unexported_pixel_packs_are_refused() {
        // clamp
        assert!(translate(Stage::Compute, &[0xd768_8008, 0x0002_0b04]).is_err());
        // op_sel bits
        assert!(translate(Stage::Compute, &[0xd768_0808, 0x0002_0b04]).is_err());
        // omod (mul:2) cannot be written in LLVM syntax but can be encoded.
        assert!(translate(Stage::Compute, &[0xd768_0008, 0x0802_0b04]).is_err());
        // A pixel pack that is not immediately exported stays refused.
        for words in [I16, U16] {
            let err = translate(Stage::Pixel, &words).unwrap_err();
            assert!(err.contains("unresolved packed"), "{err}");
        }
    }

    #[test]
    fn pixel_packs_export_two_normalized_channels() {
        // LLVM gfx1030: exp mrt0 v8, v8, off, off compr
        const EXPORT: [u32; 2] = [0xf800_0403, 0x0000_0008];
        let hlsl = pixel(&[I16[0], I16[1], EXPORT[0], EXPORT[1]]).unwrap();
        assert!(hlsl.contains("max((float)(asint((((v8 >> 0) & 0xffffu)) << 16) >> 16) / 32767.0, -1.0)"), "{hlsl}");
        assert!(hlsl.contains("max((float)(asint((((v8 >> 16) & 0xffffu)) << 16) >> 16) / 32767.0, -1.0)"), "{hlsl}");
        assert!(!hlsl.contains("f16tof32((v8"), "{hlsl}");
        let hlsl = pixel(&[U16[0], U16[1], EXPORT[0], EXPORT[1]]).unwrap();
        assert!(hlsl.contains("((float)((v8 >> 16) & 0xffffu) / 65535.0)"), "{hlsl}");
    }

    #[test]
    fn half_source_packs_select_halves_and_apply_modifiers() {
        let compute = |w: [u32; 2]| translate(Stage::Compute, &w);
        // LLVM gfx1030: v_cvt_pknorm_u16_f16 v8, v4, v5
        let hlsl = compute([0xd713_0008, 0x0002_0b04]).unwrap();
        assert!(hlsl.contains(&pack(false, "f16tof32((v4 >> 0u) & 0xffffu)", "f16tof32((v5 >> 0u) & 0xffffu)")), "{hlsl}");
        // v_cvt_pknorm_i16_f16 v8, -|v4|, v5 op_sel:[1,0,0]: source 0 is the high half.
        let hlsl = compute([0xd712_0908, 0x2002_0b04]).unwrap();
        assert!(hlsl.contains("lo = (-abs(f16tof32((v4 >> 16u) & 0xffffu)));"), "{hlsl}");
        assert!(hlsl.contains("hi = f16tof32((v5 >> 0u) & 0xffffu);"), "{hlsl}");
        // v_cvt_pknorm_u16_f16 v8, v4, v5 op_sel:[0,1,0]
        assert!(compute([0xd713_1008, 0x0002_0b04]).unwrap().contains("f16tof32((v5 >> 16u) & 0xffffu)"));
        // v_cvt_pknorm_u16_f16 v8, 0.5, v5: the inline constant is the half 0.5.
        assert!(compute([0xd713_0008, 0x0002_0af0]).unwrap().contains("saturate(asfloat(0x3f000000u))"));
        // Refused: high-half selection of an inline constant, and op_sel on the destination.
        assert!(compute([0xd713_0808, 0x0002_0af0]).is_err());
        assert!(compute([0xd712_4008, 0x0002_0b04]).is_err());
        // A pixel pack feeds the two-channel export like the f32 form.
        let hlsl = pixel(&[0xd712_0008, 0x0002_0b04, 0xf800_0403, 0x0000_0008]).unwrap();
        assert!(hlsl.contains(&export(true, "((v8 >> 16) & 0xffffu)")), "{hlsl}");
        assert_eq!(signed("v_cvt_pknorm_i16_f16"), Some(true));
        assert_eq!(signed("v_cvt_pknorm_u16_f16"), Some(false));
    }
}
