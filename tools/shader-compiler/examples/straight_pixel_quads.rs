//! Independently authored nonheap FI=1 DPP16 pixel programs for GPU readback.
//!
//! The emitted programs use `SV_Position` as their pixel-coordinate input and
//! export exact unsigned integer values through `SV_Target`.
//!
//! # Boundaries
//!
//! This example constructs synthetic shader metadata and ISA, then delegates
//! translation to `tuxbox_xshader`. Its outputs are translated HLSL on stdout
//! and translation diagnostics on stderr.
use tuxbox_xshader::{hlsl, ShaderExtract, Stage};

/// Build a synthetic pixel shader for a selected quad-swap case.
///
/// `variant` is `copy`, `inplace`, or `guarded`; the returned shader declares
/// `SV_Position` input and unsigned `SV_Target` output. Position seeds the
/// pixel-coordinate registers, and integer exports preserve exact lane values.
fn shader(variant: &str) -> ShaderExtract {
    let mut words = vec![
        0xbe88047eu32, // s_mov_b64 s[8:9], exec: preserve raster coverage.
        0xbefe0a7e,    // s_wqm_b64 exec, exec: evaluate complete quads.
        0x7e040f02, 0x7e060f03, // Convert pixel X/Y to unsigned integers.
        0x34060685, 0x4a080702, // v4 = x + (y << 5).
        0x4a0808ff, 99,        // f(x,y) = 99 + x + 32*y.
    ];
    match variant {
        "copy" | "guarded" => words.extend([
            0x7e0a02fa, 0xff0cb104, // v5 = X(v4), FI=1, full row/bank masks.
            0x4a0c0905,            // v6 = v5 + v4.
            0x7e0e02fa, 0xff0c4e06, // v7 = Y(v6), FI=1.
            0x4a140f06,            // v10 = v7 + v6.
            0x87fe087e,            // Restore original coverage for export.
            0xf800180f, 0x040a0705, // Export v5,v7,v10,v4 as uint RGBA.
        ]),
        "inplace" => words.extend([
            0x7e160304,            // v11 = v4: preserve the unswapped input.
            0x7e0802fa, 0xff0cb104, // v4 = X(v4), FI=1.
            0x4a0c1704,            // v6 = v4 + v11.
            0x7e0c02fa, 0xff0c4e06, // v6 = Y(v6), FI=1 (in-place).
            0x4a140906,            // v10 = v6 + v4.
            0x87fe087e,
            0xf800180f, 0x0b0a0604, // Export v4,v6,v10,v11.
        ]),
        _ => panic!("variant must be copy, inplace, or guarded"),
    }
    words.push(0xbf810000); // s_endpgm.
    if variant == "guarded" {
        // A scalar branch before the first exchange invalidates convergence proof.
        words.insert(2, 0xbf840002); // s_cbranch_scc0 skips both position conversions.
    }
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        stage: Stage::Pixel, shader_model: (6, 6), required_features: None,
        bitcode_stripped: true, fspx_config: None, ps09_pixel_inputs: None,
        fssm: Vec::new(), entry: Some("main".into()), isa_data: isa.clone(), isa,
        isa_offset_in_fssm: 0, psv_threads: None, flat_inputs: Vec::new(),
        flat_outputs: Vec::new(), isa_preamble: [0, 64, 1, 1],
        has_input_sig: true, has_output_sig: true, has_psv0: false,
        has_root_sig: false, input_sig: vec![tuxbox_xshader::meta::SigElement {
            name: "SV_Position".into(), index: 0, sysval: 1, comp_type: 3,
            register: 0, mask: 15, rw_mask: 15,
        }],
        output_sig: vec![tuxbox_xshader::meta::SigElement {
            name: "SV_Target".into(), index: 0, sysval: 64, comp_type: 1,
            register: 0, mask: 15, rw_mask: 15,
        }], resources: Vec::new(), gs_info: None, mesh_info: None,
        primitive_sig: Vec::new(), root_sig: None,
    }
}

/// Translate the `copy`, `inplace`, or `guarded` command-line variant.
///
/// Successful translation prints native `ps_6_6` HLSL to stdout. Translation
/// refusal prints the diagnostic to stderr and exits with status 1.
fn main() {
    let variant = std::env::args().nth(1).expect("variant: copy, inplace, guarded");
    match hlsl::translate(&shader(&variant)) {
        Ok(t) => {
            assert_eq!(t.profile, "ps_6_6");
            assert_eq!(t.required_wave_size, None);
            print!("{}", t.hlsl);
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
