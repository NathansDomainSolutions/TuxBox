//! Emit a compute program whose LDS writes and reads go past the group's allocation.
//!
//! Each lane writes its input to its own word, then writes a second value
//! 32 KiB further on, past any group shared allocation, and reads both back.
//! RDNA2 drops the far write and reads 0 there.
use tuxbox_xshader::{ShaderExtract, Stage, hlsl};

fn main() {
    let words: [u32; 13] = [
        0xd8340000, 0x00000100, // ds_write_b32 v0, v1: this lane's word
        0xd8340000, 0x00000203, // ds_write_b32 v3, v2: 32 KiB past it
        0xbf8cc07f, // s_waitcnt lgkmcnt(0)
        0xd8d80000, 0x04000000, // ds_read_b32 v4, v0
        0xd8d80000, 0x05000003, // ds_read_b32 v5, v3
        0xbf8cc07f, // s_waitcnt lgkmcnt(0)
        0xbf810000, // s_endpgm
        0xbf800000, 0xbf800000, // s_nop padding
    ];
    let isa: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let shader = ShaderExtract {
        flat_inputs: Vec::new(), flat_outputs: Vec::new(), stage: Stage::Compute, shader_model: (6, 0),
        required_features: None, bitcode_stripped: true, fspx_config: None, ps09_pixel_inputs: None,
        fssm: Vec::new(), entry: Some("main".into()), isa_data: isa.clone(), isa, isa_offset_in_fssm: 0,
        psv_threads: None, isa_preamble: [0, 64, 1, 1], has_input_sig: false, has_output_sig: false,
        has_psv0: false, has_root_sig: false, input_sig: Vec::new(), output_sig: Vec::new(), resources: Vec::new(),
        gs_info: None, mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
    };
    let out = hlsl::translate(&shader).expect("translate the LDS bounds program");
    print!("{}", out.hlsl);
}
