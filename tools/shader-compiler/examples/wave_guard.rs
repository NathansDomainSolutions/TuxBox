//! Emit a synthetic guarded reduction via the actual RDNA2 translator.
use tuxbox_xshader::{ShaderExtract, Stage, hlsl};

mod fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/support/wave_guard_fixture.rs"));
}

fn main() {
    let width: u32 = std::env::args().nth(1).expect("wave width").parse().expect("integer");
    assert!(matches!(width, 32 | 64));
    let (mut shader, mut words, cmp) = fixture::shader(width);
    let group: Vec<u32> = std::env::args().skip(3).map(|s| s.parse().expect("group dimension")).collect();
    if !group.is_empty() {
        let group: [u32; 3] = group.try_into().expect("three group dimensions");
        assert_eq!(group.into_iter().try_fold(1u32, u32::checked_mul), Some(width));
        assert!(width == 64 || group == [32, 1, 1]);
        shader.psv_threads = Some(group);
    }
    match std::env::args().nth(2).as_deref() {
        Some("literal") => { words.insert(cmp, 0xbe87_0381); } // s_mov_b32 s7, 1
        Some("chain") => {
            // s_mov_b32 s9, 1; s_add_u32 s7, s9, 1.
            words.splice(cmp..cmp, [0xbe89_0381, 0x8007_8109]);
        }
        Some("chain-zero") => {
            // s_mov_b32 s9, 0; s_add_u32 s7, s9, 0.
            words.splice(cmp..cmp, [0xbe89_0380, 0x8007_8009]);
        }
        _ => (),
    }
    shader.isa = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    shader.isa_data = shader.isa.clone();
    let translated = hlsl::translate(&shader).expect("translate synthetic reductions");
    assert_eq!(translated.required_wave_size, (width == 32).then_some(32));
    print!("{}", translated.hlsl);
}
