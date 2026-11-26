//! Translate a synthetic wave64 compute program, supplied as native words, into
//! the two-lanes-per-thread form. Arguments: thread group size, then the words.
use tuxbox_xshader::{ShaderExtract, Stage, hlsl};

mod fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/support/wave_guard_fixture.rs"));
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: u32 = args.next().expect("thread count").parse().expect("integer");
    let words: Vec<u32> = args.map(|s| u32::from_str_radix(&s, 16).expect("hex native word")).collect();
    assert!(threads % 64 == 0 && !words.is_empty() && words.last() == Some(&0xbf810000));
    let (mut shader, _, _): (ShaderExtract, _, _) = fixture::shader(64);
    shader.isa = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    shader.isa_data = shader.isa.clone();
    shader.isa_preamble = [0, threads, 1, 1];
    assert!(matches!(shader.stage, Stage::Compute));
    match hlsl::translate_paired_wave64(&shader) {
        Ok(result) => print!("{}", result.hlsl),
        Err(reason) => {
            eprintln!("{reason}");
            std::process::exit(65);
        }
    }
}
