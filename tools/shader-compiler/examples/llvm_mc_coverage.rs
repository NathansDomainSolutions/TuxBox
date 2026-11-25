//! Instruction coverage against LLVM's own RDNA2 disassembler tests.
//!
//! usage: llvm_mc_coverage <dir with LLVM gfx10_*.txt files> [rows.tsv]
//!
//! Fetch the inputs with `tools/fetch_llvm_amdgpu_mc_tests.sh`. Each test file
//! pairs a `# GFX10: <asm> ; encoding: [...]` check line with the bytes that
//! produce it. Every encoding runs through the decoder and then through the
//! HLSL translator as a compute shader with a 64-thread group, so complete
//! 32- and 64-lane waves (two literal moves, the instruction,
//! `s_endpgm`). The report counts which instructions decode as LLVM does,
//! which translate and which are refused, and ranks the refusals.
//!
//! A refusal here is not always a gap: an instruction out of context (a buffer
//! access without a known descriptor, a wave operation in an incomplete
//! wave) can be refused although real shaders translate. The reasons tell the
//! two apart, so read the ranking by reason, not only by count.
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tuxbox_xshader::{hlsl, ShaderExtract, Stage};

/// One encoding from an LLVM test file.
struct Case {
    file: String,
    asm: String,
    words: Vec<u32>,
}

/// LLVM's mnemonic without the encoding suffix it prints. The decoder names
/// global and scratch accesses by their shared FLAT opcode table, so those
/// prefixes compare as `flat_`.
fn base_mnemonic(asm: &str) -> String {
    let m = asm.split_whitespace().next().unwrap_or("");
    let flat;
    let m = match m.strip_prefix("global_").or_else(|| m.strip_prefix("scratch_")) {
        Some(rest) => { flat = format!("flat_{rest}"); flat.as_str() }
        None => m,
    };
    let m = m.strip_suffix("_e32").or_else(|| m.strip_suffix("_e64")).or_else(|| m.strip_suffix("_sdwa"))
        .or_else(|| m.strip_suffix("_dpp8")).or_else(|| m.strip_suffix("_dpp")).unwrap_or(m);
    m.to_string()
}

/// Pair each check line with the byte line that follows it. Lines that check
/// for a warning or no output (invalid encodings) are skipped.
fn parse(file: &str, text: &str) -> Vec<Case> {
    let mut out = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('#') {
            let rest = rest.trim();
            if rest.starts_with("RUN:") { continue; }
            if let Some((prefix, check)) = rest.split_once(':') {
                let is_check = !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-' || c == '_');
                // Wave32 checks only (`W32`), since the wave64 lines repeat the same bytes.
                let wanted = prefix.starts_with("GFX10") || prefix == "W32";
                if is_check && wanted && !prefix.contains("ERR") && !prefix.contains("WARN") {
                    let check = check.trim();
                    let check = check.strip_prefix("-NEXT:").map(str::trim).unwrap_or(check);
                    pending = (!check.is_empty() && !check.contains("warning") && !check.contains("error"))
                        .then(|| check.split(';').next().unwrap_or("").trim().to_string());
                }
            }
            continue;
        }
        if !line.starts_with("0x") { continue; }
        let bytes: Option<Vec<u8>> = line.split(|c| c == ',' || c == ' ').filter(|s| !s.is_empty())
            .map(|s| u8::from_str_radix(s.trim_start_matches("0x"), 16).ok()).collect();
        let asm = pending.take();
        let (Some(bytes), Some(asm)) = (bytes, asm) else { continue };
        if bytes.is_empty() || bytes.len() % 4 != 0 { continue; }
        let words = bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push(Case { file: file.to_string(), asm, words });
    }
    out
}

const V4: [u32; 2] = [0x7e08_02ff, 0x1357_2468];
const V5: [u32; 2] = [0x7e0a_02ff, 0x2468_1357];
const ENDPGM: u32 = 0xbf81_0000;

fn shader(body: &[u32]) -> ShaderExtract {
    let words: Vec<u32> = V4.iter().chain(&V5).chain(body).chain(&[ENDPGM]).copied().collect();
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        flat_inputs: Vec::new(), flat_outputs: Vec::new(), stage: Stage::Compute, shader_model: (6, 0),
        required_features: None, bitcode_stripped: true, fspx_config: None, ps09_pixel_inputs: None,
        fssm: Vec::new(), entry: Some("main".into()), isa_data: isa.clone(), isa, isa_offset_in_fssm: 0,
        psv_threads: None, isa_preamble: [0, 64, 1, 1], has_input_sig: false, has_output_sig: false,
        has_psv0: false, has_root_sig: false, input_sig: Vec::new(), output_sig: Vec::new(),
        resources: Vec::new(), gs_info: None, mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
    }
}

/// A refusal message with offsets, registers and numbers taken out, so that
/// refusals of the same kind group together.
fn reason(msg: &str) -> String {
    let msg = msg.split(" [").next().unwrap_or(msg);
    let mut out = String::new();
    let mut chars = msg.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || (c == '0' && chars.peek() == Some(&'x')) {
            while chars.peek().is_some_and(|d| d.is_ascii_hexdigit() || *d == 'x') { chars.next(); }
            out.push('#');
        } else {
            out.push(c);
        }
    }
    out.chars().take(110).collect()
}

enum Outcome { Translated, Refused(String), Panicked }

fn run(case: &Case) -> (bool, String, Outcome) {
    let isa: Vec<u8> = case.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let insns = tuxbox_dxbc::rdna2::disassemble(&isa);
    let want = base_mnemonic(&case.asm);
    let decoded = insns.len() == 1 && insns[0].dwords == case.words.len()
        && base_mnemonic(&insns[0].mnemonic) == want;
    let seen = insns.first().map_or_else(String::new, |i| i.mnemonic.clone());
    let outcome = match catch_unwind(AssertUnwindSafe(|| hlsl::translate(&shader(&case.words)))) {
        Ok(Ok(_)) => Outcome::Translated,
        Ok(Err(e)) => Outcome::Refused(e),
        Err(_) => Outcome::Panicked,
    };
    (decoded, seen, outcome)
}

#[derive(Default)]
struct Tally { total: usize, decoded: usize, translated: usize, refused: usize, panicked: usize }

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: llvm_mc_coverage <dir> [rows.tsv]");
    let rows_path = args.next();
    let mut paths: Vec<_> = std::fs::read_dir(&dir).expect("read input dir").filter_map(|e| e.ok())
        .map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    paths.sort();
    let mut cases = Vec::new();
    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        cases.extend(parse(&name, &std::fs::read_to_string(p).expect("read test file")));
    }
    std::panic::set_hook(Box::new(|_| {}));

    let mut by_file: BTreeMap<String, Tally> = BTreeMap::new();
    let mut all = Tally::default();
    let mut reasons: BTreeMap<String, (usize, BTreeMap<String, usize>)> = BTreeMap::new();
    let mut mnemonics: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut rows = String::from("file\tasm\twords\tdecoded\tdecoder_mnemonic\toutcome\treason\n");
    for case in &cases {
        let (decoded, seen, outcome) = run(case);
        let m = base_mnemonic(&case.asm);
        for t in [by_file.entry(case.file.clone()).or_default(), &mut all] {
            t.total += 1;
            t.decoded += decoded as usize;
            match &outcome {
                Outcome::Translated => t.translated += 1,
                Outcome::Refused(_) => t.refused += 1,
                Outcome::Panicked => t.panicked += 1,
            }
        }
        let entry = mnemonics.entry(m.clone()).or_default();
        entry.0 += 1;
        let (tag, why) = match &outcome {
            Outcome::Translated => { entry.1 += 1; ("translated", String::new()) }
            Outcome::Refused(e) => ("refused", reason(e)),
            Outcome::Panicked => ("panicked", "translator panicked".to_string()),
        };
        if !why.is_empty() {
            let r = reasons.entry(why.clone()).or_default();
            r.0 += 1;
            *r.1.entry(m.clone()).or_default() += 1;
        }
        let words: Vec<String> = case.words.iter().map(|w| format!("{w:08x}")).collect();
        rows += &format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", case.file, case.asm, words.join(" "), decoded, seen, tag, why);
    }
    if let Some(path) = rows_path { std::fs::write(path, rows).expect("write rows"); }

    let pct = |n: usize, d: usize| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
    println!("# LLVM MC coverage ({} encodings from {} files)\n", all.total, paths.len());
    println!("| file | encodings | decoded | translated | refused | panicked |\n|---|---|---|---|---|---|");
    for (f, t) in by_file.iter().chain(std::iter::once((&"all".to_string(), &all))) {
        println!("| {f} | {} | {} ({:.1}%) | {} ({:.1}%) | {} | {} |", t.total, t.decoded, pct(t.decoded, t.total),
            t.translated, pct(t.translated, t.total), t.refused, t.panicked);
    }
    let untouched: Vec<_> = mnemonics.iter().filter(|(_, (_, ok))| *ok == 0).collect();
    println!("\n{} of {} distinct mnemonics translate in at least one form.", mnemonics.len() - untouched.len(), mnemonics.len());
    let mut ranked: Vec<_> = reasons.into_iter().collect();
    ranked.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("\n## Refusal reasons by count\n\n| count | reason | mnemonics (most common first) |\n|---|---|---|");
    for (why, (n, ms)) in ranked.iter().take(60) {
        let mut ms: Vec<_> = ms.iter().collect();
        ms.sort_by(|a, b| b.1.cmp(a.1));
        let names: Vec<String> = ms.iter().take(8).map(|(m, c)| format!("{m} ({c})")).collect();
        let more = if ms.len() > 8 { format!(", +{} more", ms.len() - 8) } else { String::new() };
        println!("| {n} | {} | {}{more} |", why.replace('|', "\\|"), names.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_checks_with_bytes_and_skips_invalid_encodings() {
        let text = "# RUN: llvm-mc\n# GFX10: v_bfrev_b32_e32 v255, v1 ; encoding: [0x01,0x71,0xfe,0x7f]\n0x01,0x71,0xfe,0x7f\n\
                    # GFX10: warning: invalid instruction encoding\n0x00,0x00,0x00,0xff\n\n0x11,0x22,0x33,0x44\n";
        let cases = parse("t.txt", text);
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].asm, "v_bfrev_b32_e32 v255, v1");
        assert_eq!(cases[0].words, vec![0x7ffe_7101]);
        assert_eq!(base_mnemonic(&cases[0].asm), "v_bfrev_b32");
        assert_eq!(base_mnemonic("global_load_dword v1, v[2:3], off"), "flat_load_dword");
        let w32 = parse("c.txt", "# W32: v_cmp_class_f32_e32 vcc_lo, -1, v2 ; encoding: [0xc1,0x04,0x10,0x7d]\n# W64: v_cmp_class_f32_e32 vcc, -1, v2\n0xc1,0x04,0x10,0x7d\n");
        assert_eq!(w32.len(), 1);
        assert_eq!(w32[0].asm, "v_cmp_class_f32_e32 vcc_lo, -1, v2");
    }

    #[test]
    fn reasons_group_without_offsets_and_registers() {
        assert_eq!(reason("unsupported vector op [v_add_f64 @ 0x10]"), "unsupported vector op");
        assert_eq!(reason("mubuf at 0x10: s8 not a buffer descriptor"), "mubuf at #: s# not a buffer descriptor");
    }

    #[test]
    fn a_bfrev_decodes_and_translates() {
        let case = Case { file: "t".into(), asm: "v_bfrev_b32_e32 v5, v1".into(), words: vec![0x7e0a_7101] };
        let (decoded, seen, outcome) = run(&case);
        assert!(decoded, "{seen}");
        assert!(matches!(outcome, Outcome::Translated));
    }
}
