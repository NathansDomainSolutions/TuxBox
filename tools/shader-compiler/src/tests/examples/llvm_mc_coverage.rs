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
