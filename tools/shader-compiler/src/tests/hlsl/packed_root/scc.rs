use super::*;

fn ops(words: &[u32]) -> Vec<Insn> {
    disassemble(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())
}

#[test]
fn scc_consumers_need_a_data_only_writer() {
    // From ProjectScarlett (00663d62d605): the pointer's high byte sets SCC,
    // then a compare of loaded data replaces it before the select.
    let masked = ops(&[0x8703_ff01, 0x0000_00ff]); // s_and_b32 s3, s1, 0xff
    let compare = ops(&[0xbf06_8007]); // s_cmp_eq_u32 s7, 0
    let select = ops(&[0x850c_807e]); // s_cselect_b32 s12, exec_lo, 0
    let branch = ops(&[0xbf85_0000]); // s_cbranch_scc1 +0
    let mut state = [Word::Unavailable; 107];
    state[0] = Word::PointerLo;
    state[1] = Word::PointerHi;
    state[7] = Word::Data;
    let mut plan = PackedRoot::default();
    // Unknown SCC on entry.
    assert!(plan.step(0, &select[0], &[], &mut state.clone()).is_err());
    plan.step(0, &masked[0], &[], &mut state).unwrap();
    assert!(plan.step(1, &select[0], &[], &mut state.clone()).is_err());
    assert!(plan.step(1, &branch[0], &[], &mut state.clone()).is_err());
    plan.step(1, &compare[0], &[], &mut state).unwrap();
    plan.step(2, &select[0], &[], &mut state).unwrap();
    assert!(state[12] == Word::Data);
    plan.step(3, &branch[0], &[], &mut state).unwrap();
}
