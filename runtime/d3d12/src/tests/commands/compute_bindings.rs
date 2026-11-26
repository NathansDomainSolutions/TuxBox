use super::*;

/// A version 1.0 RTS0 with one table of four SRVs and the given flags.
fn root(flags: u32) -> RootSource {
    let words: [u32; 16] = [
        1, 1, 24, 0, 0, flags, // version, parameters at 24, no samplers, flags
        0, 0, 36, // parameter 0: a table for every stage, at 36
        1, 44, // one range, at 44
        0, 4, 0, 0, 0, // SRVs t0-t3 in space 0, from the table's first slot
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    RootSource::new(&bytes)
}

#[test]
fn paged_layout_gap_names_why_a_root_is_not_paged() {
    assert_eq!(paged_layout_gap(Some(&root(0))), None);
    assert_eq!(
        paged_layout_gap(Some(&root(0x400))),
        Some("its root indexes the resource heap directly")
    );
    assert_eq!(
        paged_layout_gap(Some(&RootSource::new(&[1, 2, 3]))),
        Some("its root signature could not be parsed")
    );
    assert_eq!(paged_layout_gap(None), Some("no root signature was recorded for the pipeline"));
}
