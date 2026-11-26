use super::{matches, CHECKS};

#[test]
fn recognizes_only_the_exact_watchdog() {
    let image = |patch: Option<(usize, u8)>| {
        move |rva: usize, len: usize| {
            let (_, bytes) = CHECKS.iter().find(|(at, _)| *at == rva)?;
            let mut out = bytes[..len].to_vec();
            if let Some((_, value)) = patch.filter(|(at, _)| *at == rva) {
                out[0] = value;
            }
            Some(out)
        }
    };
    assert!(matches(image(None)));
    // A different displacement or crash store is another build.
    assert!(!matches(image(Some((0x221_5B50, 0x8B)))));
    assert!(!matches(image(Some((0x221_5DA9, 0x90)))));
    assert!(!matches(|_, _| None));
}

#[test]
fn flag_displacements_name_the_same_byte() {
    // RIP-relative: the instruction ends six bytes after its start.
    for (rva, bytes) in &CHECKS[..2] {
        let disp = i32::from_le_bytes(bytes[2..6].try_into().unwrap()) as i64;
        assert_eq!(*rva as i64 + 6 + disp, super::FLAG_RVA as i64);
    }
}
