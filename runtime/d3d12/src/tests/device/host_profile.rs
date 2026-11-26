use super::*;

#[test]
fn overrides_name_the_lane_range_and_the_import_result() {
    assert_eq!(parse_overrides(""), Ok(Overrides::default()));
    assert_eq!(parse_overrides("wave=32").unwrap().wave, Some((32, 32)));
    assert_eq!(
        parse_overrides(" wave = 32-64 ; section-import = substituted ; "),
        Ok(Overrides { wave: Some((32, 64)), section_import: Some(SectionImport::Substituted) })
    );
    assert_eq!(parse_overrides("section-import=refused").unwrap().section_import, Some(SectionImport::Refused));
    assert_eq!(parse_overrides("section-import=aliased").unwrap().section_import, Some(SectionImport::Aliased));
}

#[test]
fn one_bad_setting_rejects_every_setting() {
    for text in ["wave", "wave=", "wave=48", "wave=64-32", "wave=2", "wave=256", "waves=32",
                 "section-import=yes", "wave=32;vendor=nvidia"] {
        assert!(parse_overrides(text).is_err(), "{text}");
    }
}

#[test]
fn only_a_failed_import_probe_selects_the_copy() {
    assert!(needs_mirror(SectionImport::Substituted));
    assert!(needs_mirror(SectionImport::Refused));
    assert!(!needs_mirror(SectionImport::Aliased));
    assert!(!needs_mirror(SectionImport::Unknown(-2147467259)));
}

#[test]
fn the_profile_line_marks_what_was_set() {
    let host = describe(Some((0x10de, 0x2705)), Some((32, 32)), SectionImport::Aliased, (64, 32), Overrides::default());
    assert_eq!(host, "host profile: adapter=10de:2705 wave=32-32 section-import=aliased descriptor-bytes=64/32");
    let set = Overrides { wave: Some((32, 32)), section_import: None };
    let forced = describe(None, set.wave, SectionImport::Unknown(-2147467259), (32, 16), set);
    assert_eq!(forced, "host profile: adapter=unknown wave=32-32 (set) section-import=unknown(0x80004005) descriptor-bytes=32/16");
}
