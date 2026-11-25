use super::*;
use crate::rootsig::RootParam;

#[test]
fn bounded_psv_range_requires_every_physical_binding() {
    let mut program = vec![0; 0x100];
    program[..8].copy_from_slice(b"FSSM\x01\x01\0\0");
    for (field, count, rows) in [
        (0xe8, 0xf8, [[3, 0, 1, 7, 0x009u32], [3, 8, 1, 8, 0x009]]),
        (0xf0, 0xfc, [[0x300, 0, 7, 64, 0x00904], [0x300, 8, 8, 64, 0x00904]]),
    ] {
        let offset = (program.len() - 16) as u64;
        program[field..field + 8].copy_from_slice(&offset.to_le_bytes());
        program[count..count + 4].copy_from_slice(&2u32.to_le_bytes());
        program.extend(rows.into_iter().flatten().flat_map(u32::to_le_bytes));
    }
    let size = (program.len() - 16) as u32;
    program[8..12].copy_from_slice(&size.to_le_bytes());
    let root = RootSig { version: 2, flags: 0,
        params: vec![RootParam { visibility: 7,
            kind: RootParamKind::Table(vec![RootRange { range_type: 0, num: 2,
                base_register: 7, space: 1, offset_in_table: 0 }]) }],
        num_static_samplers: 0, static_samplers: Vec::new() };
    let layout = MeshLayout::read(&program, &[[5, 1, 7, 8]], &root).unwrap();
    assert_eq!(layout.tables[&(17, 0)].register, 7);
    assert_eq!(layout.tables[&(17, 32)].register, 8);
    assert!(layout.replacements.is_empty());
    assert!(MeshLayout::read(&program, &[[5, 1, 7, 9]], &root).is_none());
    assert!(MeshLayout::read(&program, &[[5, 1, 7, u32::MAX]], &root).is_none());
    assert!(MeshLayout::read(&program, &[[5, 1, 7, 8], [5, 1, 8, 8]], &root).is_none());
    program[0x114..0x118].copy_from_slice(&4u32.to_le_bytes());
    assert!(MeshLayout::read(&program, &[[5, 1, 7, 8]], &root).is_none());
}

#[test]
fn remaps_mesh_slots_only_if_both_fssm_tables_agree() {
    let mut program = vec![0; 0x100];
    program[..8].copy_from_slice(b"FSSM\x01\x01\0\0");
    for (field, count, rows) in [
        (0xe8, 0xf8, [[3, 16, 1, 7, 0x109u32], [0, 0, 2, 500, 0x20a]]),
        (
            0xf0,
            0xfc,
            [[0x300, 16, 7, 64, 0x10904], [2, 0, 500, 0, 0x20a04]],
        ),
    ] {
        let offset = (program.len() - 16) as u64;
        program[field..field + 8].copy_from_slice(&offset.to_le_bytes());
        program[count..count + 4].copy_from_slice(&2u32.to_le_bytes());
        program.extend(rows.into_iter().flatten().flat_map(u32::to_le_bytes));
    }
    let size = (program.len() - 16) as u32;
    program[8..12].copy_from_slice(&size.to_le_bytes());
    let table = RootParam {
        visibility: 0,
        kind: RootParamKind::Table(vec![RootRange {
            range_type: 0,
            num: 1,
            base_register: 4,
            space: 1,
            offset_in_table: 0,
        }]),
    };
    let root = RootSig {
        version: 2,
        flags: 0,
        params: vec![
            table.clone(),
            table,
            RootParam {
                visibility: 0,
                kind: RootParamKind::Constants {
                    register: 500,
                    space: 0,
                    num32: 2,
                },
            },
        ],
        num_static_samplers: 0,
        static_samplers: Vec::new(),
    };
    let resources = [[5, 1, 7, 7], [2, 0, 500, 500]];
    let layout = MeshLayout::read(&program, &resources, &root).unwrap();
    let RootParamKind::Table(ranges) = &layout.replacements[&1] else {
        panic!("expected a table")
    };
    assert_eq!((ranges[0].offset_in_table, ranges[0].base_register), (2, 7));
    assert_eq!(layout.tables[&(17, 64)].stride_words, Some(16));
    assert!(layout.takes_sgpr(19));
    assert!(!layout.takes_sgpr(20));
    program[0x104..0x108].copy_from_slice(&8u32.to_le_bytes());
    assert!(MeshLayout::read(&program, &resources, &root).is_none());
}

#[test]
fn pixel_programs_can_hold_unordered_access_textures() {
    let mut program = vec![0; 0x100];
    program[..8].copy_from_slice(b"FSSM\x01\x01\0\0");
    for (field, count, rows) in [
        (0xe8, 0xf8, [[3, 0, 1, 4, 0x008u32], [3, 8, 1, 2, 0x008]]),
        (0xf0, 0xfc, [[0x300, 0, 4, 0, 0x00808], [0x303, 8, 2, 4, 0x00808]]),
    ] {
        let offset = (program.len() - 16) as u64;
        program[field..field + 8].copy_from_slice(&offset.to_le_bytes());
        program[count..count + 4].copy_from_slice(&2u32.to_le_bytes());
        program.extend(rows.into_iter().flatten().flat_map(u32::to_le_bytes));
    }
    let size = (program.len() - 16) as u32;
    program[8..12].copy_from_slice(&size.to_le_bytes());
    // The root declares the unordered-access texture two slots later
    // than the program reads it.
    let table = |visibility| RootParam {
        visibility,
        kind: RootParamKind::Table(vec![
            RootRange { range_type: 0, num: 1, base_register: 4, space: 0, offset_in_table: 0 },
            RootRange { range_type: 1, num: 1, base_register: 2, space: 0, offset_in_table: 3 },
        ]),
    };
    let root = |visibility| RootSig {
        version: 2,
        flags: 0,
        params: vec![table(visibility)],
        num_static_samplers: 0,
        static_samplers: Vec::new(),
    };
    let resources = [[3, 0, 4, 4], [6, 0, 2, 2]];
    for visibility in [0, 5] {
        let layout = MeshLayout::read_stage(&program, &resources, &root(visibility), true).unwrap();
        let texture = layout.tables[&(16, 32)];
        assert_eq!((texture.kind, texture.register, texture.space), (1, 2, 0));
        let RootParamKind::Table(ranges) = &layout.replacements[&0] else {
            panic!("expected a table")
        };
        let slots: Vec<_> = ranges.iter().map(|r| (r.range_type, r.base_register, r.offset_in_table)).collect();
        assert_eq!(slots, [(0, 4, 0), (1, 2, 1)]);
    }
    // A mesh program holds no such descriptor, and each stage reads only
    // the parameters visible to it.
    assert!(MeshLayout::read(&program, &resources, &root(0)).is_none());
    assert!(MeshLayout::read_stage(&program, &resources, &root(7), true).is_none());
    assert!(MeshLayout::read_stage(&program, &resources, &root(5), false).is_none());
}

#[test]
fn pixel_layout_replaces_a_table_the_root_declares_one_slot_short() {
    // The program reads a texture at slot 2 and a constant buffer at slot
    // 3. The root declares the buffer at slot 2 and no texture at all.
    let mut program = vec![0; 0x100];
    program[..8].copy_from_slice(b"FSSM\x01\x01\0\0");
    for (field, count, rows) in [
        (0xe8, 0xf8, [[3, 16, 1, 75, 0x109u32], [3, 24, 1, 5, 0x109]]),
        (0xf0, 0xfc, [[0x300, 16, 75, 4, 0x10908], [0x302, 24, 5, 0, 0x10904]]),
    ] {
        let offset = (program.len() - 16) as u64;
        program[field..field + 8].copy_from_slice(&offset.to_le_bytes());
        program[count..count + 4].copy_from_slice(&2u32.to_le_bytes());
        program.extend(rows.into_iter().flatten().flat_map(u32::to_le_bytes));
    }
    let size = (program.len() - 16) as u32;
    program[8..12].copy_from_slice(&size.to_le_bytes());
    let range = |range_type, base_register, space| RootRange {
        range_type,
        num: 1,
        base_register,
        space,
        offset_in_table: u32::MAX,
    };
    let root = RootSig {
        version: 2,
        flags: 0,
        params: vec![
            RootParam { visibility: 0, kind: RootParamKind::Table(vec![range(0, 67, 0)]) },
            RootParam {
                visibility: 0,
                kind: RootParamKind::Table(vec![range(0, 66, 1), range(0, 2, 1), range(2, 5, 1), range(2, 0, 1)]),
            },
        ],
        num_static_samplers: 0,
        static_samplers: Vec::new(),
    };
    let resources = [[2, 1, 5, 5], [3, 1, 75, 75]];
    let layout = MeshLayout::read_stage(&program, &resources, &root, true).unwrap();
    assert!(!layout.replacements.contains_key(&0));
    let RootParamKind::Table(ranges) = &layout.replacements[&1] else {
        panic!("expected a table")
    };
    let slots: Vec<_> = ranges.iter().map(|r| (r.range_type, r.base_register, r.space, r.offset_in_table)).collect();
    assert_eq!(slots, [(0, 75, 1, 2), (2, 5, 1, 3)]);
}
