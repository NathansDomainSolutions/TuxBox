use super::*;

fn fixture(rows: &[[u32; 5]]) -> Vec<u8> {
    let start = 0x160;
    let mut data = vec![0u8; start];
    data[..8].copy_from_slice(b"FSSM\x01\x01\0\0");
    data[0xf0..0xf8].copy_from_slice(&((start - 16) as u64).to_le_bytes());
    data[0xfc..0x100].copy_from_slice(&(rows.len() as u32).to_le_bytes());
    for row in rows {
        data.extend(row.iter().flat_map(|v| v.to_le_bytes()));
    }
    let size = (data.len() - 16) as u32;
    data[8..12].copy_from_slice(&size.to_le_bytes());
    data
}

#[test]
fn reads_reflected_stride_for_structured_bindings_only() {
    let rows = [
        [0x300, 24, 3, 800, 0x404],
        [0x303, 8, 1, 44, 0x30704],
        [0x300, 0, 0, 1, 0x404],
    ];
    let data = fixture(&rows);
    let bindings = [(5, 3), (8, 1), (4, 0)].map(|(res_type, slot)| PsvResource {
        res_type,
        space: 0,
        lower: slot,
        upper: slot,
    });
    assert_eq!(
        stride_words(&data, &bindings, Stage::Compute, 3, false),
        Some(200)
    );
    assert_eq!(
        stride_words(&data, &bindings, Stage::Compute, 1, true),
        Some(11)
    );
    assert_eq!(
        stride_words(&data, &bindings, Stage::Compute, 0, false),
        None
    );
    assert_eq!(
        stride_words(&data, &bindings, Stage::Compute, 1, false),
        None
    );

    let binding = &bindings[..1];
    for row in [
        [0x300, 8, 3, 800, 0x404],
        [0x300, 24, 3, 799, 0x404],
        [0x300, 24, 3, 800, 0x408],
    ] {
        assert_eq!(
            stride_words(&fixture(&[row]), binding, Stage::Compute, 3, false),
            None
        );
    }
    assert_eq!(
        stride_words(&fixture(&[rows[0]; 2]), binding, Stage::Compute, 3, false),
        None
    );
    assert_eq!(
        stride_words(&data[..data.len() - 1], binding, Stage::Compute, 3, false),
        None
    );
    let mut out_of_range = data;
    out_of_range[0xf0..0xf8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        stride_words(&out_of_range, binding, Stage::Compute, 3, false),
        None
    );
}

#[test]
fn pixel_srv_stride_includes_unread_fields() {
    // These MainPS inputs only fetch the first word or vector of each record.
    let data = fixture(&[[0x300, 0, 0, 16, 0x804], [0x300, 16, 2, 32, 0x804]]);
    let resources = [0, 2].map(|slot| PsvResource {
        res_type: 5,
        space: 0,
        lower: slot,
        upper: slot,
    });
    assert_eq!(
        stride_words(&data, &resources, Stage::Pixel, 0, false),
        Some(4)
    );
    assert_eq!(
        stride_words(&data, &resources, Stage::Pixel, 2, false),
        Some(8)
    );
    assert_eq!(
        stride_words(&data, &resources, Stage::Compute, 0, false),
        None
    );
    assert_eq!(stride_words(&data, &resources, Stage::Pixel, 0, true), None);
}

#[test]
fn vertex_and_mesh_views_keep_the_complete_record() {
    let resource = [PsvResource { res_type: 5, space: 0, lower: 3, upper: 3 }];
    for (stage, tag, other) in [
        (Stage::Vertex, 0x30b04, Stage::Mesh),
        (Stage::Mesh, 0x91104, Stage::Vertex),
    ] {
        let table = fixture(&[[0x300, 24, 3, 800, tag]]);
        assert_eq!(stride_words(&table, &resource, stage, 3, false), Some(200));
        assert_eq!(stride_words(&table, &resource, other, 3, false), None);
    }
}

/// A root whose third table holds one structured buffer, and a program that loads its
/// descriptor through that table's pointer and reads the record's first word.
fn third_table(stage: Stage, uav: bool, locations: &[u32]) -> ShaderExtract {
    use crate::meta::{RootParam, RootParamKind, RootRange};
    let pointer = if stage == Stage::Compute { 6 } else { 10 };
    let words = [
        0xf408_0400 | pointer / 2, 0xfa00_0000, // s_load_dwordx4 s[16:19], s[pointer:pointer+1], 0
        0xe030_2000, 0x8004_0400, // buffer_load_dword v4, v0, s[16:19], 0 idxen
        0xbf81_0000, // s_endpgm
    ];
    let isa: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let mut ex = crate::hlsl::mesh::heap_extract(&[], 0, &isa);
    ex.stage = stage;
    ex.psv_threads = (stage == Stage::Compute).then_some([64, 1, 1]);
    ex.required_features = None;
    let class = if uav { 0x303 } else { 0x300 };
    // Only the first row is about the buffer the program reads.
    let rows: Vec<_> = locations.iter().enumerate().map(|(i, &l)| [class, 8 * i as u32, 128 + i as u32, 136, l]).collect();
    ex.fssm = fixture(&rows);
    ex.resources.push(PsvResource { res_type: if uav { 8 } else { 5 }, space: 1, lower: 128, upper: 128 });
    let root = ex.root_sig.as_mut().unwrap();
    root.version = 2;
    root.params = [(0, 0), (64, 0), (128, 1)]
        .map(|(base_register, space)| RootParam {
            visibility: 0,
            kind: RootParamKind::Table(vec![RootRange { range_type: uav.into(), num: 1, base_register, space, offset_in_table: 0 }]),
        })
        .into();
    ex
}

#[test]
fn a_record_behind_any_table_keeps_its_stride() {
    for (stage, location) in [(Stage::Compute, 0x20604), (Stage::Pixel, 0x20a04), (Stage::Vertex, 0x20a04)] {
        for uav in [false, true] {
            let element = |locations: &[u32]| {
                let hlsl = crate::hlsl::translate(&third_table(stage, uav, locations)).unwrap().hlsl;
                let start = hlsl.find("struct TbElem_").unwrap();
                let line = hlsl[start..].lines().next().unwrap();
                line[line.find('{').unwrap()..].to_owned()
            };
            assert_eq!(element(&[location]), "{ uint w[34]; };", "{stage:?} {uav}");
            // An image descriptor, another table's pointer, and parameters that do not
            // belong to the pointer do not describe this buffer. The guess is the one
            // word read. A vertex record may name any parameter for its register.
            let others = [location + 4, location - 0x100, location - 0x10000, location + 0x10000];
            for &other in if stage == Stage::Vertex { &others[..1] } else { &others } {
                assert_eq!(element(&[other]), "{ uint w[1]; };", "{stage:?} {uav} {other:#x}");
            }
        }
    }
    // Two parameters for one vertex register leave it without a known one.
    let hlsl = crate::hlsl::translate(&third_table(Stage::Vertex, false, &[0x20a04, 0x10a04])).unwrap().hlsl;
    assert!(hlsl.contains("{ uint w[1]; };"), "{hlsl}");
}
