use super::{default_texture_uav_desc, desktop_cbv_bytes, earlier_page};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT};

#[test]
fn an_earlier_page_puts_a_view_on_a_whole_element() {
    // ProjectScarlett: 20-byte elements starting 352 bytes into a page.
    let page = earlier_page(0xaa8f_0160, 20, 0xaa00_0000).unwrap();
    assert_eq!(page, 0xaa8c_0000);
    assert_eq!((0xaa8f_0160 - page) % 20, 0);
    // The allocation starts at the view's own page: nothing earlier to use.
    assert_eq!(earlier_page(0xaa8f_0160, 20, 0xaa8f_0000), None);
    // No page boundary can fix an offset the stride's page factor rules out.
    assert_eq!(earlier_page(0xaa8f_0162, 20, 0), None);
    assert_eq!(earlier_page(0x2_0028, 20, 0), Some(0x2_0000));
}

#[test]
fn desktop_cbv_size_is_a_256_byte_multiple_up_to_the_limit() {
    assert_eq!(desktop_cbv_bytes(0), 0);
    assert_eq!(desktop_cbv_bytes(1), 256);
    assert_eq!(desktop_cbv_bytes(144), 256);
    assert_eq!(desktop_cbv_bytes(256), 256);
    assert_eq!(desktop_cbv_bytes(257), 512);
    assert_eq!(desktop_cbv_bytes(65536), 65536);
    assert_eq!(desktop_cbv_bytes(65537), 65537);
}

#[test]
fn default_texture_uav_desc_synthesizes_raw_views_for_all_texture_dimensions() {
    let base = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 16,
        Height: 16,
        DepthOrArraySize: 1,
        MipLevels: 1,
        ..Default::default()
    };
    let v2d = default_texture_uav_desc(&base, DXGI_FORMAT_R32_UINT).expect("2D UAV");
    assert_eq!(v2d.Format, DXGI_FORMAT_R32_UINT);
    assert_eq!(v2d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2D);

    let v2da = default_texture_uav_desc(
        &D3D12_RESOURCE_DESC { DepthOrArraySize: 6, ..base },
        DXGI_FORMAT_R32_UINT,
    )
    .expect("2D array UAV");
    assert_eq!(v2da.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
    assert_eq!(unsafe { v2da.Anonymous.Texture2DArray.ArraySize }, 6);

    let v1d = default_texture_uav_desc(
        &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D, Height: 1, ..base },
        DXGI_FORMAT_R16_UINT,
    )
    .expect("1D UAV");
    assert_eq!(v1d.Format, DXGI_FORMAT_R16_UINT);
    assert_eq!(v1d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE1D);

    let v1da = default_texture_uav_desc(
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            Height: 1,
            DepthOrArraySize: 4,
            ..base
        },
        DXGI_FORMAT_R16_UINT,
    )
    .expect("1D array UAV");
    assert_eq!(v1da.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE1DARRAY);
    assert_eq!(unsafe { v1da.Anonymous.Texture1DArray.ArraySize }, 4);

    let v3d = default_texture_uav_desc(
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            DepthOrArraySize: 8,
            ..base
        },
        DXGI_FORMAT_R32_UINT,
    )
    .expect("3D UAV");
    assert_eq!(v3d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
    assert_eq!(unsafe { v3d.Anonymous.Texture3D.WSize }, 8);

    assert!(default_texture_uav_desc(
        &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, ..base },
        DXGI_FORMAT_R32_UINT,
    )
    .is_none());
}
