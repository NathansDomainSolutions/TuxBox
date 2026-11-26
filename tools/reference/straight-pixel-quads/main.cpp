// Synthetic uint render-target readback for nonheap FI=1 DPP16 axis swaps.
// Shared control headers include a static upload helper unused by this draw-only guest.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include "../independent-shaders/context.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif
using namespace shader_control;
static constexpr UINT side = 16, pitch = 256, bytes = side * pitch;

// Create and return the fixed-size uint render target used for exact component
// comparisons. The device supplies allocation, and the caller owns the resource.
static ComPtr<ID3D12Resource> target_image(ID3D12Device* device) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = d.Height = side;
    d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
    d.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    hp.CreationNodeMask = hp.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> target;
    check(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
          D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&target)), "target");
    return target;
}

// Draw the named translated shader with full or clipped scissor coverage, then
// save its uint pixels under a name derived from the shader and coverage mode.
// The oversized triangle covers scissor helper lanes, so their position inputs
// remain defined while the clear color identifies pixels outside the scissor.
static void draw(Context& ctx, const std::string& shader, bool clipped) {
    auto device = ctx.device.Get();
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rd{};
    rd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeVersionedRootSignature(&rd, &blob, nullptr), "serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                     IID_PPV_ARGS(&root)), "root");
    auto vs = read_shader("vertex"), ps = read_shader(shader);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {ps.data(), ps.size()};
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.SampleDesc.Count = 1;
    pd.SampleMask = UINT_MAX;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_UINT;
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "pipeline");
    auto target = target_image(device);
    auto readback = buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtv;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv)), "RTV heap");
    auto rt = rtv->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rt);
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                   IID_PPV_ARGS(&list)), "list");
    const float guard[]{17, 29, 43, 71};
    list->ClearRenderTargetView(rt, guard, 0, nullptr);
    list->OMSetRenderTargets(1, &rt, FALSE, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    D3D12_VIEWPORT vp{0, 0, side, side, 0, 1};
    D3D12_RECT rect = clipped ? D3D12_RECT{3, 3, 11, 11} : D3D12_RECT{0, 0, side, side};
    list->RSSetViewports(1, &vp);
    list->RSSetScissorRects(1, &rect);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list.Get(), target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = target.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_UINT, side, side, 1, pitch}};
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    submit_and_wait(device, ctx.queue.Get(), list.Get());
    write_buffer(readback.Get(), shader + (clipped ? "_clipped" : "_full"), bytes);
}

// Run both register layouts in full and clipped coverage modes. Write adapter
// and guest errors to result.txt, and return nonzero when setup or drawing fails.
int main() {
    FILE* report{};
    if (fopen_s(&report, "result.txt", "wb")) return 2;
    int status = 0;
    try {
        Context ctx(report);
        for (const char* shader : {"copy", "inplace"}) {
            draw(ctx, shader, false);
            draw(ctx, shader, true);
        }
    } catch (const std::exception& e) {
        std::fprintf(report, "error=%s\n", e.what());
        status = 1;
    }
    return std::fclose(report) ? 2 : status;
}
