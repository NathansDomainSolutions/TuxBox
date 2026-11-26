// Run each synthetic program's translation over its input lanes and read back every lane's words.
#include "context.h"
#include <sstream>

using namespace shader_control;

struct Case {
    std::string name, shader, input;
    UINT groups, lanes;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12Resource> source, out;
    std::vector<char> bytes;
};

int main() {
    FILE *receipt{};
    if (fopen_s(&receipt, "result.txt", "wb")) return 2;
    int status = 0;
    try {
        Context ctx(receipt);
        ID3D12Device *device = ctx.device.Get();
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC description{2, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> serialized;
        check(D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr), "serialize root");
        ComPtr<ID3D12RootSignature> root;
        check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)), "root");
        UINT64 frequency{};
        check(ctx.queue->GetTimestampFrequency(&frequency), "frequency");
        D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
        ComPtr<ID3D12QueryHeap> queries;
        check(device->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)), "query heap");
        auto stamps = buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

        std::vector<Case> cases;
        std::ifstream list_file("cases.txt");
        Case next;
        while (list_file >> next.name >> next.shader >> next.input >> next.groups >> next.lanes) cases.push_back(next);
        for (auto &c : cases) {
            const auto code = read_shader(c.shader);
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
            pd.pRootSignature = root.Get();
            pd.CS = {code.data(), code.size()};
            const HRESULT hr = device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&c.pipeline));
            if (FAILED(hr)) { std::fprintf(receipt, "%s_pipeline=%#lx\n", c.name.c_str(), static_cast<unsigned long>(hr)); continue; }
            std::ifstream input(c.input, std::ios::binary);
            c.bytes.assign(std::istreambuf_iterator<char>{input}, {});
            if (c.bytes.empty()) throw std::runtime_error("missing input " + c.input);
            c.source = buffer(device, static_cast<UINT>(c.bytes.size()), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            upload(c.source.Get(), c.bytes.data(), c.bytes.size());
            c.out = buffer(device, c.lanes * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        D3D12_COMMAND_QUEUE_DESC qd{};
        for (int sample = 0; sample < 1; ++sample) {
            for (size_t at = 0; at < cases.size(); ++at) {
                // Start each sample at a different case so no shader always follows the same one.
                Case &c = cases[(at + sample) % cases.size()];
                if (!c.pipeline) continue;
                ComPtr<ID3D12CommandAllocator> allocator;
                ComPtr<ID3D12GraphicsCommandList> list;
                check(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)), "allocator");
                check(device->CreateCommandList(0, qd.Type, allocator.Get(), c.pipeline.Get(), IID_PPV_ARGS(&list)), "list");
                list->SetComputeRootSignature(root.Get());
                list->SetComputeRootShaderResourceView(0, c.source->GetGPUVirtualAddress());
                list->SetComputeRootUnorderedAccessView(1, c.out->GetGPUVirtualAddress());
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = c.out.Get();
                list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                list->Dispatch(c.groups, 1, 1); list->ResourceBarrier(1, &barrier);
                list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, stamps.Get(), 0);
                submit_and_wait(device, ctx.queue.Get(), list.Get());
                UINT64 *mapped{};
                D3D12_RANGE read{0, 16}, none{};
                check(stamps->Map(0, &read, reinterpret_cast<void **>(&mapped)), "map stamps");
                std::fprintf(receipt, "%s_ns%d=%.1f\n", c.name.c_str(), sample,
                             1e9 * double(mapped[1] - mapped[0]) / double(frequency) );
                stamps->Unmap(0, &none);
            }
        }
        for (auto &c : cases) {
            if (!c.pipeline) continue;
            const UINT bytes = c.lanes * 16;
            auto readback = buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> list;
            check(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)), "allocator");
            check(device->CreateCommandList(0, qd.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "list");
            transition(list.Get(), c.out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(readback.Get(), 0, c.out.Get(), 0, bytes);
            submit_and_wait(device, ctx.queue.Get(), list.Get());
            write_buffer(readback.Get(), c.name, bytes);
        }
    } catch (const std::exception &error) {
        std::fprintf(receipt, "error=%s\n", error.what());
        status = 1;
    }
    return std::fclose(receipt) ? 2 : status;
}
