cbuffer Params : register(b1, space2) {
    uint writeArguments;
    uint groupCount;
};
RWStructuredBuffer<uint> dest : register(u2, space1);

[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (writeArguments) {
        if (id.x == 0) {
            dest[0] = groupCount;
            dest[1] = 1;
            dest[2] = 1;
        }
    } else {
        dest[id.x] = id.x + 1;
    }
}
