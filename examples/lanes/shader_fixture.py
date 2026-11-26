"""The synthetic Xbox compute shaders the Lanes self-test loads.

Both are small hand-written console programs, not taken from any game. One is a
`v_mov_b32` and `s_endpgm` with an empty root signature embedded. The other sums
the lane numbers of each 64-lane console wave and writes a word per thread.
"""
import hashlib
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.reference.scarlett_worker_fixture import container, pack, uav_shader  # noqa: E402

CODE = (0x7E0002FF, 0x12345678, 0xBF810000)
# Root signature 1.0 with no parameters, samplers or flags.
EMPTY_ROOT = pack([1, 0, 24, 0, 24, 0])


def shader():
    base = container(list(CODE))
    count = struct.unpack_from("<I", base, 28)[0]
    offsets = struct.unpack_from(f"<{count}I", base, 32)
    chunks = [base[start:start + 8 + struct.unpack_from("<I", base, start + 4)[0]] for start in offsets]
    chunks.append(b"RTS0" + pack([len(EMPTY_ROOT)]) + EMPTY_ROOT)
    start = 32 + 4 * len(chunks)
    table = []
    for chunk in chunks:
        table.append(start)
        start += len(chunk)
    return b"DXBC" + bytes(16) + pack([1, start, len(chunks), *table]) + b"".join(chunks)


# Two console waves in one group. v0 is the thread number.
WAVE_THREADS = 128
WAVE_CODE = (
    0xF4080402, 0xFA000000,  # s_load_dwordx4 s[16:19], s[4:5]: the u0 descriptor
    0xBF8CC07F,  # s_waitcnt lgkmcnt(0)
    # The 64-lane sum of v0, left in s7: an EXEC-saving prologue, DPP and
    # cross-row exchanges, then reads of lanes 31 and 63.
    0xBE8025C1, 0xD5010002, 0x00020080, 0x4A0404E9, 0xDE54C102, 0x4A0404E9, 0x93601202, 0x4A0404E9, 0x00092402,
    0x4A0404FA, 0xFF094002, 0xD7780003, 0x02010102, 0x4A040503, 0xD7600006, 0x00014102, 0x4A040406, 0xD7600007,
    0x00010102, 0xBEFE0400,
    0x4A080007,  # v_add_nc_u32 v4, s7, v0
    0x340A0082,  # v_lshlrev_b32 v5, 2, v0
    0xE0701000, 0x80040405,  # buffer_store_dword v4, v5, s[16:19], offen
    0xBF810000,  # s_endpgm
)


def wave_shader():
    return uav_shader(list(WAVE_CODE), width=WAVE_THREADS)


def wave_expected():
    """The word each thread writes: its number plus the sum over its wave."""
    return [n + sum(range(n & ~63, (n & ~63) + 64)) for n in range(WAVE_THREADS)]


def declaration(name, data):
    values = ",".join(str(byte) for byte in data)
    return (f"constexpr unsigned char {name}[] = {{{values}}};\n"
            "// The name its translation is cached under.\n"
            f"constexpr wchar_t {name}_digest[] = L\"{hashlib.sha256(data).hexdigest()}\";\n")


def header():
    words = ",".join(f"{word}u" for word in wave_expected())
    return ("// Generated from shader_fixture.py.\n"
            + declaration("xbox_shader", shader()) + declaration("wave_shader", wave_shader())
            + f"constexpr unsigned wave_shader_expected[] = {{{words}}};\n")


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(header())
