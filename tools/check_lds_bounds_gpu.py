#!/usr/bin/env python3
"""GPU check: group shared accesses past the allocation read 0 and drop writes.

RDNA2 returns 0 for an LDS read past the group's allocation and drops the
write. The translated program writes each lane's input to its own word, writes
a second value 32 KiB further on, and reads both back over its inputs. The
in-range word must keep the first value and the far read must return 0. An index that wraps
instead (the old `& 8191u`) overwrites the first value and reads it back.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.frosty_gpu_guest import run_gpu_guest

LANES = 64
# compute.cpp seeds and reads back exactly this many words, and the shader's
# `output` view starts at word START and covers 192 words.
WORDS = 384
START = 96
INITIAL = "    v0 = gtid.x; v1 = gtid.y; v2 = gtid.z;"
END = "    // 0x0028 s_endpgm"


def instrument(hlsl):
    if hlsl.count(INITIAL) != 1 or hlsl.count(END) != 1:
        raise ValueError("unexpected LDS bounds fixture")
    hlsl = hlsl.replace(INITIAL, f"    v0 = gtid.x * 4u; v1 = output[gtid.x]; v2 = output[{LANES}u + gtid.x];"
                                 " v3 = gtid.x * 4u + 0x8000u;")
    # Each lane's results overwrite its two inputs, keeping inside the 192-word view.
    hlsl = hlsl.replace(END, f"    output[gtid.x] = v4; output[{LANES}u + gtid.x] = v5;\n" + END)
    return "RWBuffer<uint> output : register(u0);\n" + hlsl


def cases():
    rng = random.Random(0x4C445342)
    yield [0x11110000 + lane for lane in range(LANES)], [0x22220000 + lane for lane in range(LANES)]
    yield [0xffffffff] * LANES, [0x00000001] * LANES
    for _ in range(6):
        yield ([rng.getrandbits(32) | 1 for _ in range(LANES)],
               [rng.getrandbits(32) | 1 for _ in range(LANES)])


def expected(near, far):
    seed = [0x76540000 + i for i in range(WORDS)]
    seed[START:START + 2 * LANES] = near + far
    # The lane's own word keeps its value and the read past the allocation is 0.
    want = seed.copy()
    want[START:START + 2 * LANES] = near + [0] * LANES
    return struct.pack(f"<{WORDS}I", *seed), struct.pack(f"<{WORDS}I", *want)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("emitter", "provider", "proton", "prefix", "compiler-dir", "dxc", "dxv", "work", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--adapter", required=True)
    a = ap.parse_args(argv)
    if a.work.exists() or a.output.exists():
        ap.error("fresh work/output required")
    a.work.mkdir(parents=True)
    with TemporaryDirectory(dir=a.work) as tmp:
        trial = Path(tmp)
        source = trial / "lds-bounds.hlsl"
        source.write_text(instrument(subprocess.check_output([a.emitter], text=True, timeout=15)))
        blob = trial / "lds-bounds.cso"
        subprocess.run([a.dxc, "-T", "cs_6_0", "-E", "main", "-Fo", blob, source], check=True, timeout=30)
        subprocess.run([a.dxv, blob], check=True, timeout=30)
        want = {}
        for i, (near, far) in enumerate(cases()):
            name = f"lds{i:02}"
            seed, want[name] = expected(near, far)
            (trial / (name + ".seed")).write_bytes(seed)
            (trial / (name + ".cso")).hardlink_to(blob)
        (trial / "cases.txt").write_text("".join(f"{name} 0 uint 1\n" for name in want))
        exe, status, fields = run_gpu_guest(ROOT / "tools/reference/independent-shaders/compute.cpp",
            "lds-bounds.exe", trial, a.work, a.proton, a.prefix, a.provider, a.compiler_dir,
            extra_env={"TUXBOX_NULL_PIPELINES": "0"})
        rows = []
        for name, reference in want.items():
            path = trial / (name + ".bin")
            data = path.read_bytes() if path.is_file() else b""
            row = {"case": name, "matched": data == reference}
            if data != reference:
                row.update(expected=reference.hex(), actual=data.hex())
            rows.append(row)
        report = dict(schema=1,
            scope="Translated LDS writes 32 KiB past a lane's word are dropped and read 0; the lane's own word keeps its value",
            passed=status == 0 and fields.get("adapter") == a.adapter and all(row["matched"] for row in rows),
            adapter=fields.get("adapter"), guest_exit=status, cases=len(rows), compared_words=len(rows) * WORDS,
            emitter_sha256=digest(a.emitter), dxc_sha256=digest(a.dxc), dxv_sha256=digest(a.dxv),
            provider_sha256=digest(a.provider), guest_sha256=digest(exe), shader_sha256=digest(blob), rows=rows)
        if fields.get("error"):
            report["error"] = fields["error"]
        with a.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(report["passed"], report["cases"], report["compared_words"], "failed",
          sum(not row["matched"] for row in rows))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
