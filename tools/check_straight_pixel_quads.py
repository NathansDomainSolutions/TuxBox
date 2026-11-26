#!/usr/bin/env python3
"""Check synthetic nonheap FI=1 DPP16 pixel X/Y swaps with GPU readback.

The check compares full-target and scissored draws with an integer CPU oracle,
and records whether translation rejects a control with a guarded prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.frosty_gpu_guest import run_gpu_guest

SIDE = 16
GUARD = (17, 29, 43, 71)


def sha(path):
    """Return the SHA-256 identity of one tool, shader, or readback file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(variant, clipped):
    """Return expected little-endian RGBA uints for one draw.

    ``variant`` selects copy-style or in-place X/Y swaps. ``clipped`` preserves
    the clear color outside the draw rectangle, including helper lanes whose
    positions remain defined by the oversized triangle.
    """
    def value(x, y):
        """Encode integer pixel coordinates as a distinct per-pixel value."""
        return 99 + x + 32 * y

    pixels = []
    for y in range(SIDE):
        for x in range(SIDE):
            original = value(x, y)
            across_x = value(x ^ 1, y)
            across_y = value(x, y ^ 1) + value(x ^ 1, y ^ 1)
            total = across_y + (original + across_x if variant == "copy" else across_x)
            color = (across_x, across_y, total, original)
            if clipped and not (3 <= x < 11 and 3 <= y < 11):
                color = GUARD
            pixels.append(struct.pack("<4I", *color))
    return b"".join(pixels)


def main(argv=None):
    """Compile emitter output, run four GPU draws, and write a JSON report.

    Required paths identify the emitter, provider, Proton files, source prefix,
    compiler DLLs, work directory, and new report file; ``--adapter`` selects
    the expected GPU. The report records shader and tool hashes, guest status,
    guarded-prefix refusal, and exact results for four 16-by-16 readbacks.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("emitter", "provider", "proton", "prefix", "compiler-dir", "dxc", "dxv", "work", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--adapter", required=True)
    a = ap.parse_args(argv)
    if a.work.exists() or a.output.exists():
        ap.error("fresh work/output required")
    a.work.mkdir(parents=True)
    fingerprints = {name: sha(getattr(a, name)) for name in ("emitter", "provider", "dxc", "dxv")}
    rows, shaders = [], {}
    with TemporaryDirectory(dir=a.work) as tmp:
        trial = Path(tmp)
        negative = subprocess.run([a.emitter, "guarded"], capture_output=True, text=True, timeout=15)
        refused = negative.returncode == 1 and not negative.stdout and (
            "wave operation requires complete 32-lane compute waves" in negative.stderr)
        sources = {"vertex": "float4 main(uint id:SV_VertexID):SV_Position { return float4(id==2?3:-1,id==1?3:-1,0.5,1); }"}
        for variant in ("copy", "inplace"):
            sources[variant] = subprocess.check_output([a.emitter, variant], text=True, timeout=15)
        for name, source in sources.items():
            hlsl = trial / (name + ".hlsl")
            hlsl.write_text(source)
            blob = trial / (name + ".cso")
            subprocess.run([a.dxc, "-T", "vs_6_0" if name == "vertex" else "ps_6_6", "-E", "main", "-O3", "-Fo", blob, hlsl],
                           check=True, timeout=30)
            subprocess.run([a.dxv, blob], check=True, timeout=30)
            shaders[name] = {"hlsl_sha256": sha(hlsl), "dxil_sha256": sha(blob)}
        exe, status, fields = run_gpu_guest(
            ROOT / "tools/reference/straight-pixel-quads/main.cpp", "straight-pixel-quads.exe",
            trial, a.work, a.proton, a.prefix, a.provider, a.compiler_dir,
            extra_env={"TUXBOX_NULL_PIPELINES": "0"})
        for variant in ("copy", "inplace"):
            for clipped in (False, True):
                name = variant + ("_clipped" if clipped else "_full")
                want = expected(variant, clipped)
                path = trial / (name + ".bin")
                data = path.read_bytes() if path.is_file() else b""
                row = dict(case=name, matched=data == want, expected_sha256=hashlib.sha256(want).hexdigest(),
                           actual_sha256=hashlib.sha256(data).hexdigest(), actual_bytes=len(data))
                if len(data) == len(want) and data != want:
                    offset = next(i for i in range(0, len(want), 16) if data[i:i+16] != want[i:i+16])
                    row.update(first_mismatch_pixel=[offset // 16 % SIDE, offset // 16 // SIDE],
                               expected=list(struct.unpack_from("<4I", want, offset)),
                               actual=list(struct.unpack_from("<4I", data, offset)))
                rows.append(row)
        unchanged = fingerprints == {name: sha(getattr(a, name)) for name in fingerprints}
        report = dict(schema=1, scope="Synthetic nonheap FI=1 DPP16 pixel axis swaps",
                      passed=status == 0 and fields.get("adapter") == a.adapter.lower() and unchanged
                             and refused and all(row["matched"] for row in rows),
                      adapter=fields.get("adapter"), guest_exit=status, cases=len(rows), pixels=SIDE*SIDE*len(rows),
                      guarded_refused=refused, guarded_exit=negative.returncode, guarded_error=negative.stderr.strip(),
                      tools_unchanged=unchanged, tools_sha256=fingerprints, guest_sha256=sha(exe), shaders=shaders, rows=rows)
        if fields.get("error"):
            report["error"] = fields["error"]
        with a.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(report["passed"], report["cases"], report["pixels"], "guarded_refused", refused,
          "failures", sum(not row["matched"] for row in rows), report.get("error", ""))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
