from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from tools.triage_scarlett_shaders import command, construct, main, normalize, requests

KEY = "ab" * 32


@unittest.skipUnless(sys.platform == "linux", "The fake compiler is a POSIX shell script")
class TriageScarlettShaders(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "requests"
        self.cache.mkdir()

    def test_masks_counters_and_registers(self):
        self.assertEqual(normalize("warning\ns_load at 0xec: dynamic soffset unsupported\n"),
                         "s_load at 0x_: dynamic soffset unsupported")
        self.assertEqual(normalize("s_buffer_load at 0x108: base s48 is not a buffer descriptor (Some(Table(7)))"),
                         "s_buffer_load at 0x_: base sN is not a buffer descriptor (Some(Table(N)))")
        self.assertEqual(normalize("unmapped descriptor at s9 + 0xe0"), "unmapped descriptor at sN + 0x_")
        self.assertEqual(normalize("MUBUF with a dynamic SOFFSET is unsupported [buffer_load_dwordx3 @ 0x1a0]"),
                         "MUBUF with a dynamic SOFFSET is unsupported")
        self.assertEqual(normalize(""), "(no message)")

    def test_groups_by_named_construct(self):
        self.assertEqual(construct("wave reduction at 0x_: unsupported reduction op v_add_fN"), "wave reduction")
        self.assertEqual(construct("unverified mesh shader bindings"), "unverified mesh shader bindings")
        self.assertEqual(construct("unsupported Hull program: no emitter"), "unsupported Hull program")

    def test_mirrors_runtime_request_flags(self):
        work = Path("/w")
        stencil = self.cache / f"{KEY}_rt3.stencil"
        self.assertEqual(command("x", stencil, work)[1:4], [stencil, "--stencil-target", "3"])
        link = self.cache / "_".join([KEY] * 3)
        link = link.with_name(link.name + ".link")
        self.assertEqual(command("x", link, work)[1:6],
                         [link / "producer.dxbc", "--link-pixel", link / "pixel.dxbc", "--pixel-order", link / "pixel.cso"])
        self.assertIn("--hlsl-out", command("x", self.cache / f"{KEY}.dxbc", work))

    def test_lists_only_recorded_requests(self):
        for name in (f"{KEY}.dxbc", f"{KEY}_rt1.stencil", "notes.dxbc", f"{KEY}.cso"):
            (self.cache / name).write_bytes(b"")
        self.assertEqual([p.name for p in requests(self.cache)], [f"{KEY}.dxbc", f"{KEY}_rt1.stencil"])

    def test_ranks_refusals_and_separates_failures(self):
        names = {"1" * 64: 0, "2" * 64: 65, "3" * 64: 65, "4" * 64: 65, "5" * 64: 1}
        for key in names:
            (self.cache / f"{key}.dxbc").write_bytes(key.encode())
        fake = self.root / "xshader"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$(cat "$1")" in\n'
            "  1*) exit 0;;\n"
            "  2*) echo 's_load at 0xec: dynamic soffset unsupported' >&2; exit 65;;\n"
            "  3*) echo 's_load at 0xb0: dynamic soffset unsupported' >&2; exit 65;;\n"
            "  4*) echo 'wave reduction at 0x10 may execute in divergent control flow' >&2; exit 65;;\n"
            "  *) echo 'expected a cooked shader path' >&2; exit 1;;\n"
            "esac\n")
        fake.chmod(0o755)
        output = self.root / "report.json"
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main([str(self.cache), "--xshader", str(fake), "--jobs", "2", "--json", str(output)]), 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["totals"], {"requests": 5, "translated": 1, "refused": 3, "failed": 1})
        self.assertEqual(report["reasons"][0]["reason"], "s_load at 0x_: dynamic soffset unsupported")
        self.assertEqual(report["reasons"][0]["count"], 2)
        self.assertEqual(report["constructs"][0], {"construct": "s_load", "count": 2})
        self.assertEqual(report["failures"], [{"reason": "expected a cooked shader path", "count": 1}])
        self.assertIn("| `s_load` | 2 | 66.7% |", stdout.getvalue())
        # The cache is read only: no markers or outputs appear beside the requests.
        self.assertEqual(len(os.listdir(self.cache)), 5)


if __name__ == "__main__":
    unittest.main()
