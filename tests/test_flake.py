import re
import unittest
from pathlib import Path

from project_tuxbox.host import PINNED_PROTON

ROOT = Path(__file__).resolve().parents[1]


class FlakeTests(unittest.TestCase):
    def test_dev_shell_pins_the_launcher_default_proton(self):
        flake = (ROOT / "flake.nix").read_text(encoding="utf-8")
        pinned = set(re.findall(r"GE-Proton\d+-\d+", flake))
        self.assertEqual(pinned, {PINNED_PROTON})
        self.assertIn("PROTONPATH", flake)


if __name__ == "__main__":
    unittest.main()
