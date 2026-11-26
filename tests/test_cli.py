from argparse import Namespace
from pathlib import Path
import unittest
from unittest.mock import patch

import tuxbox


class MisplacedLauncherOptions(unittest.TestCase):
    OPTIONS = {"--mangohud", "--log-file", "--offline"}

    def test_launcher_option_after_the_target_is_reported(self):
        self.assertEqual(tuxbox.misplaced_option(["--mangohud"], self.OPTIONS), "--mangohud")
        self.assertEqual(tuxbox.misplaced_option(["-x", "--log-file=a.log"], self.OPTIONS), "--log-file=a.log")

    def test_game_arguments_pass_through(self):
        self.assertIsNone(tuxbox.misplaced_option([], self.OPTIONS))
        self.assertIsNone(tuxbox.misplaced_option(["--replay", "--report", "r.txt", "-nullrhi"], self.OPTIONS))


class ProjectScarlettLaunch(unittest.TestCase):
    def launch(self, executable, backend=None, offline=False):
        args = Namespace(target=Path("stage"), app_id=None, backend=backend, proton=None, prefix=None,
                         offline=offline, graphics_menu=False, graphics_profile=None)
        with patch("tuxbox.launch_target", return_value=Path("stage") / executable), \
             patch("tuxbox.package_root", return_value=Path("stage")), \
             patch("tuxbox.Host") as host, patch("tuxbox.run_game", return_value=0):
            tuxbox.launch_game(args)
        return host.call_args.kwargs["offline"]

    def test_is_always_offline_on_linux(self):
        self.assertTrue(self.launch("game_scarlett_master.exe", backend="proton"))
        self.assertTrue(self.launch("Game_Scarlett_Master.exe", backend="wine"))

    def test_is_refused_where_it_cannot_be_isolated(self):
        with self.assertRaisesRegex(ValueError, "without network access"):
            self.launch("game_scarlett_master.exe", backend="native")

    def test_other_titles_choose(self):
        self.assertFalse(self.launch("Lanes.exe", backend="proton"))
        self.assertTrue(self.launch("Lanes.exe", backend="proton", offline=True))
