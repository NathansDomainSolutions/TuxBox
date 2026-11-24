from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from project_tuxbox.console_cbv import configure_console_cbv, console_cbv_layer
from project_tuxbox.host import Host, claim_package_drive, proton_installation


class SelectProton(unittest.TestCase):
    def test_explicit_runner_in_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="provider with spaces ") as tmp:
            install = Path(tmp).resolve()
            runner = install / "proton"
            runner.touch()
            self.assertEqual(proton_installation(str(runner)), install)
            self.assertEqual(proton_installation(str(install)), install)

    def test_registered_steam_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "steamapps").mkdir()
            library = root / "Steam Library"
            install = library / "steamapps/common/Proton - Experimental"
            install.mkdir(parents=True)
            (install / "proton").touch()
            (root / "steamapps/libraryfolders.vdf").write_text(f'"libraryfolders" {{ "0" {{ "path" "{library}" }} }}')
            with patch.dict(os.environ, {"STEAM_COMPAT_CLIENT_INSTALL_PATH": str(root)}), patch.object(Path, "home", return_value=root):
                self.assertEqual(proton_installation(None), install)

    @unittest.skipUnless(sys.platform == "linux", "Wine prefix drive links and POSIX modes require Linux")
    def test_linux_title_drive_uses_prefix_and_preserves_existing_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            drive = prefix / "dosdevices/t:"
            drive.parent.mkdir()
            host = Host.__new__(Host)
            host.backend, host.command = "proton", ["umu-run"]
            host.env = {"WINEPREFIX": tmp, "TUXBOX_RUNTIME_SHADERS": "0"}
            log = Mock()

            def configured_target():
                host.run_game(prefix / "game.exe", ["-nullrhi"], log)
                return log.run.call_args.kwargs["env"].get("TUXBOX_WINE_TITLE_TEMP")

            scratch = prefix / "drive_t"
            self.assertEqual(configured_target(), str(scratch))
            self.assertEqual(scratch.stat().st_mode & 0o777, 0o700)
            drive.symlink_to("../custom")
            self.assertIsNone(configured_target())
            (prefix / "custom").mkdir()
            self.assertEqual(configured_target(), str(prefix / "custom"))
            host.backend = "native"
            self.assertIsNone(configured_target())

    @unittest.skipUnless(sys.platform == "linux", "Wine prefix drive links require Linux")
    def test_linux_package_drive_is_restored_after_the_title_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            prefix, package, other = root / "prefix", root / "game", root / "user-g"
            dosdevices = prefix / "dosdevices"
            for folder in (package, other, dosdevices):
                folder.mkdir(parents=True)
            host = Host.__new__(Host)
            host.backend, host.command = "proton", ["umu-run"]
            host.env = {"WINEPREFIX": str(prefix), "TUXBOX_RUNTIME_SHADERS": "0"}
            log = Mock()
            drive = dosdevices / "g:"

            def provider_mounts_package(*args, **kwargs):
                # Stand in for the GDK provider's mount manager call.
                if "TUXBOX_WINE_PACKAGE_DRIVE" not in kwargs["env"]:
                    return
                drive.unlink(missing_ok=True)
                (dosdevices / "g::").unlink(missing_ok=True)
                drive.symlink_to(kwargs["env"]["TUXBOX_WINE_PACKAGE_DRIVE"])

            def launch(package_dir):
                log.run.side_effect = provider_mounts_package
                host.run_game(package / "game.exe", ["-nullrhi"], log, package=package_dir)
                return log.run.call_args.kwargs["env"].get("TUXBOX_WINE_PACKAGE_DRIVE")

            # A prefix without G: gets the package during the launch and no G: after it.
            self.assertEqual(launch(package), str(package))
            self.assertFalse(os.path.lexists(drive))
            self.assertIsNone(launch(None))

            # The user's own G: and its device link come back after the title exits.
            drive.symlink_to(other)
            (dosdevices / "g::").symlink_to("/dev/sdz1")
            self.assertEqual(launch(package), str(package))
            self.assertEqual(os.readlink(drive), str(other))
            self.assertEqual(os.readlink(dosdevices / "g::"), "/dev/sdz1")
            self.assertFalse((prefix / "tuxbox-package-drive.json").exists())

            # A launch killed before it could restore G: is repaired by the next one.
            (dosdevices / "g::").unlink()
            self.assertTrue(claim_package_drive(prefix, package))
            drive.unlink()
            drive.symlink_to(package)
            self.assertIsNone(launch(None))
            self.assertEqual(os.readlink(drive), str(other))
            self.assertFalse(os.path.lexists(dosdevices / "g::"))

            # A G: that is not a link is left alone and the package is not mapped.
            drive.unlink()
            drive.mkdir()
            self.assertIsNone(launch(package))
            self.assertTrue(drive.is_dir())
            host.backend = "native"
            self.assertIsNone(launch(package))


@unittest.skipUnless(sys.platform == "linux", "Network namespaces require Linux")
class OfflineLaunch(unittest.TestCase):
    def host(self, probe_code=0, stderr=""):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        probe = subprocess.CompletedProcess([], probe_code, stderr=stderr)
        with patch("project_tuxbox.host.shutil.which", side_effect=lambda tool: f"/usr/bin/{tool}"), \
             patch("project_tuxbox.host.subprocess.check_output", return_value="wine-10.0\n"), \
             patch("project_tuxbox.host.subprocess.run", return_value=probe) as run:
            host = Host("wine", None, Path(tmp.name), offline=True)
        return host, run

    def test_wraps_every_launch_in_a_network_namespace(self):
        host, run = self.host()
        wrapper = ["/usr/bin/unshare", "--user", "--map-current-user", "--net", "--"]
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [*wrapper, "true"])
        self.assertEqual(host.command, [*wrapper, "/usr/bin/wine"])
        self.assertEqual(host.env["UMU_RUNTIME_UPDATE"], "0")
        self.assertTrue(host.offline)

    def test_refuses_to_launch_online_when_isolation_fails(self):
        with self.assertRaisesRegex(ValueError, "Operation not permitted"):
            self.host(1, "unshare: unshare failed: Operation not permitted\n")

    def test_native_backend_rejects_offline(self):
        with patch("project_tuxbox.host.platform.machine", return_value="x86_64"), \
             self.assertRaisesRegex(ValueError, "--offline"):
            Host("native", None, None, offline=True)


class ConsoleConstantBuffers(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stage = self.root / "Sample"
        self.stage.mkdir()
        self.exe = self.stage / "game.exe"
        self.built = self.root / "build/d3d12-smoke/d3d12_xs.dll"
        self.built.parent.mkdir(parents=True)
        self.built.write_bytes(b"a built TuxBox provider")
        self.provider = self.stage / "D3D12_XS.DLL"
        self.provider.write_bytes(self.built.read_bytes())
        self.layer = str(self.root / "build/frosty-cbv-layer")
        self.log = Mock()
        for target, value in (("project_tuxbox.shader_prep.TUXBOX_PROVIDER", self.built),
                              ("project_tuxbox.shader_prep.ROOT", self.root)):
            override = patch(target, value)
            override.start()
            self.addCleanup(override.stop)

    def test_direct_stock_launch_preserves_user_settings_and_host_environment(self):
        host = Host.__new__(Host)
        host.backend, host.command = "proton", ["umu-run"]
        host.env = {"PROTONPATH": str(self.root), "VK_LAYER_PATH": "/user/layers",
                    "VK_ADD_LAYER_PATH": "/user/extra", "VK_INSTANCE_LAYERS": "overlay",
                    "VKD3D_CONFIG": "force_static_cbv", "VKD3D_DISABLE_EXTENSIONS": "first;second",
                    "PRESSURE_VESSEL_FILESYSTEMS_RO": "/user/mount", "TUXBOX_RUNTIME_SHADERS": "0"}
        layer = "/tuxbox/build/frosty-cbv-layer"
        with patch("project_tuxbox.console_cbv.console_cbv_layer", return_value=layer):
            host.run_game(self.exe, [], self.log)
            env = self.log.run.call_args.kwargs["env"]
            self.assertEqual(env["VK_LAYER_PATH"], layer + ":/user/layers")
            self.assertEqual(env["VK_ADD_LAYER_PATH"], "/user/extra")
            self.assertEqual(env["VK_INSTANCE_LAYERS"], "VK_LAYER_TUXBOX_console_cbv:overlay")
            self.assertEqual(env["VKD3D_DISABLE_EXTENSIONS"], "VK_EXT_descriptor_buffer,first;second")
            self.assertEqual(env["VKD3D_CONFIG"], "force_static_cbv")
            self.assertEqual(env["PRESSURE_VESSEL_FILESYSTEMS_RO"], layer + ":/user/mount")
            before = env.copy()
            configure_console_cbv("proton", env, self.exe, self.log.status)
            self.assertEqual(env, before)
            self.assertEqual(host.env["VK_LAYER_PATH"], "/user/layers")
            host.run_game(self.exe, ["-NullRHI"], self.log)
            self.assertEqual(self.log.run.call_args.kwargs["env"]["VK_INSTANCE_LAYERS"], "overlay")

    def test_explicit_fork_feature_and_wine_with_an_older_staged_copy(self):
        sample = self.root / "build/lanes/d3d12_xs.dll"
        sample.parent.mkdir(parents=True)
        sample.write_bytes(b"older sample")
        self.provider.write_bytes(sample.read_bytes())
        (self.root / "tuxbox-features").write_text("cbv_16_byte_alignment\n")
        env = {"PROTONPATH": str(self.root), "VKD3D_DISABLE_EXTENSIONS": "first;VK_EXT_descriptor_buffer",
               "VKD3D_CONFIG": "other"}
        with patch("project_tuxbox.console_cbv.console_cbv_layer", return_value=self.layer) as layer:
            configure_console_cbv("proton", env, self.exe, self.log.status)
            self.assertEqual(env["VKD3D_CONFIG"], "cbv_16_byte_alignment,other")
            self.assertEqual(env["VKD3D_DISABLE_EXTENSIONS"], "first;VK_EXT_descriptor_buffer")
            layer.assert_not_called()
            wine = {"PROTONPATH": str(self.root)}
            configure_console_cbv("wine", wine, self.exe, self.log.status)
            self.assertEqual(wine["VK_ADD_LAYER_PATH"], self.layer)
            self.assertNotIn("VKD3D_CONFIG", wine)
            self.assertNotIn("PRESSURE_VESSEL_FILESYSTEMS_RO", wine)

    def test_native_foreign_missing_and_explicit_comparison(self):
        with patch("project_tuxbox.console_cbv.console_cbv_layer", return_value=self.layer) as layer:
            env = {}
            configure_console_cbv("native", env, self.exe, self.log.status)
            self.provider.write_bytes(b"unknown")
            configure_console_cbv("proton", env, self.exe, self.log.status)
            self.provider.unlink()
            configure_console_cbv("proton", env, self.exe, self.log.status)
            self.assertEqual(env, {})
            layer.assert_not_called()
            env["TUXBOX_REFERENCE_CBV"] = "1"
            configure_console_cbv("proton", env, self.exe, self.log.status)
            self.assertEqual(env["VK_INSTANCE_LAYERS"], "VK_LAYER_TUXBOX_console_cbv")

    def test_rebuilds_missing_or_outdated_layer(self):
        sources = self.root / "tools/reference/cbv-layer"
        sources.mkdir(parents=True)
        script = self.root / "tools/build_frosty_cbv_layer.sh"
        layer = Path(self.layer)
        layer.mkdir(parents=True)
        for path in (script, sources / "layer.c", sources / "frosty.json"):
            path.touch()
            os.utime(path, (10, 10))
        env = {"CC": "clang"}
        with patch("project_tuxbox.console_cbv.ROOT", self.root), \
             patch("project_tuxbox.console_cbv.subprocess.run") as build:
            self.assertEqual(console_cbv_layer(env), self.layer)
            build.assert_called_once_with([script, layer], env=env, check=True)
            build.reset_mock()
            for path in (layer / "frosty.json", layer / "frosty_cbv.so"):
                path.touch()
                os.utime(path, (20, 20))
            console_cbv_layer(env)
            build.assert_not_called()
            os.utime(sources / "layer.c", (30, 30))
            console_cbv_layer(env)
            build.assert_called_once_with([script, layer], env=env, check=True)
