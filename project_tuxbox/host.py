"""Launch through the host PE loader, or through a selected userspace Wine provider."""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from contextlib import nullcontext

from .backend_log import game_diagnostics
from .console_cbv import configure_console_cbv
from .mangohud import performance_capture
from .pipeline_profile import pipeline_capture
from .runtime_shaders import runtime_shader_compiler


def proton_installation(selection):
    if selection:
        path = Path(selection).expanduser().resolve()
        return path.parent if path.is_file() and path.name == "proton" else path
    steam = Path(os.environ.get("STEAM_COMPAT_CLIENT_INSTALL_PATH") or Path.home() / ".steam/root")
    libraries = [steam, Path.home() / ".local/share/Steam"]
    vdf = steam / "steamapps/libraryfolders.vdf"
    if vdf.is_file():
        libraries += [Path(p) for p in re.findall(r'"path"\s*"([^"]+)"', vdf.read_text(errors="replace"))]
    for library in libraries:
        candidate = library / "steamapps/common/Proton - Experimental"
        if (candidate / "proton").is_file():
            return candidate.resolve()
    local = steam / "compatibilitytools.d/Proton - Experimental"
    if (local / "proton").is_file():
        return local.resolve()
    raise ValueError("Select an installed Proton distribution with --proton or PROTONPATH")


PACKAGE_DRIVE_RECORD = "tuxbox-package-drive.json"


def claim_package_drive(prefix, package):
    """Record the prefix's G: links before the GDK provider replaces them with the package.

    Returns False, and records nothing, when G: is not a drive link TuxBox can put back.
    """
    links = {}
    for name in ("g:", "g::"):
        link = prefix / "dosdevices" / name
        if link.is_symlink():
            links[name] = os.readlink(link)
        elif os.path.lexists(link):
            return False
        else:
            links[name] = None
    (prefix / PACKAGE_DRIVE_RECORD).write_text(json.dumps({"package": str(package), "links": links}))
    return True


def restore_package_drive(prefix, report):
    """Put back the G: links a launch replaced, including after an interrupted launch."""
    record_path = prefix / PACKAGE_DRIVE_RECORD
    try:
        record = json.loads(record_path.read_text())
        package, links = record["package"], record["links"]
    except FileNotFoundError:
        return
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report(f"Ignoring an unreadable G: record {record_path}: {exc}")
        record_path.unlink(missing_ok=True)
        return
    try:
        for name, target in links.items():
            link = prefix / "dosdevices" / name
            # Leave a link that something other than the provider changed meanwhile.
            if link.is_symlink() and os.readlink(link) != package:
                continue
            if os.path.lexists(link) and not link.is_symlink():
                continue
            link.unlink(missing_ok=True)
            if target is not None:
                link.symlink_to(target)
        record_path.unlink()
    except OSError as exc:
        report(f"Could not restore the prefix's G: drive from {record_path}: {exc}")


class Host:
    def __init__(self, backend, proton, prefix, offline=False):
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise ValueError("This native CPU baseline requires an x86-64 host")
        self.backend = backend or ("native" if os.name == "nt" else "proton")
        self.env = os.environ.copy()
        self.command = []
        self.version = platform.platform()
        self.offline = False
        if self.backend == "native":
            if offline:
                raise ValueError("--offline requires the Wine or Proton backend on Linux")
            if os.name != "nt" or proton or prefix:
                raise ValueError("Native execution requires Windows and takes no Proton or prefix arguments")
            return
        if sys.platform != "linux":
            raise ValueError("The Wine and Proton backends require Linux")
        if self.backend == "proton":
            installation = proton_installation(proton or os.getenv("PROTONPATH"))
            if not (installation / "proton").is_file():
                raise ValueError(f"Missing Proton distribution launcher in {installation}")
            tool = "umu-run"
            self.env.update(PROTONPATH=str(installation), GAMEID="umu-default")
            version = installation / "version"
            self.version = version.read_text().strip() if version.is_file() else installation.name
            prefix_name = installation.name
        else:
            if proton:
                raise ValueError("--proton selects a distribution for the proton backend")
            tool = "wine"
            prefix_name = "wine"
        executable = shutil.which(tool)
        if not executable:
            raise ValueError(f"Install {tool} to use the {self.backend} backend")
        if self.backend == "wine":
            self.version = subprocess.check_output([executable, "--version"], text=True).strip()
        data_home = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
        prefix = (prefix or data_home / "project-tuxbox/prefixes" / prefix_name).expanduser().resolve()
        prefix.mkdir(parents=True, exist_ok=True)
        self.env["WINEPREFIX"] = str(prefix)
        self.command = [executable]
        if offline:
            self.isolate_network()

    def isolate_network(self):
        # A new network namespace has only a loopback device that is down, so the
        # game and its crash handler cannot upload minidumps or reach any server.
        unshare = shutil.which("unshare")
        if not unshare:
            raise ValueError("Install util-linux unshare to launch with --offline")
        wrapper = [unshare, "--user", "--map-current-user", "--net", "--"]
        probe = subprocess.run([*wrapper, "true"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if probe.returncode:
            reason = probe.stderr.strip() or f"exit code {probe.returncode}"
            raise ValueError(f"Cannot create an offline network namespace: {reason}")
        self.command = [*wrapper, *self.command]
        # umu-run otherwise tries to update the Steam Runtime before starting.
        self.env["UMU_RUNTIME_UPDATE"] = "0"
        self.offline = True

    def run(self, executable, arguments, *, timeout=None, capture=False):
        options = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT} if capture else {}
        return subprocess.run([*self.command, str(executable), *arguments], cwd=executable.parent,
                              env=self.env, timeout=timeout, **options)

    def run_game(self, executable, arguments, log, *, package=None, mangohud=False, profile_pipelines=False):
        env = self.env.copy()
        env.setdefault("TUXBOX_LOG", "stderr" if self.backend == "native" else "1")
        env.pop("TUXBOX_WINE_TITLE_TEMP", None)
        env.pop("TUXBOX_WINE_PACKAGE_DRIVE", None)
        drive_prefix = None
        if self.backend in ("proton", "wine") and env.get("WINEPREFIX"):
            prefix = Path(env["WINEPREFIX"])
            drive = prefix / "dosdevices/t:"
            # Proton can remove a user-configured T: during startup. Give the GDK
            # provider its original target so it can reattach it after that point.
            if drive.is_dir():
                env["TUXBOX_WINE_TITLE_TEMP"] = str(drive.resolve())
            elif not os.path.lexists(drive) and not os.path.lexists(drive.with_name("t::")):
                storage = prefix / "drive_t"
                try:
                    storage.mkdir(mode=0o700, exist_ok=True)
                except OSError as exc:
                    log.status(f"Could not prepare the title scratch directory {storage}: {exc}")
                else:
                    env["TUXBOX_WINE_TITLE_TEMP"] = str(storage)
            # Xbox titles open their package files through G:, as on the console. The GDK
            # provider replaces the prefix's G: with the package, so record the existing
            # links and put them back when the title exits.
            restore_package_drive(prefix, log.status)
            if package is not None:
                package = Path(package).resolve()
                try:
                    claimed = claim_package_drive(prefix, package)
                except OSError as exc:
                    log.status(f"Could not record the prefix's G: drive: {exc}")
                    claimed = False
                if claimed:
                    env["TUXBOX_WINE_PACKAGE_DRIVE"] = str(package)
                    drive_prefix = prefix
                else:
                    drive = prefix / "dosdevices/g:"
                    log.status(f"The package is not mapped to G: because {drive} is not a drive link")
        if "-nullrhi" not in (arg.lower() for arg in arguments):
            configure_console_cbv(self.backend, env, executable, log.status)
        capture = performance_capture(self.backend, env, executable, log) if mangohud else nullcontext()
        pipelines = pipeline_capture(self.backend, env, log) if profile_pipelines else nullcontext()
        try:
            with capture, pipelines, runtime_shader_compiler(self.backend, env, executable, log.status), \
                 game_diagnostics(self.backend, env, executable.parent, log.stderr_bytes,
                                  log_path=log.path) as stderr:
                return log.run([*self.command, str(executable), *arguments], cwd=executable.parent, env=env,
                               stderr_bytes=stderr)
        finally:
            if drive_prefix is not None:
                restore_package_drive(drive_prefix, log.status)
