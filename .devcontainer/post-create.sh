#!/usr/bin/env bash
# One-time container setup, run by the devcontainer postCreateCommand as the
# container user (not root).
#
# The image already has the C/C++ cross toolchain, DXC and umu-launcher; what
# is left is per-container state: the MSVC CRT/SDK that build.py expects under
# ~/.cache/cargo-xwin/xwin, and a virtualenv for the Python entry points.
#
# Re-running this script is safe: the xwin cache and the venv are reused.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PATH="${CARGO_HOME:-${HOME}/.cargo}/bin:/usr/local/cargo/bin:${PATH}"

if [[ "$(id -u)" == "0" ]]; then
  sudo=""
else
  sudo="sudo"
fi

# --- Rust --------------------------------------------------------------------
# The toolchain comes from the rust devcontainer feature and is installed as
# root, so hand the cargo/rustup trees to the container user once: cargo install
# writes there.
${sudo} chown -R "$(id -u):$(id -g)" /usr/local/cargo /usr/local/rustup 2>/dev/null || true

if ! command -v cargo-xwin >/dev/null 2>&1; then
  cargo install --locked cargo-xwin
fi

xwin_root="${XWIN_SDK_ROOT:-${HOME}/.cache/cargo-xwin/xwin}"
if [[ ! -f "${xwin_root}/sdk/include/um/windows.h" ]]; then
  cargo xwin cache xwin --xwin-arch x86_64
fi
if [[ ! -f "${xwin_root}/sdk/include/um/windows.h" ]]; then
  echo "post-create: MSVC SDK missing at ${xwin_root}/sdk/include/um/windows.h" >&2
  exit 1
fi

# --- Python ------------------------------------------------------------------
# PEP 668: the system interpreter refuses regular installs, and the project
# docs already use .venv (it is in .gitignore).
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

# --- Rust helpers the launcher builds on first use ---------------------------
# tuxbox.py run and the shader preparation compile two helper crates at launch
# time. Build them here, while the container still has network: in a fresh
# container the documented offline launch ("unshare --net -- tuxbox.py run")
# otherwise dies on "Could not resolve host: static.crates.io".
cargo build --locked --release --target "$(rustc -vV | sed -n 's/^host: //p')" \
  --manifest-path tools/shader-compiler/Cargo.toml --target-dir build/shader-compiler
cargo build --locked --release --target x86_64-pc-windows-msvc \
  --target-dir build/cargo --package tuxbox-pipeline-prep

# --- Smoke check -------------------------------------------------------------
echo "post-create: user $(id -un) ($(id -u):$(id -g))"
printf '  %-12s %s\n' rustc "$(rustc --version)"
printf '  %-12s %s\n' windows-target "$(rustc --print target-list | grep -qx x86_64-pc-windows-msvc && echo x86_64-pc-windows-msvc)"
printf '  %-12s %s\n' clang-cl "$(clang-cl --version | head -1)"
printf '  %-12s %s\n' lld-link "$(lld-link --version | head -1)"
printf '  %-12s %s\n' dxc "$(dxc --version | head -1)"
printf '  %-12s %s\n' xwin-sdk "${xwin_root}"
printf '  %-12s %s\n' umu-run "$(umu-run --version)"
printf '  %-12s %s\n' python "$(.venv/bin/python --version)"

if compgen -G '/dev/dri/renderD*' >/dev/null; then
  echo "post-create: render nodes: $(ls /dev/dri | tr '\n' ' ')"
  echo "post-create: before GPU runs, warm Proton and the umu runtime: bash .devcontainer/warm-gpu.sh"
else
  echo "post-create: no /dev/dri render nodes - build and 'check --no-gpu' work, GPU runs do not"
fi
