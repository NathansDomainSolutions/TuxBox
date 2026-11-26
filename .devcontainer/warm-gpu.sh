#!/usr/bin/env bash
# One-time warm-up for the GPU suites, run it inside the dev container before
# the first `tuxbox.py verify`.
#
# Two downloads are too big to happen inside a gate: verify gives every guest
# 15 seconds (verify.py: timeouts), while GE-Proton is ~600 MB and the umu
# runtime another ~700 MB. Without this step a fresh container fails all 67
# gates with "timed out after 15 seconds" and the failure looks like a broken
# GPU, so the two are fetched here, once, with a generous budget.
#
# It is idempotent: whatever is already installed is left alone.
set -euo pipefail

export UMU_RUNTIME_UPDATE="${UMU_RUNTIME_UPDATE:-0}"

# Must match PINNED_PROTON in project_tuxbox/host.py, otherwise verify would
# look for a different installation directory.
proton="${PROTON_NAME:-GE-Proton11-14}"
tools="${HOME}/.local/share/Steam/compatibilitytools.d"
proton_path="${tools}/${proton}-x86_64"
runtime="${HOME}/.local/share/umu/steamrt4"

echo "warm-gpu: user $(id -un), ${proton} in ${tools}"

# --- GE-Proton -----------------------------------------------------------------
if [[ -x "${proton_path}/proton" ]]; then
  echo "warm-gpu: ${proton} already installed"
else
  archive="${proton}-x86_64.tar.gz"
  url="https://github.com/GloriousEggroll/proton-ge-custom/releases/download/${proton}/${archive}"
  echo "warm-gpu: downloading ${proton} (~600 MB)"
  mkdir -p "${tools}"
  curl -fsSL -o "/tmp/${archive}" "${url}"
  tar -xzf "/tmp/${archive}" -C "${tools}"
  rm -f "/tmp/${archive}"
  [[ -x "${proton_path}/proton" ]] || { echo "warm-gpu: ${proton_path} missing after unpack" >&2; exit 1; }
fi

# --- umu runtime ----------------------------------------------------------------
# The runtime is installed by the first umu-run before it execs the game, so a
# path that does not exist is enough: umu fetches the runtime, creates the
# prefix, then fails to launch. That non-zero exit is expected.
if [[ -d "${runtime}" ]] && [[ -n "$(ls -A "${runtime}" 2>/dev/null)" ]]; then
  echo "warm-gpu: umu runtime already installed"
else
  echo "warm-gpu: fetching the umu runtime (~700 MB, a few minutes)"
  set +e
  PROTONPATH="${proton_path}" GAMEID=umu-default \
    timeout 1800 umu-run "${HOME}/.cache/tuxbox-warm-up.exe" >"${TMPDIR:-/tmp}/warm-gpu.log" 2>&1
  status=$?
  set -e
  [[ -d "${runtime}" ]] && [[ -n "$(ls -A "${runtime}" 2>/dev/null)" ]] || {
    echo "warm-gpu: the runtime did not install, see ${TMPDIR:-/tmp}/warm-gpu.log" >&2
    exit 1
  }
  echo "warm-gpu: runtime installed (umu-run exited ${status}, which is expected)"
fi

echo "warm-gpu: ready - run .venv/bin/python tuxbox.py verify --adapter 1002:73df"
