#!/usr/bin/env bash
# Download LLVM's RDNA2 (gfx10) disassembler test files for
# tools/shader-compiler/examples/llvm_mc_coverage.rs.
#
# The files are part of LLVM (Apache-2.0 WITH LLVM-exception); LLVM's
# LICENSE.TXT is saved next to them. They are fetched on demand, not committed.
#
# usage: tools/fetch_llvm_amdgpu_mc_tests.sh [dest dir]   (default: build/llvm-mc-gfx10)
set -euo pipefail
tag=llvmorg-18.1.8
dest=${1:-build/llvm-mc-gfx10}
base=https://raw.githubusercontent.com/llvm/llvm-project/$tag/llvm
files=(
  gfx10_ds.txt gfx10_exp.txt gfx10_flat.txt gfx10_mimg.txt gfx10_mtbuf.txt gfx10_mubuf.txt gfx10_smem.txt
  gfx10_sop1.txt gfx10_sop2.txt gfx10_sopc.txt gfx10_sopk.txt gfx10_sopp.txt
  gfx10_vop1.txt gfx10_vop1_dpp16.txt gfx10_vop1_dpp8.txt gfx10_vop1_sdwa.txt
  gfx10_vop2.txt gfx10_vop2_dpp16.txt gfx10_vop2_dpp8.txt gfx10_vop2_sdwa.txt
  gfx10_vop3.txt gfx10_vop3c.txt gfx10_vop3cx.txt gfx10_vopc.txt gfx10_vopc_sdwa.txt gfx10_vopcx.txt
  gfx1030_new.txt
)
mkdir -p "$dest"
curl -fsSL "$base/LICENSE.TXT" -o "$dest/LICENSE.TXT"
for f in "${files[@]}"; do
  curl -fsSL "$base/test/MC/Disassembler/AMDGPU/$f" -o "$dest/$f"
done
echo "LLVM $tag gfx10 disassembler tests saved in $dest (${#files[@]} files, license in LICENSE.TXT)"
