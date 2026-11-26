# syntax=docker/dockerfile:1
#
# TuxBox development container: cross-compiles the Windows x64 providers, builds
# the launcher's Vulkan layer and runs the guests through Proton on the host's
# GPU. It carries everything the Lanes build requirements ask for except
# Proton, which is ~600 MB and version-pinned per run (see warm-gpu.sh).
#
# Versions are pinned in the ARGs below so a container created months apart
# still matches CI. Bump the ARGs, not the commands.

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Microsoft's Linux DirectX Shader Compiler: distro builds abort on every
# shader (build.py probes for that), and the shader prep also runs the DXIL
# validator that ships beside it.
ARG DXC_VERSION=v1.9.2609
ARG DXC_ARCHIVE=linux_dxc_2026_09_28.x86_x64.tar.gz

# The MSVC standard library rejects compilers older than Clang 19 (STL1000 in
# yvals_core.h) and Ubuntu 24.04 ships clang 18, so LLVM comes from
# apt.llvm.org. That also has the clang-cl and lld-link launchers Ubuntu's
# packages leave out.
ARG LLVM_VERSION=21

# umu-launcher publishes per-distro packages rather than an archive build; the
# noble .deb is the one meant for this image.
ARG UMU_VERSION=1.4.4
ARG UMU_DEB=python3-umu-launcher_${UMU_VERSION}-1_amd64_ubuntu-noble.deb

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1. LLVM from apt.llvm.org, with the launchers on PATH.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /etc/apt/trusted.gpg.d/llvm.asc \
 && echo "deb https://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
      > /etc/apt/sources.list.d/llvm.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      clang-${LLVM_VERSION} \
      lld-${LLVM_VERSION} \
      llvm-${LLVM_VERSION} \
 && ln -sf /usr/lib/llvm-${LLVM_VERSION}/bin/clang /usr/local/bin/clang-cl \
 && ln -sf /usr/lib/llvm-${LLVM_VERSION}/bin/lld-link /usr/local/bin/lld-link \
 && ln -sf /usr/lib/llvm-${LLVM_VERSION}/bin/llvm-mc /usr/local/bin/llvm-mc \
 && ln -sf /usr/lib/llvm-${LLVM_VERSION}/bin/llvm-objdump /usr/local/bin/llvm-objdump \
 && clang-cl --version | head -1 \
 && lld-link --version \
 && rm -rf /var/lib/apt/lists/*

# 2. Build tools, Python, Vulkan and the X side of the headless display.
#
#    i386 is enabled first because umu's runtime bridge is 32-bit. weston and
#    xwayland provide the display the guests run against (start-display.sh);
#    Xvfb is not enough, it has no DRI3, so the present suite cannot put a
#    frame on the GPU-backed screen. Xvfb itself still has to be installed:
#    the GPU guest runner spawns xvfb-run inside its own bubblewrap sandbox
#    (tools/frosty_gpu_guest.py, tools/check_d3d12_regressions.py), which is
#    the one place the container's display is deliberately not used.
RUN dpkg --add-architecture i386 \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      bubblewrap \
      git \
      pkg-config \
      python3 \
      python3-pip \
      python3-venv \
      sudo \
      mesa-vulkan-drivers \
      libvulkan1 \
      libvulkan-dev \
      vulkan-tools \
      libegl1 \
      libgbm1 \
      libgl1 \
      libgles2 \
      libx11-6 \
      libxext6 \
      libxi6 \
      libxkbcommon0 \
      libxrandr2 \
      libwayland-client0 \
      weston \
      x11-utils \
      xauth \
      xvfb \
      xwayland \
 && rm -rf /var/lib/apt/lists/*

# 3. umu-launcher from its upstream release package. It pulls the 32-bit mesa
#    libraries and the AppArmor profile set as dependencies, which is why i386
#    had to be enabled above.
RUN curl -fsSL -o /tmp/umu.deb \
      "https://github.com/Open-Wine-Components/umu-launcher/releases/download/${UMU_VERSION}/${UMU_DEB}" \
 && apt-get update \
 && apt-get install -y --no-install-recommends /tmp/umu.deb \
 && rm -f /tmp/umu.deb \
 && rm -rf /var/lib/apt/lists/* \
 && umu-run --version

# 4. Microsoft's DXC system-wide: the launcher build and tools/shader-compiler
#    both look it up on PATH.
RUN curl -fsSL -o /tmp/dxc.tar.gz \
      "https://github.com/microsoft/DirectXShaderCompiler/releases/download/${DXC_VERSION}/${DXC_ARCHIVE}" \
 && mkdir -p /opt/dxc \
 && tar -xzf /tmp/dxc.tar.gz -C /opt/dxc --strip-components=1 \
 && install -m 0755 /opt/dxc/bin/dxc /usr/local/bin/dxc \
 && install -m 0755 /opt/dxc/bin/dxv /usr/local/bin/dxv \
 && install -m 0644 /opt/dxc/lib/libdxcompiler.so /usr/local/lib/libdxcompiler.so \
 && install -m 0644 /opt/dxc/lib/libdxil.so /usr/local/lib/libdxil.so \
 && rm -rf /tmp/dxc.tar.gz /opt/dxc \
 && ldconfig \
 && dxc --version \
 && command -v dxc dxv

# 5. Non-root user. Proton and umu refuse to run as root, and the devcontainer
#    CLI maps this uid onto the host user (Podman keep-id), which keeps the
#    bind-mounted workspace writable.
#
#    /tmp/.X11-unix is pre-created sticky because Xwayland runs unprivileged
#    and will not create the socket directory itself.
RUN userdel -r ubuntu 2>/dev/null || true \
 && groupdel ubuntu 2>/dev/null || true \
 && groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
 && groupadd -f render \
 && usermod -aG video,render ${USERNAME} \
 && mkdir -p /tmp/.X11-unix \
 && chmod 1777 /tmp/.X11-unix \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && sudo -u ${USERNAME} bash -lc 'id'
