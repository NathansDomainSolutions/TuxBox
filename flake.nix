{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default];
      };

      rust = pkgs.rust-bin.stable."1.95.0".default.override {
        extensions = ["rust-src" "rust-analyzer"];
        targets = ["x86_64-unknown-linux-gnu" "x86_64-pc-windows-msvc"];
      };

      # The launcher's default Proton (PINNED_PROTON in project_tuxbox/host.py).
      # nixpkgs' proton-ge-bin follows its own release schedule, so pin the
      # upstream release archive here and keep both versions in sync.
      proton-ge = pkgs.fetchzip {
        name = "GE-Proton11-14";
        url = "https://github.com/GloriousEggroll/proton-ge-custom/releases/download/GE-Proton11-14/GE-Proton11-14-x86_64.tar.gz";
        hash = "sha256-Bru3l0AqrvDl+XYC1dYhmy5XqKP49yFjWVJwZzel60Q=";
      };

      tools = with pkgs; [
        # Languages
        rust
        (python312.withPackages (ps: [ps.pefile]))

        # Windows PE toolchain (clang-cl is only in the unwrapped package)
        llvmPackages.clang-unwrapped
        llvmPackages.lld
        llvmPackages.llvm
        cargo-xwin
        directx-shader-compiler

        # Launcher and GPU checks
        umu-launcher
        bubblewrap
        xvfb-run
        xwayland
        mangohud
        zenity
        vulkan-tools
      ];

      # The isolated GPU checks start Proton's Wine directly, without umu's runtime
      fhs = pkgs.buildFHSEnv {
        name = "tuxbox-fhs";
        targetPkgs = _: tools ++ [pkgs.stdenv.cc];
        # Proton's 64-bit and 32-bit Wine
        multiArch = true;
        multiPkgs = p:
          with p; [
            alsa-lib
            fontconfig
            freetype
            gnutls
            libGL
            libpulseaudio
            libx11
            libxcb
            libxcomposite
            libxcursor
            libxext
            libxfixes
            libxi
            libxinerama
            libxkbcommon
            libxrandr
            libxrender
            libxxf86vm
            lz4
            systemd
            vulkan-loader
            wayland
            zlib
          ];
        profile = ''
          export VULKAN_HEADERS_INCLUDE=${pkgs.vulkan-headers}/include
          # The checks mount an empty /tmp, so nix develop's TMPDIR would not exist
          export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
          # 32-bit preloads such as Steam's extest fail in 64-bit processes
          unset LD_PRELOAD
        '';
        runScript = "env";
      };
    in {
      packages.proton-ge = proton-ge;

      devShells.default = pkgs.mkShell {
        packages = tools ++ [fhs];
        buildInputs = [pkgs.lz4];

        # Python loads these through ctypes
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.lz4 pkgs.libx11];
        VULKAN_HEADERS_INCLUDE = "${pkgs.vulkan-headers}/include";

        PROTONPATH = proton-ge;
      };
    });
}
