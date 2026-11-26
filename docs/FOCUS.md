# Where to improve

Open areas that need work, with what is known and a useful next step. [Progress](PROGRESS.md) records what works today. If you start on one, say so in an issue so effort isn't doubled. Measurements below are from ProjectScarlett 1.0.2.1 on a Ryzen with 16 threads, 32 GB of memory and an RTX 4070 Ti SUPER, unless noted.

## Startup and shader compilation

A warm launch reached player selection in 69 s. Cold launches take many minutes, and it is too slow.

- **Measure each stage.** A shader goes from console code to HLSL, DXC (DXIL), dxil-spirv in vkd3d-proton, then the driver. We don't have time and memory per stage for a cold and a warm start. Start there.
- **Benchmark a shorter route.** On Linux, translating straight to SPIR-V would drop DXC and dxil-spirv. Windows would keep DXIL. Compare both against the per-stage baseline before building either.
- **Reduce memory use.** With no limit on large compiles, the game filled 16 GB of swap. The limit of four should be chosen from available memory.
- **Large compute shaders.** Single driver compiles took up to 158 s. Smaller or better structured translated code would help every stage.

## Pipeline and driver caches

- The NVIDIA driver stopped updating its cache index once one file grew past about 5 GB, so every launch recompiled the same pipelines. TuxBox now limits the cache to 4 GiB. Check that long sessions stay warm, and find the real limit.
- A new translator version retranslates shaders on the next launch. Measure how much of that cost can be avoided.

## Audio start-up

ProjectScarlett's presenter shares a lock with its audio output. When audio start-up was slow, the game's own code left that lock held and froze presentation; TuxBox now patches that. The slow start came from a failing USB audio device on the host: while it was enabled, streams timed out and piled up.

- Find out why one failing device stalls streams on another output, and whether Wine can recover or report it.
- Check whether the black screen before player selection is the splash screen that should be drawn there.

## Frame pacing

At high frame rates ProjectScarlett shows pairs of frame time spikes between otherwise even frames. Capture them with `--mangohud` and find what runs at those moments.

## Shader translation

- Many pipelines still get empty stand-ins because a stage can't be translated. Rank them with the [pipeline failure record](PROJECTSCARLETT.md#finding-why-pipelines-fail).
- dxil-spirv reports `Failed to remap` errors for some translated shaders. Find which bindings cause them.
- The two-lanes-per-thread translation for 64-lane programs passes synthetic tests. Its output hasn't been verified in a game.

## AMD and NVIDIA

- RADV substitutes other memory for imported game pages, and TuxBox copies them instead, which is slow and loses GPU writes ([#191](https://github.com/brcly/TuxBox/issues/191)).
- Shared code needs a run on both vendors. See [CONTRIBUTING](../CONTRIBUTING.md#amd-and-nvidia).

## GPU synchronisation

GPU queues stall on fence label waits at ProjectScarlett's story intro, and the title's watchdog ends the run ([#215](https://github.com/brcly/TuxBox/issues/215)).
