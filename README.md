# TuxBox

TuxBox is an experimental Xbox Series X|S compatibility layer for x86-64 Windows and Linux.

- **Linux** uses Valve’s stock Proton Experimental. NVIDIA support is verified on a GeForce RTX 4070 Ti SUPER, including ProjectScarlett’s player selection screen and Frosty’s mesh shader tests. [GPU and host test results](docs/RESULTS.md)
- **Windows** has a native Direct3D 12 backend. Windows game compatibility is still in development. [Windows build instructions](examples/lanes/README.md#build-requirements)

## Why so few Xbox libraries?

Many calls an Xbox Series game makes already work through standard Windows APIs. Windows handles those calls natively, and Proton provides the same APIs on Linux. Games can also use compatible DLLs they ship, such as video decoders. TuxBox only needs to translate the Xbox-specific behavior a game actually uses, such as graphics commands and texture layouts. That’s why we track working game features instead of counting every function in every Xbox DLL. [Xbox library coverage](docs/LIBRARIES.md)

## What runs today?

- **Lanes** is a small game that uses TuxBox’s Xbox graphics, memory and input libraries. Collect coins and dodge barriers using a keyboard, mouse or Xbox controller.
- **The Matrix Awakens (Frosty)** reaches the Wake Up chase cinematic on Linux with TuxBox XG and reduced graphics settings. Earlier reduced-settings runs reached the highway chase and open city. With the original Xbox settings it loads the Construct map and presents a black frame before stopping while creating a compute shader during scene setup. You supply the game, its matching Xbox system files and Proton. [Setup and test results](docs/MATRIX_AWAKENS.md)

**GTA VI (ProjectScarlett)** shows the player selection prompt at 3840×2160 on stock Proton Experimental. [Title progress](docs/PROGRESS.md) · [Setup guide](docs/PROJECTSCARLETT.md)

### Project Scarlett graphics trials

On Linux, put `--graphics-menu` before the game path to choose the packaged settings or experimental Series S mode. Use `--graphics-profile series-s` to select it from a script. Series S mode uses Bubblewrap and a temporary copy of the Wine prefix. The copy is discarded at exit.

The Series S trial requested a 1920×1080 render resource and stopped on a texture allocation before presenting. Use the packaged settings for the current playable startup path.

## Shader preparation

On Linux, the launcher prepares shaders before opening Frosty and ProjectScarlett. Frosty uses a profile of its game files. ProjectScarlett records the shaders the game requests and prepares them before subsequent launches. New shaders can still compile as the game encounters them.

The launcher shows compilation progress and reuses the cache on later runs. Pass `--prepare-only` to prepare the cache without opening the game, or `--no-shader-prep` to skip preparation for one launch. Place either option before the game path.

## Try Lanes

Once the build tools are installed, run these commands from the project folder. [Build requirements](examples/lanes/README.md#build-requirements)

```sh
python -m pip install -r requirements.txt
python build.py lanes
python tuxbox.py run --app-id Lanes build/lanes
```

Press Left or Right to change lanes, or Space to take a turn in your current lane. R restarts the game and Escape quits.

## Game logs

`tuxbox.py run` shows game output in the terminal as it arrives. The first line gives the path to a saved copy. On Linux, logs go in `~/.local/state/project-tuxbox/logs/` (or under `XDG_STATE_HOME`). On Windows, they go in `%LOCALAPPDATA%\TuxBox\logs\`.

To choose a file, pass `--log-file` before the game path. Each run appends to that file.

```sh
python tuxbox.py run --log-file build/lanes.log --app-id Lanes build/lanes
```

On Linux, add `--mangohud` before the game path to see FPS, frame times and hardware load while playing. TuxBox saves a CSV of presented frames and a `summary.json` with average FPS and median, p95 and p99 frame times in a capture folder beside the game log. The Frosty launcher enables this automatically when MangoHud is installed.

On Linux, add `--offline` before the game path to launch without network access. The game runs in a new network namespace, so its crash handler cannot upload reports. ProjectScarlett's handler sends minidumps to an online crash service unless it is blocked. Offline launches need unprivileged user namespaces.

```sh
python tuxbox.py run --offline /path/to/game
```

Run `python tuxbox.py inspect /path/to/game` to see the executable TuxBox will launch and what it found in the package’s `.bin` files. For Frosty, its shipping symbols can also turn addresses from the log into function names. [Symbol lookup example](docs/MATRIX_AWAKENS.md#what-carries-over-to-projectscarlett)

## Compatibility

[![Compatibility and shader progress for Frosty and Project Scarlett](docs/progress.svg)](docs/progress.svg)

[Current progress](docs/PROGRESS.md) · [Xbox libraries](docs/LIBRARIES.md) · [Shader coverage](docs/SHADERS.md) · [Test results](docs/RESULTS.md) · [Architecture](docs/ARCHITECTURE.md) · [Credits and references](docs/REFERENCES.md)

## License

Copyright © 2026 TuxBox contributors. TuxBox is licensed under the Apache License 2.0. Third-party materials retain their respective terms. [License](LICENSE) · [Third-party notices](docs/REFERENCES.md#shader-compiler-attribution)
