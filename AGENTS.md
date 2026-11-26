# Instructions for AI coding agents

TuxBox is a compatibility layer that runs Xbox Series X|S games on Linux and Windows. Read [CONTRIBUTING](CONTRIBUTING.md) before changing anything; its rules apply to you. [Source layout](docs/SOURCE_LAYOUT.md) says where code goes, and [Where to improve](docs/FOCUS.md) lists open areas.

## Ask the person first

- Commit, push, open or merge a pull request, or post a comment only when they ask.
- Ask before deleting files, changing their system settings, or replacing a Wine prefix, cache or save.
- A game launch can take many minutes and take over the screen. Say so before starting one.

## Safety and privacy

- Run ProjectScarlett only through `tuxbox.py run`, which keeps it offline. Its crash handler uploads crash reports otherwise.
- Never commit retail game files, dumps, captures, shader bytecode from a title, logs or Wine prefixes.
- Keep usernames, home paths and other personal details out of commits, issues, comments and evidence.
- Don't copy code from projects under licences that don't fit [Apache 2.0](LICENSE), such as the GPL. Reimplement an idea from your own understanding and credit it.

## Finding a bug

- Find the cause before changing code. Don't skip, hide or work around a fault unless the person agrees.
- Say what you measured and what you are guessing. A theory is not a finding until a test could have disproved it and didn't.
- Compare against a build that worked, under the same conditions: same caches, settings and input. Check the host too (audio, drivers, disk, memory).
- Keep temporary tracing out of the commit that fixes the bug.

## Writing code

- Reproduce the behaviour in a test or a [Lanes self-test](examples/lanes/README.md#self-test) first. The test should fail without your change.
- Choose paths by what the driver can do, not by GPU vendor.
- A patch to a title's own code applies only when the bytes match the exact build, and the comment explains the fault it fixes.
- Rust files stay under 1,200 lines, and tests go in the crate's `src/tests/`.
- Match the code around you. Don't reformat or refactor what your change doesn't need.

## Checking your work

```sh
python tuxbox.py check --no-gpu   # while you work
python tuxbox.py check            # before a pull request, on a real GPU
```

- Report failures as they are, with the output. Never describe a skipped or failed check as passing.
- Shared code needs a run on an AMD card and an NVIDIA card. Say which one is still missing.
- Compiling is not proof that translated shaders are correct. Compare their output with a reference on the GPU.
- `docs/RESULTS.md` and `docs/progress.svg` are generated from `evidence/`. Don't edit them by hand, and don't write evidence for something that wasn't run.
