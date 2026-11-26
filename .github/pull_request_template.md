## Summary

## Verified on Lanes

A pull request is not merged until Lanes passes on a GPU. Run:

```sh
python tuxbox.py check
```

- [ ] `python tuxbox.py check` passed on this branch
- [ ] Changed or new behaviour is covered by a Lanes scenario, or the reason it cannot be is stated below

GPU, driver and backend:

- [ ] This changes code both vendors run, and the check passed on an AMD and an NVIDIA card ([why](../CONTRIBUTING.md#amd-and-nvidia)). Otherwise, say which vendor still needs to run it:

Paste the summary printed at the end:

```text
```

## Checks

CI runs the Python unit tests, the shader compiler, DXBC, and XG layout tests, and a Windows typecheck of `tuxbox-d3d12`.

If this pull request changes evidence JSON, regenerate the results table and the progress map and include both files:

```sh
python tuxbox.py status > docs/RESULTS.md
python tuxbox.py status --svg docs/progress.svg
```
