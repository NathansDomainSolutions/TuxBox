#!/usr/bin/env python3
"""Rank the reasons xshader refuses ProjectScarlett's recorded shader requests.

Preparation and the runtime worker keep only a count of refused requests. This
tool runs the HLSL emitter alone (no DXC) on each recorded request and groups
the refusal messages. Program counters and register numbers are masked, so
identical gaps in different shaders share one row.

Example:
    python tools/triage_scarlett_shaders.py \\
        "$WINEPREFIX/drive_c/users/steamuser/AppData/Local/Temp/tuxbox_shader_cache/projectscarlett/<exe sha256>/requests" \\
        --json /tmp/scarlett-refusals.json

The cache is read only. The report holds xshader messages and request digests,
not game shader bytes.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_tuxbox.runtime_shaders import REQUEST_PATTERNS, XSHADER_UNSUPPORTED_HLSL

MASKS = (
    # "[buffer_load_dwordx3 @ 0x1a0]" names one variant of the same gap.
    (re.compile(r"\s*\[[a-z0-9_]+ @ 0x[0-9a-fA-F]+\]"), ""),
    (re.compile(r"0x[0-9a-fA-F]+"), "0x_"),
    (re.compile(r"\b([svt])\[?\d+(?::\d+)?\]?"), r"\1N"),
    (re.compile(r"\b\d+\b"), "N"),
)


def normalize(message):
    """One line per refusal, with program counters, registers and counts masked."""
    line = next((part.strip() for part in reversed(message.splitlines()) if part.strip()), "")
    for pattern, replacement in MASKS:
        line = pattern.sub(replacement, line)
    return line or "(no message)"


def construct(reason):
    """Coarse group: the instruction or construct named before ' at 0x'."""
    head, found, _ = reason.partition(" at 0x_")
    return head if found else reason.split(":", 1)[0]


def requests(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.suffix in REQUEST_PATTERNS and re.fullmatch(REQUEST_PATTERNS[path.suffix], path.stem)
    )


def command(xshader, source, work):
    """The emitter invocation runtime_shaders.compile_request uses for this request."""
    flags = []
    cooked = source
    if source.suffix == ".link":
        cooked = source / "producer.dxbc"
        flags = ["--link-pixel", source / "pixel.dxbc", "--pixel-order", source / "pixel.cso"]
    elif source.suffix == ".stencil":
        stem, sep, target = source.stem.rpartition("_rt")
        if not sep:
            raise ValueError(f"stencil request without a render-target suffix: {source.name}")
        flags = ["--stencil-target", target]
    return [xshader, cooked, *flags, "--hlsl-out", work / "shader.hlsl", "--profile-out", work / "profile"]


def probe(xshader, source, timeout):
    with TemporaryDirectory(prefix="tuxbox-triage-") as tmp:
        try:
            run = subprocess.run(command(xshader, source, Path(tmp)), capture_output=True, text=True,
                                 errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return source, "failed", f"timed out after {timeout:g}s"
        except (OSError, ValueError) as exc:
            return source, "failed", str(exc)
    if run.returncode == 0:
        return source, "translated", None
    status = "refused" if run.returncode == XSHADER_UNSUPPORTED_HLSL else "failed"
    return source, status, normalize(run.stderr or run.stdout)


def triage(xshader, sources, jobs, timeout, examples):
    totals = Counter()
    reasons = Counter()
    samples = defaultdict(list)
    failures = Counter()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for source, status, reason in pool.map(lambda s: probe(xshader, s, timeout), sources):
            totals[status] += 1
            if status == "refused":
                reasons[reason] += 1
                if len(samples[reason]) < examples:
                    samples[reason].append(source.name)
            elif status == "failed":
                failures[reason] += 1
    groups = Counter()
    for reason, count in reasons.items():
        groups[construct(reason)] += count
    return {
        "xshader": str(xshader),
        "totals": {"requests": len(sources), **{k: totals[k] for k in ("translated", "refused", "failed")}},
        "constructs": [{"construct": name, "count": count} for name, count in groups.most_common()],
        "reasons": [{"reason": reason, "count": count, "examples": samples[reason]}
                    for reason, count in reasons.most_common()],
        "failures": [{"reason": reason, "count": count} for reason, count in failures.most_common()],
    }


def markdown(report, top):
    refused = report["totals"]["refused"] or 1
    cell = lambda text: text.replace("|", "\\|")
    lines = [
        "| Requests | Translated | Refused | Failed |",
        "| ---: | ---: | ---: | ---: |",
        "| {requests:,} | {translated:,} | {refused:,} | {failed:,} |".format(**report["totals"]),
        "",
        "| Construct | Refused | Share |",
        "| --- | ---: | ---: |",
        *(f"| `{cell(row['construct'])}` | {row['count']:,} | {100 * row['count'] / refused:.1f}% |"
          for row in report["constructs"][:top]),
        "",
        "| Refusal | Requests | Share |",
        "| --- | ---: | ---: |",
        *(f"| `{cell(row['reason'])}` | {row['count']:,} | {100 * row['count'] / refused:.1f}% |"
          for row in report["reasons"][:top]),
    ]
    if report["failures"]:
        lines += ["", "Failures other than refusals (input or tool errors):", "",
                  *(f"- {row['count']:,} × `{row['reason']}`" for row in report["failures"][:top])]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("requests", type=Path, help="projectscarlett/<exe sha256>/requests in the Proton prefix")
    parser.add_argument("--xshader", type=Path, help="host xshader (default: build it like shader preparation)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=float, default=30, help="seconds per request, as at runtime")
    parser.add_argument("--top", type=int, default=25, help="rows per table")
    parser.add_argument("--examples", type=int, default=3, help="request names kept per refusal in --json")
    parser.add_argument("--limit", type=int, default=0, help="only the first N requests")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be 0 (all) or positive")
    if not args.requests.is_dir():
        parser.error(f"not a directory: {args.requests}")
    sources = requests(args.requests)[:args.limit or None]
    if not sources:
        parser.error(f"no recorded shader requests in {args.requests}")
    if args.xshader is None:
        from project_tuxbox.shader_prep import shader_compiler
        args.xshader = shader_compiler()

    report = triage(args.xshader, sources, max(1, args.jobs), args.timeout, args.examples)
    print(markdown(report, args.top))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
