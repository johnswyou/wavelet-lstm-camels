#!/usr/bin/env python3
"""Compare two text files line‑by‑line and as sets.

Usage (default filenames):
    python diff_files.py

Usage (custom filenames):
    python diff_files.py --correct path/to/correct_output.txt --output path/to/output.txt

Each input file is expected to contain **one code per line**. Blank lines are ignored.
The script reports:
    • Codes in *correct* that are **missing** from *output* ("missing in output").
    • Codes in *output* that are **not** in *correct* ("extra in output").
    • Optional line‑by‑line mismatches (off by default).

Exit status is non‑zero when any difference is found, which makes the
script handy in CI pipelines or grading automation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set, Tuple


def read_codes(path: Path) -> List[str]:
    """Return *non‑empty* stripped lines from *path*."""
    try:
        with path.open("r", encoding="utf‑8") as fh:
            return [line.strip() for line in fh if line.strip()]
    except FileNotFoundError:
        sys.exit(f"❌ File not found: {path}")


def positional_mismatches(ref: List[str], other: List[str]) -> List[Tuple[int, str, str]]:
    """Return a list of (1‑based index, ref_line, other_line) tuples where lines differ."""
    mismatches: List[Tuple[int, str, str]] = []
    min_len = min(len(ref), len(other))
    for idx in range(min_len):
        if ref[idx] != other[idx]:
            mismatches.append((idx + 1, ref[idx], other[idx]))

    if len(ref) > len(other):
        for idx in range(min_len, len(ref)):
            mismatches.append((idx + 1, ref[idx], "<missing>"))
    elif len(other) > len(ref):
        for idx in range(min_len, len(other)):
            mismatches.append((idx + 1, "<missing>", other[idx]))
    return mismatches


def compare(correct_path: Path, output_path: Path, show_line_diff: bool = False) -> int:
    """Compare *correct_path* and *output_path*.

    Returns 0 when files match exactly, otherwise 1.
    """
    correct_lines: List[str] = read_codes(correct_path)
    output_lines: List[str] = read_codes(output_path)

    correct_set: Set[str] = set(correct_lines)
    output_set: Set[str] = set(output_lines)

    missing_in_output = correct_set - output_set
    extra_in_output = output_set - correct_set

    any_diff = bool(missing_in_output or extra_in_output)

    if not any_diff and not show_line_diff:
        print("✅ Files match exactly – no differences found.")
        return 0

    # --- Summary ---
    print("# Comparison summary")
    print(f"Total codes in correct : {len(correct_set)}")
    print(f"Total codes in output  : {len(output_set)}\n")

    # Missing codes
    print("## Missing in output (expected but absent)")
    if missing_in_output:
        for code in sorted(missing_in_output):
            print(code)
    else:
        print("(none)")
    print()

    # Extra codes
    print("## Extra in output (unexpected)")
    if extra_in_output:
        for code in sorted(extra_in_output):
            print(code)
    else:
        print("(none)")
    print()

    # Optional positional differences
    if show_line_diff:
        print("## Positional mismatches (line‑by‑line)")
        pos_diff = positional_mismatches(correct_lines, output_lines)
        if pos_diff:
            for idx, corr, out in pos_diff:
                print(f"Line {idx:>6}: expected {corr!r} | got {out!r}")
        else:
            print("(none) – order matches for the overlapping portion")
        print()

    return 1 if any_diff else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two text files containing codes line‑by‑line as well as set differences.")
    parser.add_argument("--correct", default="correct_output.txt", type=Path, help="Reference file (default: correct_output.txt)")
    parser.add_argument("--output", default="output.txt", type=Path, help="File to check (default: output.txt)")
    parser.add_argument("--positional", action="store_true", help="Also show line‑by‑line mismatches and differing length issues.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = compare(args.correct, args.output, show_line_diff=args.positional)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
