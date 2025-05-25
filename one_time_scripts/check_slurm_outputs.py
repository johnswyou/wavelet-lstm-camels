#!/usr/bin/env python3
"""
Scan slurm-28743507_*.out files, find ones that did **not** finish cleanly,
grab the corresponding lines from csv_filenames_v3.txt, and write them to
csv_filenames_v5.txt.
"""

from pathlib import Path
import re
import sys

# ---- CONFIG (edit if your file names ever change) --------------------------
OUT_PATTERN   = re.compile(r"^slurm-28743507_(\d+)\.out$")   # captures the integer
GOOD_TRAILER  = "Program finished with exit code 0"
CSV_IN        = Path("csv_filenames_v4.txt")
CSV_OUT       = Path("csv_filenames_v5.txt")
# ----------------------------------------------------------------------------

def last_line(path: Path) -> str:
    """Return the last non‑empty line of a text file (stripped)."""
    with path.open("rb") as f:                    # read as bytes for speed
        f.seek(0, 2)                              # jump to file end
        block, size = b"", f.tell()
        step = 1024
        while size >= 0:
            f.seek(max(size - step, 0))
            block = f.read(step) + block
            if block.count(b"\n") > 1 or size == 0:
                break
            size -= step
    *_, tail = block.splitlines()
    return tail.decode(errors="replace").strip()

def gather_failed_indices() -> list[int]:
    """Return the integer suffixes whose .out file did **not** finish with code 0."""
    failed = []
    for path in Path.cwd().iterdir():
        m = OUT_PATTERN.match(path.name)
        if not m:
            continue
        index = int(m.group(1))
        try:
            if GOOD_TRAILER not in last_line(path):
                failed.append(index)
        except Exception as e:   # corrupted / empty file?
            print(f"⚠️  Could not read {path}: {e}", file=sys.stderr)
            failed.append(index)
    return sorted(failed)

def write_failed_csv_rows(failed_idxs: list[int]) -> None:
    """Write the lines (1‑based) matching *failed_idxs* from CSV_IN into CSV_OUT."""
    if not failed_idxs:
        CSV_OUT.write_text("")          # create/empty the file for consistency
        return
    with CSV_IN.open("r", encoding="utf‑8") as src:
        lines = src.readlines()

    selected = []
    for i in failed_idxs:
        if 1 <= i <= len(lines):
            selected.append(lines[i - 1])
        else:
            print(f"⚠️  Requested line {i} not present in {CSV_IN}", file=sys.stderr)

    with CSV_OUT.open("w", encoding="utf‑8") as dest:
        dest.writelines(selected)

def main() -> None:
    failed = gather_failed_indices()
    write_failed_csv_rows(failed)

    total   = sum(1 for p in Path.cwd().iterdir() if OUT_PATTERN.match(p.name))
    print(f"✔️  Scanned {total} slurm output file(s).")
    print(f"❌  {len(failed)} file(s) lacked a clean exit (code 0).")
    if failed:
        print("   Offending job array indices:", ", ".join(map(str, failed)))
        print(f"→  Wrote {len(failed)} filename(s) to {CSV_OUT}")
    else:
        print("🎉  All jobs finished with exit code 0.")

if __name__ == "__main__":
    main()
