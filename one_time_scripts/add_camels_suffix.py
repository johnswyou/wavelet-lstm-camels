#!/usr/bin/env python3
"""
add_camels_suffix.py

Read a text file containing one code per line (e.g. 14306340) and
write out a new file where each line has “_camels.csv” appended
(e.g. 14306340_camels.csv).

Usage
-----
# create a new file:
python add_camels_suffix.py codes.txt codes_camels.txt

# overwrite the original in‑place:
python add_camels_suffix.py codes.txt
"""
from pathlib import Path
import argparse
import sys

def add_suffix(lines):
    """Yield each line stripped of its newline + '_camels.csv\\n'."""
    for line in lines:
        code = line.rstrip("\r\n")
        if code:                         # skip blank lines
            yield f"{code}_camels.csv\n"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append '_camels.csv' to every code line in a text file."
    )
    parser.add_argument("input",  help="Path to the input text file.")
    parser.add_argument(
        "output", nargs="?",
        help="Output path. Omit to overwrite the input file in‑place."
    )
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    try:
        lines = in_path.read_text(encoding="utf‑8").splitlines()
        out_path.write_text("".join(add_suffix(lines)), encoding="utf‑8")
    except FileNotFoundError:
        sys.exit(f"Input file not found: {in_path}")
    except Exception as exc:
        sys.exit(f"Error processing file: {exc}")

if __name__ == "__main__":
    main()
