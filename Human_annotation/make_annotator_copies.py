#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Duplicate a base CSV into A/B copies for annotators.")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--names", default="A,B", help="Annotator names, comma-separated")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    for n in names:
        dst = out_dir / f"annotator_{n}.csv"
        dst.write_text(Path(args.in_csv).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {dst}")

if __name__ == "__main__":
    main()
