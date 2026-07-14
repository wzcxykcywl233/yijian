"""Create a UTF-8 ZIP package for the raw dataset directories.

Windows tar/zip tools may store Chinese file names using a local code page.
Python's zipfile sets the UTF-8 filename flag for non-ASCII paths, which makes
the archive easier to extract on another Windows machine.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_DIRS = ["天回", "张家山", "武威", "马王堆"]


def iter_files(root: Path, dirs: list[str]):
    for dirname in dirs:
        base = root / dirname
        if not base.exists():
            raise FileNotFoundError(base)
        for path in base.rglob("*"):
            if path.is_file():
                yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package raw Yijian dataset directories into a UTF-8 ZIP.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root containing raw source directories.")
    parser.add_argument("--out", type=Path, required=True, help="Output .zip path.")
    parser.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS, help="Directory names to include.")
    parser.add_argument("--compresslevel", type=int, default=6)
    args = parser.parse_args()

    root = args.root.resolve()
    files = list(iter_files(root, args.dirs))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        args.out,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=args.compresslevel,
        allowZip64=True,
    ) as zf:
        for idx, path in enumerate(files, 1):
            arcname = path.relative_to(root).as_posix()
            zf.write(path, arcname)
            if idx % 1000 == 0 or idx == len(files):
                print(f"packed {idx}/{len(files)}")

    print(f"wrote {args.out}")
    print(f"files {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
