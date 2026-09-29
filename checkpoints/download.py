"""Download the released checkpoints and preview features from Google Drive and verify them.

    python checkpoints/download.py                # download into checkpoints/ and verify
    python checkpoints/download.py --verify-only  # only check files already in place

Every file is checked against checkpoints/SHA256SUMS; a mismatch is reported and the script
exits non-zero. Files already present with the right checksum are not downloaded again.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

#: the public Google Drive folder that holds the release (router/, preview/, features/)
DRIVE_FOLDER_URL = ""

HERE = os.path.dirname(os.path.abspath(__file__))


def expected(sums_file: str) -> dict:
    out = {}
    with open(sums_file) as handle:
        for line in handle:
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: str, files: dict) -> list:
    """Names of files that are missing or whose checksum differs."""
    bad = []
    for name, digest in sorted(files.items()):
        path = os.path.join(root, name)
        ok = os.path.isfile(path) and sha256(path) == digest
        print(f"{'ok  ' if ok else 'BAD '} {name}")
        if not ok:
            bad.append(name)
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=os.environ.get("MODALFIDELITY_DATA", HERE),
                   help="where to put the files (default: $MODALFIDELITY_DATA, else checkpoints/)")
    p.add_argument("--verify-only", action="store_true", help="check files in --dest; download nothing")
    p.add_argument("--skip-features", action="store_true",
                   help="leave out the 5.4 GB preview-feature cache (it is needed only to "
                        "evaluate or train the router, not to reproduce the figures)")
    a = p.parse_args(argv)

    files = expected(os.path.join(HERE, "SHA256SUMS"))
    if a.skip_features:
        files = {k: v for k, v in files.items() if not k.startswith("features/")}
    if not a.verify_only:
        missing = [k for k in files if not os.path.isfile(os.path.join(a.dest, k))]
        if missing:
            if not DRIVE_FOLDER_URL:
                sys.exit("DRIVE_FOLDER_URL is not set in checkpoints/download.py yet")
            try:
                import gdown
            except ImportError:
                sys.exit("pip install gdown (it is in environment.yml)")
            gdown.download_folder(DRIVE_FOLDER_URL, output=a.dest, quiet=False,
                                  remaining_ok=True)
    bad = verify(a.dest, files)
    print(f"\n{len(files) - len(bad)}/{len(files)} files verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
