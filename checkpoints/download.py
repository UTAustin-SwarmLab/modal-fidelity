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
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1DNAJJFzNpxPYvmNJlU5QsiEJdkGnQalP"

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


def fetch(gdown, file_id: str, target: str, name: str, attempts: int = 5) -> None:
    """Download one file, retrying with a growing pause: Google Drive refuses large files for a
    while when they were downloaded many times recently."""
    import time
    for attempt in range(1, attempts + 1):
        print(f"downloading {name}" + (f" (attempt {attempt})" if attempt > 1 else ""), flush=True)
        try:
            partial = target + ".part"
            if gdown.download(id=file_id, output=partial, quiet=True):
                os.replace(partial, target)
                return
        except Exception as exc:                                        # noqa: BLE001
            print(f"  failed: {str(exc).splitlines()[0][:120]}", flush=True)
        if attempt < attempts:
            time.sleep(60 * attempt)
    sys.exit(f"could not download {name} after {attempts} attempts; wait a while and re-run "
             "(files already verified are kept), or fetch it from the Drive folder by hand: "
             f"{DRIVE_FOLDER_URL}")


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
        # anything absent or with the wrong checksum (e.g. an interrupted download) is fetched
        missing = [k for k, digest in files.items()
                   if not os.path.isfile(os.path.join(a.dest, k))
                   or sha256(os.path.join(a.dest, k)) != digest]
        if missing:
            if not DRIVE_FOLDER_URL:
                sys.exit("DRIVE_FOLDER_URL is not set in checkpoints/download.py yet")
            try:
                import gdown
            except ImportError:
                sys.exit("pip install gdown (it is in environment.yml)")
            listing = {f.path: f.id for f in
                       gdown.download_folder(DRIVE_FOLDER_URL, skip_download=True, quiet=True)}
            for name in missing:
                if name not in listing:
                    sys.exit(f"{name} is not in the Drive folder")
                target = os.path.join(a.dest, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                fetch(gdown, listing[name], target, name)
    bad = verify(a.dest, files)
    print(f"\n{len(files) - len(bad)}/{len(files)} files verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
