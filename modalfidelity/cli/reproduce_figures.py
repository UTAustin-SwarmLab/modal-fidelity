"""Regenerate the paper's Fig. 2, Fig. 3 and Table 1 from the saved results in data/, and check
every number the paper states about them. No randomness, no GPU, no downloads."""
from __future__ import annotations

import argparse
import os
import sys

from ..paths import REPO_ROOT, saved_data_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=saved_data_dir(),
                   help="directory holding the saved results (default: the repo's data/)")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "figures"),
                   help="where to write fig2.png/.pdf and fig3.png/.pdf "
                        "(default: outputs/figures)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace figures already in --out (otherwise refuse)")
    a = p.parse_args(argv)

    from ..plots import claims, fig2, fig3, table1
    for name, mod in (("fig2", fig2), ("fig3", fig3)):
        for path in mod.plot(a.data, os.path.join(a.out, f"{name}.png"), a.overwrite):
            print("wrote", path)
    print()
    print(table1.as_markdown(table1.load(a.data)))
    print()
    failed = 0
    for claim, text, value, ok in claims.check(a.data):
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {claim:52s} paper: {text:48s} data: {value:.4f}")
    print(f"\n{len(claims.PAPER) - failed}/{len(claims.PAPER)} paper numbers reproduced")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
