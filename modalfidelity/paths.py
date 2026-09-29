"""Where data, checkpoints and caches live.

Everything resolves from two roots, each set by an environment variable or by a repo-root
``machine.env`` file (``KEY=value`` lines, gitignored), in that order:

* ``MODALFIDELITY_DATA``: downloaded checkpoints and the preview-feature cache.
  Defaults to ``<repo>/checkpoints``.
* ``MODALFIDELITY_DATASET``: the AV-Deepfake1M videos. Needed only to re-cache preview
  features; there is no default.

The small saved results that regenerate the paper's figures ship with the repo in ``data/``.
"""
from __future__ import annotations

import os
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_FILENAME = "machine.env"


def _from_config(key: str) -> Optional[str]:
    path = os.path.join(REPO_ROOT, CONFIG_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        for line in handle:
            name, sep, value = line.strip().partition("=")
            if sep and not name.startswith("#") and name.strip() == key:
                return os.path.expanduser(value.strip().strip("'\""))
    return None


def _resolve(key: str, fallback: Optional[str]) -> Optional[str]:
    value = os.environ.get(key) or _from_config(key) or fallback
    return os.path.expanduser(value) if value else None


def saved_data_dir() -> str:
    """The saved results shipped with the repo (``data/``)."""
    return os.path.join(REPO_ROOT, "data")


def data_root() -> str:
    """Downloaded checkpoints and caches."""
    return _resolve("MODALFIDELITY_DATA", os.path.join(REPO_ROOT, "checkpoints"))


def dataset_root() -> str:
    """AV-Deepfake1M videos. Raises if unset, because nothing sensible can be guessed."""
    root = _resolve("MODALFIDELITY_DATASET", None)
    if not root:
        raise RuntimeError("set MODALFIDELITY_DATASET (or add it to machine.env) to the "
                           "AV-Deepfake1M root; it is only needed to re-cache preview features")
    return root


def checkpoint(*parts: str) -> str:
    return os.path.join(data_root(), *parts)
