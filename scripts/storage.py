"""Shared storage checks for command-line tools."""
import os
from pathlib import Path


def work_dir():
    root = Path(os.environ["DEID_STORAGE_ROOT"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("storage root must be an existing absolute directory")
    mode = os.environ.get("DEID_REQUIRE_EXTERNAL", "1")
    if mode not in {"0", "1"}:
        raise ValueError("DEID_REQUIRE_EXTERNAL must be 0 or 1")
    if mode == "1" and (not root.is_mount() or root.stat().st_dev == Path('/').stat().st_dev):
        raise ValueError("configured external storage volume is not mounted")
    expected = (root / "medical-deid-rs").resolve()
    if not expected.is_relative_to(root.resolve()):
        raise ValueError("workspace symlink escapes configured storage")
    supplied = Path(os.environ.get("DEID_WORK", str(expected)))
    if supplied.resolve() != expected:
        raise ValueError("DEID_WORK must be the medical-deid-rs directory on configured storage")
    return expected


def require_output(path):
    if not path.is_absolute() or ".." in path.parts or not path.resolve().is_relative_to(work_dir()):
        raise ValueError("output must be an absolute path inside configured DEID_WORK")


if __name__ == "__main__":
    work_dir()
