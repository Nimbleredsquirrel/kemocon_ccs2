#!/usr/bin/env python3
"""
Set up data/ symlinks for K-EmoCon project.

The K-EmoCon dataset can be downloaded from:
    https://zenodo.org/record/3931963

Usage:
    python3 setup_data.py --kemocon-root /path/to/kemocon_extracted
"""
import argparse
import sys
from pathlib import Path

RAW_DIRS = [
    "e4_data",
    "neurosky_polar_data",
    "emotion_annotations",
    "metadata",
    "debate_audios",
    "debate_recordings",
]


def setup(kemocon_root: Path) -> bool:
    if not kemocon_root.exists():
        print(f"ERROR: Path does not exist: {kemocon_root}")
        return False

    project_data = Path(__file__).parent / "data"
    project_data.mkdir(exist_ok=True)

    ok = True
    for dirname in RAW_DIRS:
        src = kemocon_root / dirname
        dst = project_data / dirname

        if not src.exists():
            print(f"  [WARN] Not found in K-EmoCon root: {src}")
            ok = False
            continue

        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            print(f"  [SKIP] Already exists (not a symlink): {dst}")
            continue

        dst.symlink_to(src.resolve())
        print(f"  Linked: data/{dirname} → {src}")

    if ok:
        print("\nSetup complete. Verify with:")
        print("  python3 main.py --physio-only --no-optuna")
    else:
        print("\nSetup completed with warnings. Some directories were not found.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up K-EmoCon data symlinks")
    parser.add_argument(
        "--kemocon-root", required=True,
        help="Path to the extracted K-EmoCon directory "
             "(contains e4_data/, emotion_annotations/, etc.)")
    args = parser.parse_args()
    success = setup(Path(args.kemocon_root))
    sys.exit(0 if success else 1)
