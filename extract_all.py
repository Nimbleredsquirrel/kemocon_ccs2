"""
Run once to unpack all tarballs into DATA_ROOT.
Usage:  python extract_all.py
"""
import tarfile
from pathlib import Path
from config import DOWNLOADS, DATA_ROOT

ARCHIVES = [
    ("e4_data.tar",                   None),
    ("neurosky_polar_data.tar",       None),
    ("emotion_annotations.tar.gz",    None),
    ("metadata.tar",                  None),
    ("data_quality_tables.tar.gz",    None),
    ("debate_audios.tar.gz",          None),   # required for audio features
    ("debate_recordings.tar.gz",      None),   # required for video features (3 videos)
]

DATA_ROOT.mkdir(parents=True, exist_ok=True)

for fname, _ in ARCHIVES:
    src = DOWNLOADS / fname
    if not src.exists():
        print(f"[SKIP] {fname} not found")
        continue
    print(f"Extracting {fname} …")
    try:
        with tarfile.open(src) as tf:
            tf.extractall(DATA_ROOT)
        print(f"  done → {DATA_ROOT}")
    except Exception as e:
        print(f"  [WARN] Partial extraction ({e}) — some files may still have been extracted")

print("\nAll archives extracted.")
