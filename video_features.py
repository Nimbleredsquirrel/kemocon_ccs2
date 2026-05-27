from typing import Optional
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_ROOT, SEGMENT_SEC

VIDEO_DIR = DATA_ROOT / "debate_recordings"
CACHE_DIR = DATA_ROOT / "video_features_cache"
FRAMES_DIR = DATA_ROOT / "video_features"

_VIDEO_MAP = {
    2:  "p2_854.mp4",
    3:  "p3_688.mp4",
    4:  "p4_688.mp4",
    5:  "p5_629.mp4",
    7:  "p7_613.mp4",
    8:  "p8_613.mp4",
    9:  "p9_616.mp4",
    10: "p10_616.mp4",
    13: "p13_611.mp4",
    15: "p15_615.mp4",
    19: "p19_621.mp4",
    20: "p20_621.mp4",
    21: "p21_606.mp4",
    22: "p22_606.mp4",
    23: "p23_608.mp4",
    24: "p24_608.mp4",
    25: "p25_623.mp4",
    26: "p26_623.mp4",
    29: "p29_642.mp4",
    30: "p30_642.mp4",
    31: "p31_658.mp4",
}

_AU_COLS = ["AU04", "AU06", "AU12", "AU17"]
_POSE_COLS = ["Pitch", "Yaw"]

_FEAT_NAMES = (
    [f"vid_au{int(c[2:])}" for c in _AU_COLS] +
    ["vid_pitch", "vid_yaw"]
)
_NAN_ROW = {f"{n}_mean": np.nan for n in _FEAT_NAMES} | \
           {f"{n}_std":  np.nan for n in _FEAT_NAMES}


def _video_path(pid: int) -> Optional[Path]:
    fname = _VIDEO_MAP.get(pid)
    if fname is None:
        return None
    p = VIDEO_DIR / fname
    return p if p.exists() else None


def _get_detector():
    try:
        from feat import Detector
    except ImportError:
        raise ImportError("py-feat not installed. Run: pip3 install py-feat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = Detector(au_model="svm")
    return det


def _aggregate_frames(frames: pd.DataFrame, n_segs: int,
                      offset: float = 0.0) -> pd.DataFrame:
    target_cols = _AU_COLS + _POSE_COLS
    out_names = [f"vid_au{int(c[2:])}" for c in _AU_COLS] + ["vid_pitch", "vid_yaw"]
    rows = []
    for seg_idx in range(n_segs):
        annotation_t = (seg_idx + 1) * SEGMENT_SEC
        t_end = annotation_t - offset
        t_start = t_end - SEGMENT_SEC
        mask = (frames["timestamp_sec"] > t_start) & (frames["timestamp_sec"] <= t_end)
        seg = frames[mask]
        feat_row = {"seconds": annotation_t}
        for col, out in zip(target_cols, out_names):
            if col in seg.columns and len(seg):
                vals = pd.to_numeric(seg[col], errors="coerce").dropna()
                feat_row[f"{out}_mean"] = float(vals.mean()) if len(vals) else np.nan
                feat_row[f"{out}_std"] = float(vals.std()) if len(vals) > 1 else np.nan
            else:
                feat_row[f"{out}_mean"] = np.nan
                feat_row[f"{out}_std"] = np.nan
        rows.append(feat_row)
    return pd.DataFrame(rows)


def extract_video_features(pid: int, n_segs: int) -> Optional[pd.DataFrame]:
    vpath = _video_path(pid)
    if vpath is None:
        return None

    try:
        import cv2
        detector = _get_detector()
    except Exception as e:
        print(f"  [WARN] py-feat/cv2 unavailable: {e}")
        return None

    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    seg_frames = int(round(fps * SEGMENT_SEC))
    skip = max(1, int(fps * SEGMENT_SEC / 3))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fex = detector.detect_video(str(vpath), skip_frames=skip)
    except Exception as e:
        print(f"  [WARN] P{pid} py-feat detection failed: {e}")
        return None

    if fex is None or len(fex) == 0:
        return None

    if "frame" not in fex.columns:
        fex = fex.reset_index()
        if "frame" not in fex.columns:
            fex["frame"] = np.arange(len(fex)) * skip

    target_cols = _AU_COLS + _POSE_COLS
    out_names = [f"vid_au{int(c[2:])}" for c in _AU_COLS] + ["vid_pitch", "vid_yaw"]

    rows = []
    for seg_idx in range(n_segs):
        t_end = (seg_idx + 1) * SEGMENT_SEC
        f_start = seg_idx * seg_frames
        f_end = f_start + seg_frames
        mask = (fex["frame"] >= f_start) & (fex["frame"] < f_end)
        seg = fex[mask]
        feat_row = {"seconds": t_end}
        for col, out in zip(target_cols, out_names):
            if col in seg.columns and len(seg):
                vals = pd.to_numeric(seg[col], errors="coerce").dropna()
                feat_row[f"{out}_mean"] = float(vals.mean()) if len(vals) else np.nan
                feat_row[f"{out}_std"] = float(vals.std()) if len(vals) > 1 else np.nan
            else:
                feat_row[f"{out}_mean"] = np.nan
                feat_row[f"{out}_std"] = np.nan
        rows.append(feat_row)

    return pd.DataFrame(rows)


def load_or_extract(pid: int, n_segs: int, offset: float = 0.0) -> Optional[pd.DataFrame]:
    frames_path = FRAMES_DIR / f"P{pid}_frames.csv"
    if frames_path.exists():
        frames = pd.read_csv(frames_path)
        if "face_detected" in frames.columns:
            frames = frames[frames["face_detected"] == 1]
        return _aggregate_frames(frames, n_segs, offset)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"P{pid}.csv"
    if cache_path.exists() and offset == 0.0:
        return pd.read_csv(cache_path)

    if _video_path(pid) is None:
        return None

    print(f"    Extracting video features for P{pid} …", flush=True)
    df = extract_video_features(pid, n_segs)
    if df is not None:
        df.to_csv(cache_path, index=False)
    return df


def merge_video_into_seg_table(seg_table: pd.DataFrame,
                                video_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if video_df is None or video_df.empty:
        return seg_table
    vid_cols = [c for c in video_df.columns if c != "seconds"]
    return seg_table.merge(video_df[["seconds"] + vid_cols], on="seconds", how="left")


if __name__ == "__main__":
    import sys
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    print(f"\nProcessing P{pid} …")
    cache = CACHE_DIR / f"P{pid}.csv"
    if cache.exists():
        cache.unlink()
    df = load_or_extract(pid, n_segs=120)
    if df is not None:
        print(f"  {len(df)} segments")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  NaN fraction: {df.isna().mean().mean():.2%}")
        print(df.head(3).to_string())
    else:
        print("  No output.")
