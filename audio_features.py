from typing import Optional
import numpy as np
import pandas as pd
import soundfile as sf
import opensmile
from pathlib import Path

from config import DATA_ROOT, DYADS, SEGMENT_SEC

AUDIO_DIR = DATA_ROOT / "debate_audios"
CACHE_DIR = DATA_ROOT / "audio_cache"

SELECTED_COLS = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
]

_smile = None


def _get_smile():
    global _smile
    if _smile is None:
        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _smile


def _wav_path(pid_a: int, pid_b: int) -> Optional[Path]:
    p = AUDIO_DIR / f"p{pid_a}.p{pid_b}.wav"
    return p if p.exists() else None


def extract_audio_features(pid_a: int, pid_b: int,
                            n_segs: int,
                            offset: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    wav_path = _wav_path(pid_a, pid_b)
    if wav_path is None:
        empty = pd.DataFrame({"seconds": [(i + 1) * SEGMENT_SEC for i in range(n_segs)]})
        for col in SELECTED_COLS:
            empty[f"aud_{col}"] = np.nan
        return empty, empty.copy()

    y, sr = sf.read(str(wav_path))
    if y.ndim == 1:
        ch_a = ch_b = y
    else:
        ch_a = y[:, 0]
        ch_b = y[:, 1]

    sm = _get_smile()
    seg_samples = int(SEGMENT_SEC * sr)
    rows_a, rows_b = [], []

    offset_samples = int(offset * sr)
    for i in range(n_segs):
        t_end = (i + 1) * SEGMENT_SEC
        s = max(0, i * seg_samples - offset_samples)
        e = s + seg_samples

        def _extract(ch):
            seg = ch[s:e]
            if len(seg) < sr // 2:
                return {col: np.nan for col in SELECTED_COLS}
            df = sm.process_signal(seg, sr)
            return {col: float(df[col].iloc[0])
                    for col in SELECTED_COLS if col in df.columns}

        rows_a.append({"seconds": t_end, **{f"aud_{k}": v
                       for k, v in _extract(ch_a).items()}})
        rows_b.append({"seconds": t_end, **{f"aud_{k}": v
                       for k, v in _extract(ch_b).items()}})

    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def load_or_extract(pid_a: int, pid_b: int,
                    n_segs_a: int, n_segs_b: int,
                    offset: float = 0.0) -> tuple[Optional[pd.DataFrame],
                                                   Optional[pd.DataFrame]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{int(offset * 1000)}ms" if offset > 0 else ""
    path_a = CACHE_DIR / f"P{pid_a}{suffix}.csv"
    path_b = CACHE_DIR / f"P{pid_b}{suffix}.csv"

    if path_a.exists() and path_b.exists():
        return pd.read_csv(path_a), pd.read_csv(path_b)

    print(f"    Extracting audio for dyad ({pid_a},{pid_b}) offset={offset}s …", flush=True)
    n_segs = max(n_segs_a, n_segs_b)
    df_a, df_b = extract_audio_features(pid_a, pid_b, n_segs, offset=offset)
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    return df_a, df_b


def merge_audio_into_seg_table(seg_table: pd.DataFrame,
                                audio_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if audio_df is None or audio_df.empty:
        return seg_table
    aud_cols = [c for c in audio_df.columns if c != "seconds"]
    return seg_table.merge(audio_df[["seconds"] + aud_cols], on="seconds", how="left")


if __name__ == "__main__":
    from data_loader import load_metadata, load_annotations

    subjects, avail = load_metadata()
    for pid_a, pid_b in DYADS:
        ann_a = load_annotations(pid_a)
        ann_b = load_annotations(pid_b)
        n_a = len(ann_a) if ann_a is not None else 170
        n_b = len(ann_b) if ann_b is not None else 170
        df_a, df_b = load_or_extract(pid_a, pid_b, n_a, n_b)
        print(f"  P{pid_a}: {len(df_a)} segments  P{pid_b}: {len(df_b)} segments")
    print("Audio feature extraction complete.")
