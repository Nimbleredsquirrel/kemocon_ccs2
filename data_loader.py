from typing import Optional, Dict
import pandas as pd
import numpy as np
from pathlib import Path
from config import E4_DIR, NEURO_DIR, ANNOT_DIR, METADATA_DIR

PARTNER_ANNOT_DIR = ANNOT_DIR.parent / "partner_annotations"
SELF_ANNOT_DIR = ANNOT_DIR.parent / "self_annotations"


def load_metadata() -> tuple:
    subjects = pd.read_csv(METADATA_DIR / "subjects.csv")
    avail = pd.read_csv(METADATA_DIR / "data_availability.csv")
    return subjects, avail


def get_start_time(pid: int, subjects: pd.DataFrame) -> float:
    row = subjects[subjects["pid"] == pid].iloc[0]
    return float(row["startTime"])


def get_init_time(pid: int, subjects: pd.DataFrame) -> float:
    row = subjects[subjects["pid"] == pid].iloc[0]
    return float(row["initTime"])


E4_SIGNALS = {
    "E4_HR":   (1,   "value"),
    "E4_EDA":  (4,   "value"),
    "E4_TEMP": (4,   "value"),
    "E4_IBI":  (1,   "value"),
    "E4_BVP":  (64,  "value"),
    "E4_ACC":  (32,  None),
}


def _load_signal_raw(path: Path, start_ms: float) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return None
    df["t_sec"] = (df["timestamp"] - start_ms) / 1000.0
    return df


def load_e4_signal(pid: int, signal: str, start_ms: float) -> Optional[pd.DataFrame]:
    df = _load_signal_raw(E4_DIR / str(pid) / f"{signal}.csv", start_ms)
    if df is None:
        return None
    return df[df["t_sec"] >= 0].copy()


def load_e4_all(pid: int, start_ms: float) -> Dict[str, Optional[pd.DataFrame]]:
    return {sig: load_e4_signal(pid, sig, start_ms) for sig in E4_SIGNALS}


NEURO_SIGNALS = ["BrainWave", "Attention", "Meditation", "Polar_HR"]


def load_neuro_signal(pid: int, signal: str, start_ms: float) -> Optional[pd.DataFrame]:
    df = _load_signal_raw(NEURO_DIR / str(pid) / f"{signal}.csv", start_ms)
    if df is None:
        return None
    return df[df["t_sec"] >= 0].copy()


def load_neuro_all(pid: int, start_ms: float) -> Dict[str, Optional[pd.DataFrame]]:
    return {sig: load_neuro_signal(pid, sig, start_ms) for sig in NEURO_SIGNALS}


def load_baseline_stats(pid: int, subjects: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    start_ms = get_start_time(pid, subjects)
    stats: Dict[str, Dict[str, float]] = {}

    for sig in E4_SIGNALS:
        df = _load_signal_raw(E4_DIR / str(pid) / f"{sig}.csv", start_ms)
        if df is None:
            continue
        baseline = df[(df["t_sec"] < 0)]
        if sig == "E4_ACC":
            if all(c in baseline.columns for c in ("x", "y", "z")):
                vals = np.sqrt((baseline[["x", "y", "z"]].values.astype(float) ** 2).sum(axis=1))
            else:
                continue
        elif "value" in baseline.columns:
            vals = baseline["value"].replace(0, np.nan).dropna().values.astype(float)
        else:
            continue
        if len(vals) < 2:
            continue
        stats[sig] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    for sig in NEURO_SIGNALS:
        df = _load_signal_raw(NEURO_DIR / str(pid) / f"{sig}.csv", start_ms)
        if df is None:
            continue
        baseline = df[(df["t_sec"] < 0)]
        if sig == "BrainWave":
            band_cols = [c for c in ["delta", "theta", "lowAlpha", "highAlpha",
                                      "lowBeta", "highBeta", "lowGamma", "middleGamma"]
                         if c in baseline.columns]
            for band in band_cols:
                vals = baseline[band].replace(0, np.nan).dropna().values.astype(float)
                if len(vals) >= 2:
                    stats[f"BrainWave_{band}"] = {
                        "mean": float(np.nanmean(vals)),
                        "std":  float(np.nanstd(vals)),
                    }
        elif "value" in baseline.columns:
            vals = baseline["value"].replace(0, np.nan).dropna().values.astype(float)
            if len(vals) >= 2:
                stats[sig] = {"mean": float(np.nanmean(vals)),
                               "std":  float(np.nanstd(vals))}

    return stats


def _load_annot_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[["seconds", "arousal", "valence"]].copy()
    df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")
    df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
    df = df.dropna(subset=["arousal", "valence"])
    return df.reset_index(drop=True)


def load_annotations(pid: int) -> Optional[pd.DataFrame]:
    return _load_annot_file(ANNOT_DIR / f"P{pid}.external.csv")


def load_partner_annotations(pid: int) -> Optional[pd.DataFrame]:
    return _load_annot_file(PARTNER_ANNOT_DIR / f"P{pid}.partner.csv")


def load_self_annotations(pid: int) -> Optional[pd.DataFrame]:
    return _load_annot_file(SELF_ANNOT_DIR / f"P{pid}.self.csv")
