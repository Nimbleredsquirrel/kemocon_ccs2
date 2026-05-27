from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from config import SEGMENT_SEC


def _window(df: Optional[pd.DataFrame], t_end: float) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    t_start = t_end - SEGMENT_SEC
    mask = (df["t_sec"] > t_start) & (df["t_sec"] <= t_end)
    sub = df[mask]
    return sub if not sub.empty else None


def _stats(arr: np.ndarray, prefix: str) -> dict:
    if len(arr) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    return {f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr))}


def _stats_slope(arr: np.ndarray, prefix: str) -> dict:
    d = _stats(arr, prefix)
    if len(arr) >= 2:
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])
    else:
        slope = np.nan
    d[f"{prefix}_slope"] = slope
    return d


def features_e4(e4: dict, t_end: float, win_offset: float = 0.0) -> dict:
    feats = {}
    t = t_end - win_offset

    w = _window(e4.get("E4_HR"), t)
    if w is not None:
        feats.update(_stats(w["value"].values, "e4_hr"))
    else:
        feats.update({"e4_hr_mean": np.nan, "e4_hr_std": np.nan})

    w = _window(e4.get("E4_EDA"), t)
    if w is not None:
        arr = w["value"].values
        feats.update(_stats_slope(arr, "e4_eda"))
        feats["e4_eda_range"] = float(np.nanmax(arr) - np.nanmin(arr))
        peaks, _ = find_peaks(arr)
        feats["e4_eda_npeaks"] = float(len(peaks))
    else:
        feats.update({"e4_eda_mean": np.nan, "e4_eda_std": np.nan,
                      "e4_eda_slope": np.nan, "e4_eda_range": np.nan,
                      "e4_eda_npeaks": np.nan})

    w = _window(e4.get("E4_TEMP"), t)
    if w is not None:
        feats.update(_stats_slope(w["value"].values, "e4_temp"))
    else:
        feats.update({"e4_temp_mean": np.nan, "e4_temp_std": np.nan,
                      "e4_temp_slope": np.nan})

    w = _window(e4.get("E4_IBI"), t)
    if w is not None:
        arr = w["value"].values
        feats.update(_stats(arr, "e4_ibi"))
        if len(arr) >= 2:
            diffs = np.diff(arr)
            feats["e4_ibi_rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
        else:
            feats["e4_ibi_rmssd"] = np.nan
    else:
        feats.update({"e4_ibi_mean": np.nan, "e4_ibi_std": np.nan,
                      "e4_ibi_rmssd": np.nan})

    w = _window(e4.get("E4_BVP"), t)
    if w is not None:
        arr = w["value"].values
        feats["e4_bvp_mean"] = float(np.mean(arr))
        feats["e4_bvp_std"] = float(np.std(arr))
        feats["e4_bvp_rms"] = float(np.sqrt(np.mean(arr ** 2)))
        feats["e4_bvp_range"] = float(np.max(arr) - np.min(arr))
    else:
        feats.update({"e4_bvp_mean": np.nan, "e4_bvp_std": np.nan,
                      "e4_bvp_rms": np.nan, "e4_bvp_range": np.nan})

    w = _window(e4.get("E4_ACC"), t)
    if w is not None:
        if all(c in w.columns for c in ("x", "y", "z")):
            mag = np.sqrt((w[["x", "y", "z"]].values.astype(float) ** 2).sum(axis=1))
        else:
            val_cols = [c for c in w.columns if c.startswith("value") or c == "value"]
            mag = w[val_cols[0]].values.astype(float) if val_cols else np.array([np.nan])
        feats["e4_acc_mean"] = float(np.nanmean(mag))
        feats["e4_acc_std"] = float(np.nanstd(mag))
    else:
        feats.update({"e4_acc_mean": np.nan, "e4_acc_std": np.nan})

    return feats


BRAINWAVE_BANDS = ["delta", "theta", "lowAlpha", "highAlpha",
                   "lowBeta", "highBeta", "lowGamma", "middleGamma"]


def features_neuro(neuro: dict, t_end: float, win_offset: float = 0.0) -> dict:
    feats = {}
    t = t_end - win_offset

    w = _window(neuro.get("BrainWave"), t)
    for band in BRAINWAVE_BANDS:
        key = f"bw_{band}_mean"
        if w is not None and band in w.columns:
            valid = w[band].replace(0, np.nan).dropna()
            feats[key] = float(valid.mean()) if len(valid) > 0 else np.nan
        else:
            feats[key] = np.nan

    def _bw_mean(band):
        return feats.get(f"bw_{band}_mean", np.nan)

    theta = _bw_mean("theta")
    low_a, high_a = _bw_mean("lowAlpha"), _bw_mean("highAlpha")
    low_b, high_b = _bw_mean("lowBeta"), _bw_mean("highBeta")
    alpha = np.nanmean([low_a, high_a]) if not (np.isnan(low_a) and np.isnan(high_a)) else np.nan
    beta = np.nanmean([low_b, high_b]) if not (np.isnan(low_b) and np.isnan(high_b)) else np.nan
    feats["bw_theta_alpha_ratio"] = (theta / alpha) if (not np.isnan(theta) and not np.isnan(alpha) and alpha > 0) else np.nan
    denom = (alpha if not np.isnan(alpha) else 0) + (theta if not np.isnan(theta) else 0)
    feats["bw_engagement_idx"] = (beta / denom) if (not np.isnan(beta) and denom > 0) else np.nan

    w = _window(neuro.get("Attention"), t)
    if w is not None and "value" in w.columns:
        valid = w["value"].replace(0, np.nan).dropna()
        feats["attention_mean"] = float(valid.mean()) if len(valid) > 0 else np.nan
        feats["attention_std"] = float(valid.std()) if len(valid) > 1 else np.nan
    else:
        feats.update({"attention_mean": np.nan, "attention_std": np.nan})

    w = _window(neuro.get("Meditation"), t)
    if w is not None and "value" in w.columns:
        valid = w["value"].replace(0, np.nan).dropna()
        feats["meditation_mean"] = float(valid.mean()) if len(valid) > 0 else np.nan
        feats["meditation_std"] = float(valid.std()) if len(valid) > 1 else np.nan
    else:
        feats.update({"meditation_mean": np.nan, "meditation_std": np.nan})

    w = _window(neuro.get("Polar_HR"), t)
    if w is not None and "value" in w.columns:
        feats.update(_stats(w["value"].values, "polar_hr"))
    else:
        feats.update({"polar_hr_mean": np.nan, "polar_hr_std": np.nan})

    return feats


_BL_MAP = {
    "e4_hr":      "E4_HR",
    "e4_eda":     "E4_EDA",
    "e4_temp":    "E4_TEMP",
    "e4_ibi":     "E4_IBI",
    "e4_bvp":     "E4_BVP",
    "e4_acc":     "E4_ACC",
    "attention":  "Attention",
    "meditation": "Meditation",
    "polar_hr":   "Polar_HR",
}

_DERIVED_SUFFIXES = {"_std", "_slope", "_range", "_npeaks", "_rmssd"}
_DERIVED_EXACT = {"bw_theta_alpha_ratio", "bw_engagement_idx"}


def _apply_baseline_normalization(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    feat_cols = [c for c in df.columns
                 if c not in {"pid", "seconds", "arousal", "valence"}
                 and not c.startswith(("aud_", "vid_"))]

    for col in feat_cols:
        if col in _DERIVED_EXACT:
            continue
        if any(col.endswith(s) for s in _DERIVED_SUFFIXES):
            continue

        bl_key = None
        for prefix, key in _BL_MAP.items():
            if col.startswith(prefix):
                bl_key = key
                break
        if bl_key is None and col.startswith("bw_"):
            parts = col.split("_")
            if len(parts) >= 3:
                bl_key = f"BrainWave_{parts[1]}"

        if bl_key and bl_key in stats:
            mu = stats[bl_key]["mean"]
            sig = stats[bl_key]["std"] + 1e-8
            df[col] = (df[col] - mu) / sig

    return df


def build_segment_features(pid: int, e4: dict, neuro: dict,
                            annot: pd.DataFrame,
                            baseline_stats: dict = None,
                            win_offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for _, row in annot.iterrows():
        t_end = float(row["seconds"])
        f = {"pid": pid, "seconds": t_end,
             "arousal": row["arousal"], "valence": row["valence"]}
        f.update(features_e4(e4, t_end, win_offset))
        f.update(features_neuro(neuro, t_end, win_offset))
        rows.append(f)
    df = pd.DataFrame(rows)
    if baseline_stats:
        df = _apply_baseline_normalization(df, baseline_stats)
    return df
