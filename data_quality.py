"""
Summarise data quality from the pre-computed tables and from the loaded
segment feature tables to determine which sessions are usable.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_ROOT, TARGETS


QT_DIR = DATA_ROOT / "data_quality_tables"


def load_quality_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for csv in QT_DIR.glob("*.csv"):
        tables[csv.stem] = pd.read_csv(csv)
    return tables


def quality_summary_e4(completeness: pd.DataFrame) -> pd.DataFrame:
    """
    Flag participants whose E4 EDA completeness is < 0.05 (essentially missing).
    Returns per-participant summary.
    """
    df = completeness.copy()
    df.insert(0, "pid", range(1, len(df) + 1))
    df["eda_ok"] = df["EDA"].apply(lambda v: pd.to_numeric(v, errors="coerce") > 0.05)
    df["all_ok"] = df[["ACC","BVP","HR","IBI","TEMP"]].apply(
        lambda row: all(pd.to_numeric(v, errors="coerce") > 0.5
                        for v in row), axis=1)
    return df


def quality_summary_neuro(completeness: pd.DataFrame) -> pd.DataFrame:
    df = completeness.copy()
    df.insert(0, "pid", range(1, len(df) + 1))
    for col in ["Attention", "BrainWave", "Meditation", "Polar_HR"]:
        df[f"{col}_ok"] = df[col].apply(
            lambda v: pd.to_numeric(v, errors="coerce") > 0.5
        )
    return df


def nan_fraction_per_session(seg_tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    For each participant, report the fraction of NaN values per feature column
    across all segments.  High NaN → signal missing.
    """
    rows = []
    for pid, df in seg_tables.items():
        feat_cols = [c for c in df.columns
                     if c not in {"pid", "seconds", "arousal", "valence"}]
        nan_frac = df[feat_cols].isna().mean()
        rows.append({"pid": pid,
                     "n_segments": len(df),
                     "overall_nan_frac": float(nan_frac.mean()),
                     "max_col_nan_frac": float(nan_frac.max()),
                     "worst_col": nan_frac.idxmax()})
    return pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)


def usable_dyads(nan_df: pd.DataFrame,
                 dyads: list[tuple[int, int]],
                 max_nan: float = 0.8) -> list[tuple[int, int]]:
    """
    Return dyads where BOTH participants have overall_nan_frac <= max_nan
    and at least 10 segments.
    """
    good = set(nan_df[(nan_df["overall_nan_frac"] <= max_nan) &
                      (nan_df["n_segments"] >= 10)]["pid"])
    return [(a, b) for a, b in dyads if a in good and b in good]


def print_quality_report(seg_tables: dict, dyads: list) -> None:
    nan_df = nan_fraction_per_session(seg_tables)

    print("\n=== Data Quality: NaN fraction per participant ===")
    print(nan_df.to_string(index=False))

    good = usable_dyads(nan_df, dyads)
    print(f"\nUsable dyads (both members with <80% NaN and ≥10 segments): "
          f"{len(good)} / {len(dyads)}")
    for d in good:
        print(f"  {d}")
