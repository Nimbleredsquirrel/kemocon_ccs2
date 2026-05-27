import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    sig_t, sig_p = y_true.std(), y_pred.std()
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    denom = sig_t ** 2 + sig_p ** 2 + (mu_t - mu_p) ** 2
    if denom == 0:
        return np.nan
    return float(2 * cov / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) < 2:
        return {"ccc": np.nan, "pearson_r": np.nan, "mse": np.nan, "rmse": np.nan}

    r, _ = pearsonr(yt, yp)
    mse = mean_squared_error(yt, yp)
    return {
        "ccc":      ccc(yt, yp),
        "pearson_r": float(r),
        "mse":       float(mse),
        "rmse":      float(np.sqrt(mse)),
    }


def compute_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    from scipy import stats as _stats
    vals = np.array([v for v in values if not np.isnan(v)])
    if len(vals) < 2:
        return np.nan, np.nan
    n = len(vals)
    se = _stats.sem(vals)
    h = se * _stats.t.ppf((1 + confidence) / 2.0, n - 1)
    mu = float(vals.mean())
    return float(mu - h), float(mu + h)


def compute_pred_dispersion(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """std(y_pred) / std(y_true). < 1 means over-smoothed predictions."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 2 or np.std(yt) < 1e-10:
        return np.nan
    return float(np.std(yp) / np.std(yt))


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired per-session CCCs (one-sided: a > b)."""
    from scipy import stats as _stats
    a = np.asarray([v for v in a if not np.isnan(v)], float)
    b = np.asarray([v for v in b if not np.isnan(v)], float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4:
        return {"statistic": np.nan, "p_value": np.nan,
                "n": int(n), "median_delta": np.nan}
    deltas = a - b
    try:
        stat, p = _stats.wilcoxon(deltas, alternative="greater")
    except Exception:
        stat, p = np.nan, np.nan
    return {"statistic": float(stat), "p_value": float(p),
            "n": int(n), "median_delta": float(np.median(deltas))}


def sign_flip_test(a: np.ndarray, b: np.ndarray,
                   n_permutations: int = 10_000, seed: int = 42) -> dict:
    """Permutation sign-flip test on per-session CCC differences."""
    rng = np.random.default_rng(seed)
    a = np.asarray([v for v in a if not np.isnan(v)], float)
    b = np.asarray([v for v in b if not np.isnan(v)], float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4:
        return {"observed": np.nan, "p_value": np.nan, "n": int(n)}
    deltas = a - b
    obs = float(np.mean(deltas))
    signs_matrix = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    null_means = (signs_matrix * deltas).mean(axis=1)
    p = float((null_means >= obs).sum() + 1) / (n_permutations + 1)
    return {"observed": obs, "p_value": p, "n": int(n)}


def compare_conditions(results_df, model: str, target: str,
                       cond_a: str, cond_b: str) -> dict:
    def _cccs(cond):
        sub = results_df[(results_df["model"] == model) &
                         (results_df["target"] == target) &
                         (results_df["condition"] == cond)]
        return sub.sort_values("held_out_session")["ccc"].values

    a_vals = _cccs(cond_a)
    b_vals = _cccs(cond_b)
    wx = wilcoxon_test(a_vals, b_vals)
    sf = sign_flip_test(a_vals, b_vals)
    return {"model": model, "target": target,
            "cond_a": cond_a, "cond_b": cond_b,
            "median_delta": wx["median_delta"],
            "wilcoxon_p":   wx["p_value"],
            "signflip_p":   sf["p_value"],
            "n_sessions":   wx["n"]}


def annotation_agreement(
    seg_tables_a: dict,
    seg_tables_b: dict,
    dyads: list,
    target: str,
) -> "pd.DataFrame":
    """Pearson r between two annotation perspectives, per participant."""
    import pandas as pd
    rows = []
    all_pids = sorted({p for dyad in dyads for p in dyad})
    for pid in all_pids:
        if pid not in seg_tables_a or pid not in seg_tables_b:
            continue
        df_a = seg_tables_a[pid]
        df_b = seg_tables_b[pid]
        if target not in df_a.columns or target not in df_b.columns:
            continue
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < 5:
            continue
        ya = df_a.set_index("seconds").loc[common, target].values.astype(float)
        yb = df_b.set_index("seconds").loc[common, target].values.astype(float)
        mask = np.isfinite(ya) & np.isfinite(yb)
        ya, yb = ya[mask], yb[mask]
        if len(ya) < 5:
            continue
        r, _ = pearsonr(ya, yb)
        rows.append({"pid": pid, "r": float(r), "n": int(len(ya))})
    return pd.DataFrame(rows)
