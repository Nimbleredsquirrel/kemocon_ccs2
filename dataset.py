from typing import Optional, List, Dict, Tuple
"""
Build cross-person prediction pairs from pre-computed per-segment feature tables.

For each dyad (pid_A, pid_B):
  - Synchronous (t):      X = B_features[t],   y = A_label[t]
  - Retrospective (t-1):  X = B_features[t-1], y = A_label[t]
  - Roles are swapped too, so one dyad yields up to 4 DataFrames.

Returns a list of dicts:
  {
    "session_id": int,           # dyad index (1–16)
    "pid_A": int,
    "pid_B": int,
    "target": str,               # "arousal" or "valence"
    "condition": str,            # e.g. "lag_0", "sync_lag_0"
    "X": np.ndarray,             # (n_segments, n_features)
    "y": np.ndarray,             # (n_segments,)
    "feature_names": list[str],
  }
"""
import numpy as np
import pandas as pd
from config import TARGETS, LAGS


FEATURE_COLS: Optional[List[str]] = None  # filled on first call


# ── Synchrony helpers ────────────────────────────────────────────────────────

def _impute_cols(X: np.ndarray) -> np.ndarray:
    """Median-impute columns (for synchrony feature computation)."""
    X = X.copy()
    meds = np.nanmedian(X, axis=0)
    meds = np.where(np.isnan(meds), 0.0, meds)
    for j in range(X.shape[1]):
        bad = np.isnan(X[:, j])
        if bad.any():
            X[bad, j] = meds[j]
    return X


def _rolling_corr(Za: np.ndarray, Zb: np.ndarray, w: int) -> np.ndarray:
    """
    Vectorized rolling Pearson correlation per feature column.
    Returns (n, d) with NaN for the first w-1 timesteps.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    n, d = Za.shape
    out = np.full((n, d), np.nan)
    if n < w:
        return out
    for j in range(d):
        a = Za[:, j]
        b = Zb[:, j]
        aw = sliding_window_view(a, w)          # (n-w+1, w)
        bw = sliding_window_view(b, w)
        ac = aw - aw.mean(axis=1, keepdims=True)
        bc = bw - bw.mean(axis=1, keepdims=True)
        ss_a = (ac ** 2).sum(axis=1)
        ss_b = (bc ** 2).sum(axis=1)
        denom = np.sqrt(ss_a * ss_b)
        corr  = np.where(denom > 1e-10, (ac * bc).sum(axis=1) / denom, 0.0)
        out[w - 1:, j] = corr
    return out


def _compute_synchrony_features(Xa_imp: np.ndarray, Xb_imp: np.ndarray,
                                  feat_cols: list, windows: tuple) -> tuple:
    """
    Compute dyadic synchrony features from median-imputed A and B matrices.

    Features computed:
      - |A - B|              (abs difference per feature)
      - |z_A - z_B|          (z-score distance per feature)
      - A * B                (feature product)
      - cos_sim(A_vec, B_vec) (cosine similarity, scalar per timestep)
      - rolling Pearson corr  (per feature per window)

    Returns (X_sync, sync_names): (n, n_sync_features), list[str].
    """
    def _zscore(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        return (X - mu) / np.where(sd < 1e-8, 1.0, sd)

    Za = _zscore(Xa_imp)
    Zb = _zscore(Xb_imp)

    abs_diff = np.abs(Xa_imp - Xb_imp)
    z_dist   = np.abs(Za - Zb)
    prod     = Xa_imp * Xb_imp

    norms_a  = np.linalg.norm(Xa_imp, axis=1) + 1e-8
    norms_b  = np.linalg.norm(Xb_imp, axis=1) + 1e-8
    cos_sim  = ((Xa_imp * Xb_imp).sum(axis=1) / (norms_a * norms_b)).reshape(-1, 1)

    roll_parts, roll_names = [], []
    for w in windows:
        rc = _rolling_corr(Za, Zb, w)
        roll_parts.append(rc)
        w_sec = w * 5
        roll_names.extend([f"rollcorr_w{w_sec}s_{c}" for c in feat_cols])

    X_sync = np.hstack([abs_diff, z_dist, prod, cos_sim] + roll_parts)
    names  = (
        [f"absdiff_{c}" for c in feat_cols] +
        [f"zdist_{c}"   for c in feat_cols] +
        [f"prod_{c}"    for c in feat_cols] +
        ["cos_sim"]                          +
        roll_names
    )
    return X_sync, names


def _feature_cols(df: pd.DataFrame) -> list[str]:
    meta = {"pid", "seconds", "arousal", "valence"}
    return [c for c in df.columns if c not in meta]


def make_pairs(
    seg_tables: dict[int, pd.DataFrame],   # pid → segment DataFrame
    dyads: list[tuple[int, int]],
    conditions: list[tuple[int, str]] = None,  # [(lag, cond_name)]; None → use LAGS
) -> list[dict]:
    """
    seg_tables: output of build_segment_features, keyed by pid.
    dyads:      list of (pid_a, pid_b) pairs.
    conditions: list of (lag_in_segments, condition_name) to generate.
                Defaults to [(l, f"lag_{l}") for l in LAGS].
    """
    if conditions is None:
        conditions = [(lag, f"lag_{lag}") for lag in LAGS]

    records = []

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]

        # Align on common 'seconds' values
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < 5:
            continue

        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]

        # Both role directions (A listens to B, and B listens to A)
        for (speaker_df, listener_df, p_s, p_l) in [
            (db, da, pid_b, pid_a),
            (da, db, pid_a, pid_b),
        ]:
            feat_cols = _feature_cols(speaker_df.reset_index())
            X_sync = speaker_df[feat_cols].values.astype(float)
            delta_cols = [f"delta_{c}" for c in feat_cols]
            aug_cols = feat_cols + delta_cols

            for target in TARGETS:
                y = listener_df[target].values.astype(float)
                valid = ~np.isnan(y)

                for lag, cond_name in conditions:
                    if lag == 0:
                        X_lag = X_sync
                        y_lag = y
                    else:
                        # Drop first `lag` labels and align features
                        # X[t-lag] predicts y[t]
                        X_lag = X_sync[:-lag]
                        y_lag = y[lag:]

                    # Delta features on the lagged matrix
                    X_delta = np.vstack([
                        np.zeros((1, X_lag.shape[1])),
                        np.diff(X_lag, axis=0),
                    ])
                    X_aug = np.hstack([X_lag, X_delta])

                    valid_lag = ~np.isnan(y_lag)
                    x_use = X_aug[valid_lag]
                    y_use = y_lag[valid_lag]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         p_l,
                        "pid_B":         p_s,
                        "target":        target,
                        "condition":     cond_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": aug_cols,
                    })

    return records


def make_own_signal_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    conditions: list[tuple[int, str]] = None,
) -> list[dict]:
    """
    Own-signal baseline: predict person A's emotion from A's OWN features.
    If cross-person model doesn't beat this, it learns nothing partner-specific.
    """
    if conditions is None:
        conditions = [(0, "own_lag_0")]
    records = []
    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        for pid in [pid_a, pid_b]:
            if pid not in seg_tables:
                continue
            df = seg_tables[pid]
            feat_cols = _feature_cols(df)
            X = df[feat_cols].values.astype(float)
            delta_cols = [f"delta_{c}" for c in feat_cols]
            aug_cols = feat_cols + delta_cols

            for target in TARGETS:
                y = df[target].values.astype(float)
                for lag, cond_name in conditions:
                    if lag == 0:
                        X_lag, y_lag = X, y
                    else:
                        X_lag, y_lag = X[:-lag], y[lag:]
                    X_delta = np.vstack([np.zeros((1, X_lag.shape[1])),
                                         np.diff(X_lag, axis=0)])
                    X_aug = np.hstack([X_lag, X_delta])
                    valid = ~np.isnan(y_lag)
                    x_use, y_use = X_aug[valid], y_lag[valid]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         pid,
                        "pid_B":         pid,   # same person
                        "target":        target,
                        "condition":     cond_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": aug_cols,
                    })
    return records


def make_random_dyad_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    conditions: list[tuple[int, str]] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Negative control: pair each participant with a random non-partner.
    If the cross-person model doesn't beat this, it learns general debate dynamics,
    not dyadic coupling.
    """
    if conditions is None:
        conditions = [(0, "rnd_lag_0")]

    rng = np.random.default_rng(seed)
    pids = [p for dyad in dyads for p in dyad if p in seg_tables]
    records = []

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        for (self_pid, partner_pid) in [(pid_a, pid_b), (pid_b, pid_a)]:
            if self_pid not in seg_tables:
                continue
            # Pick a random participant from a different dyad
            others = [p for p in pids if p != self_pid and p != partner_pid]
            if not others:
                continue
            rand_pid = rng.choice(others)
            if rand_pid not in seg_tables:
                continue

            df_self = seg_tables[self_pid]
            df_rand = seg_tables[rand_pid]
            common = sorted(set(df_self["seconds"]) & set(df_rand["seconds"]))
            if len(common) < 5:
                continue

            da = df_self.set_index("seconds").loc[common]
            db = df_rand.set_index("seconds").loc[common]
            feat_cols = _feature_cols(db.reset_index())
            X_sync = db[feat_cols].values.astype(float)
            delta_cols = [f"delta_{c}" for c in feat_cols]
            aug_cols = feat_cols + delta_cols

            for target in TARGETS:
                y = da[target].values.astype(float)
                for lag, cond_name in conditions:
                    if lag == 0:
                        X_lag, y_lag = X_sync, y
                    else:
                        X_lag, y_lag = X_sync[:-lag], y[lag:]
                    X_delta = np.vstack([np.zeros((1, X_lag.shape[1])),
                                         np.diff(X_lag, axis=0)])
                    X_aug = np.hstack([X_lag, X_delta])
                    valid = ~np.isnan(y_lag)
                    x_use, y_use = X_aug[valid], y_lag[valid]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         self_pid,
                        "pid_B":         rand_pid,
                        "target":        target,
                        "condition":     cond_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": aug_cols,
                    })
    return records


def make_label_ar_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    n_lags: int = 4,
    mode: str = "own",   # "own" | "partner" | "combined"
) -> list[dict]:
    """
    Label autoregression baselines — uses past emotion labels as features.

    mode="own":      X = [A_label[t-1..t-n]]           → tests own-label inertia
    mode="partner":  X = [B_label[t-1..t-n]]           → label-level coupling
    mode="combined": X = [A_label[t-1..t-n], B_label[t-1..t-n]]

    If cross-person physio models don't beat "own", they capture nothing new.
    """
    cond_name = f"label_ar_{mode}"
    records = []
    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < n_lags + 5:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]

        for (listener_df, speaker_df, p_l, p_s) in [
            (da, db, pid_a, pid_b),
            (db, da, pid_b, pid_a),
        ]:
            for target in TARGETS:
                y_self = listener_df[target].values.astype(float)
                y_partner = speaker_df[target].values.astype(float)
                n = len(y_self)
                y_target = y_self[n_lags:]

                if mode == "own":
                    X = np.column_stack([y_self[n_lags - k - 1:n - k - 1]
                                          for k in range(n_lags)])
                    feat_names = [f"own_{target}_lag{k+1}" for k in range(n_lags)]
                elif mode == "partner":
                    X = np.column_stack([y_partner[n_lags - k - 1:n - k - 1]
                                          for k in range(n_lags)])
                    feat_names = [f"partner_{target}_lag{k+1}" for k in range(n_lags)]
                else:  # combined
                    X = np.hstack([
                        np.column_stack([y_self[n_lags - k - 1:n - k - 1]
                                          for k in range(n_lags)]),
                        np.column_stack([y_partner[n_lags - k - 1:n - k - 1]
                                          for k in range(n_lags)]),
                    ])
                    feat_names = (
                        [f"own_{target}_lag{k+1}"     for k in range(n_lags)] +
                        [f"partner_{target}_lag{k+1}" for k in range(n_lags)]
                    )

                valid = ~np.isnan(y_target) & np.all(np.isfinite(X), axis=1)
                x_use = X[valid]
                y_use = y_target[valid]
                if len(y_use) < 5:
                    continue
                records.append({
                    "session_id":    session_idx,
                    "pid_A":         p_l,
                    "pid_B":         p_s,
                    "target":        target,
                    "condition":     cond_name,
                    "X":             x_use,
                    "y":             y_use,
                    "feature_names": feat_names,
                })
    return records


def make_label_delta_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    conditions: list[tuple[int, str]] = None,
) -> list[dict]:
    """
    Predict ΔA_label[t] = A_label[t] - A_label[t-1] from partner features.
    Interpersonal influence may show up better in changes than in levels.
    """
    if conditions is None:
        conditions = [(0, "lag_0")]
    records = []
    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < 6:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]

        for (speaker_df, listener_df, p_s, p_l) in [
            (db, da, pid_b, pid_a),
            (da, db, pid_a, pid_b),
        ]:
            feat_cols = _feature_cols(speaker_df.reset_index())
            X_sync = speaker_df[feat_cols].values.astype(float)
            delta_cols = [f"delta_{c}" for c in feat_cols]
            aug_cols = feat_cols + delta_cols

            for target in TARGETS:
                y_raw = listener_df[target].values.astype(float)
                y_delta = np.concatenate([[np.nan], np.diff(y_raw)])

                for lag, cond_name in conditions:
                    cond_delta = cond_name + "_delta"
                    if lag == 0:
                        X_lag, y_lag = X_sync, y_delta
                    else:
                        X_lag, y_lag = X_sync[:-lag], y_delta[lag:]
                    X_d   = np.vstack([np.zeros((1, X_lag.shape[1])),
                                        np.diff(X_lag, axis=0)])
                    X_aug = np.hstack([X_lag, X_d])
                    valid = ~np.isnan(y_lag)
                    x_use, y_use = X_aug[valid], y_lag[valid]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         p_l,
                        "pid_B":         p_s,
                        "target":        target + "_delta",
                        "condition":     cond_delta,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": aug_cols,
                    })
    return records


def make_missingness_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
) -> list[dict]:
    """
    Red-flag baseline: predict labels from NaN-indicator features only.
    If this performs similarly to real features, missingness patterns are
    confounded with emotion (e.g., device removal during stress).
    """
    records = []
    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < 5:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]

        for (speaker_df, listener_df, p_s, p_l) in [
            (db, da, pid_b, pid_a),
            (da, db, pid_a, pid_b),
        ]:
            feat_cols = _feature_cols(speaker_df.reset_index())
            X_sync = speaker_df[feat_cols].values.astype(float)
            X_miss = np.isnan(X_sync).astype(float)
            miss_cols = [f"miss_{c}" for c in feat_cols]

            for target in TARGETS:
                y = listener_df[target].values.astype(float)
                valid = ~np.isnan(y)
                x_use, y_use = X_miss[valid], y[valid]
                if len(y_use) < 5:
                    continue
                records.append({
                    "session_id":    session_idx,
                    "pid_A":         p_l,
                    "pid_B":         p_s,
                    "target":        target,
                    "condition":     "missingness",
                    "X":             x_use,
                    "y":             y_use,
                    "feature_names": miss_cols,
                })
    return records


def make_incremental_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    n_label_lags: int = 4,
) -> list[dict]:
    """
    Incremental model comparison — each config adds one layer of information:

    M1_own_label_ar  : X = A_label[t-1..t-n]
    M2_own_signal    : X = A_label_lags  + A_physio[t]
    M3_label_coupling: X = A_label_lags  + B_label[t-1..t-n]
    M4_full          : X = A_label_lags  + A_physio[t] + B_physio[t]

    Key quantity: ΔCCC(M4 − M2) = added value of PARTNER physio above own signal.
    """
    records = []
    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < n_label_lags + 5:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]

        feat_cols = _feature_cols(da.reset_index())   # same cols for both after standardise
        X_a = da[feat_cols].values.astype(float)
        X_b = db[feat_cols].values.astype(float)

        for (listener_df, speaker_df, X_listener, X_speaker, p_l, p_s) in [
            (da, db, X_a, X_b, pid_a, pid_b),
            (db, da, X_b, X_a, pid_b, pid_a),
        ]:
            for target in TARGETS:
                y_self    = listener_df[target].values.astype(float)
                y_partner = speaker_df[target].values.astype(float)
                n = len(y_self)
                if n - n_label_lags < 5:
                    continue

                # Window: t = n_label_lags … n-1
                y_tgt = y_self[n_label_lags:]
                own_lags = np.column_stack([y_self[n_label_lags - k - 1:n - k - 1]
                                             for k in range(n_label_lags)])
                par_lags = np.column_stack([y_partner[n_label_lags - k - 1:n - k - 1]
                                             for k in range(n_label_lags)])
                X_l_win = X_listener[n_label_lags:]
                X_s_win = X_speaker[n_label_lags:]

                own_names = [f"own_{target}_lag{k+1}" for k in range(n_label_lags)]
                par_names = [f"par_{target}_lag{k+1}" for k in range(n_label_lags)]

                configs = {
                    "M1_own_label_ar":   (own_lags,
                                          own_names),
                    "M2_own_signal":     (np.hstack([own_lags, X_l_win]),
                                          own_names + [f"own_{c}" for c in feat_cols]),
                    "M3_label_coupling": (np.hstack([own_lags, par_lags]),
                                          own_names + par_names),
                    "M4_full":           (np.hstack([own_lags, X_l_win, X_s_win]),
                                          own_names +
                                          [f"own_{c}" for c in feat_cols] +
                                          [f"par_{c}" for c in feat_cols]),
                }

                valid_tgt = ~np.isnan(y_tgt)
                for cfg_name, (X_cfg, feat_names) in configs.items():
                    x_use = X_cfg[valid_tgt]
                    y_use = y_tgt[valid_tgt]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         p_l,
                        "pid_B":         p_s,
                        "target":        target,
                        "condition":     cfg_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": feat_names,
                    })
    return records


def make_synchrony_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    windows: tuple = (3, 6, 12),
    lags: tuple = (0, 1),
) -> list[dict]:
    """
    Dyadic synchrony features as predictors for cross-person affect.

    Features: |A-B|, |z_A-z_B|, A*B, cosine similarity, rolling correlation.
    Both role directions are built. condition = "sync_lag_{k}".

    These features capture the RELATIONSHIP between A and B, not just B's raw state.
    Key test: CCC(sync) vs CCC(lag_0) — does synchrony add beyond raw partner features?
    """
    records = []
    min_len = max(windows, default=3) + 5

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < min_len:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]
        feat_cols = _feature_cols(da.reset_index())

        Xa_imp = _impute_cols(da[feat_cols].values.astype(float))
        Xb_imp = _impute_cols(db[feat_cols].values.astype(float))
        X_sync, sync_names = _compute_synchrony_features(Xa_imp, Xb_imp, feat_cols, windows)

        # Synchrony is symmetric — both participants share the same sync features
        for listener_df, p_l, p_s in [
            (da, pid_a, pid_b),
            (db, pid_b, pid_a),
        ]:
            for target in TARGETS:
                y = listener_df[target].values.astype(float)
                for lag in lags:
                    cond_name = f"sync_lag_{lag}"
                    if lag == 0:
                        X_lag, y_lag = X_sync, y
                    else:
                        X_lag, y_lag = X_sync[:-lag], y[lag:]
                    valid = ~np.isnan(y_lag)
                    x_use, y_use = X_lag[valid], y_lag[valid]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         p_l,
                        "pid_B":         p_s,
                        "target":        target,
                        "condition":     cond_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": sync_names,
                    })
    return records


def make_synchrony_augmented_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    windows: tuple = (3, 6, 12),
    lags: tuple = (0,),
) -> list[dict]:
    """
    Raw partner features + synchrony features (M4 in the synchrony ladder).

    Compared against make_pairs (M3 = raw partner features only).
    Key test: ΔCCC(sync_aug_lag_0 − lag_0) = added value of synchrony features.
    condition = "sync_aug_lag_{k}"
    """
    records = []
    min_len = max(windows, default=3) + 5

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < min_len:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]
        feat_cols = _feature_cols(da.reset_index())

        Xa_imp = _impute_cols(da[feat_cols].values.astype(float))
        Xb_imp = _impute_cols(db[feat_cols].values.astype(float))
        X_sync, sync_names = _compute_synchrony_features(Xa_imp, Xb_imp, feat_cols, windows)

        for listener_df, speaker_imp, p_l, p_s in [
            (da, Xb_imp, pid_a, pid_b),
            (db, Xa_imp, pid_b, pid_a),
        ]:
            raw_names = [f"partner_{c}" for c in feat_cols]
            X_aug     = np.hstack([speaker_imp, X_sync])
            aug_names = raw_names + sync_names

            for target in TARGETS:
                y = listener_df[target].values.astype(float)
                for lag in lags:
                    cond_name = f"sync_aug_lag_{lag}"
                    if lag == 0:
                        X_lag, y_lag = X_aug, y
                    else:
                        X_lag, y_lag = X_aug[:-lag], y[lag:]
                    valid = ~np.isnan(y_lag)
                    x_use, y_use = X_lag[valid], y_lag[valid]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         p_l,
                        "pid_B":         p_s,
                        "target":        target,
                        "condition":     cond_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": aug_names,
                    })
    return records


def make_random_synchrony_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    windows: tuple = (3, 6, 12),
    seed: int = 42,
) -> list[dict]:
    """
    Synchrony control: compute synchrony between A and a random non-partner.
    If real-dyad synchrony > this, the result is specific to the actual pair.
    condition = "sync_rnd"
    """
    rng = np.random.default_rng(seed)
    pids = [p for dyad in dyads for p in dyad if p in seg_tables]
    records = []
    min_len = max(windows, default=3) + 5

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        for self_pid, partner_pid in [(pid_a, pid_b), (pid_b, pid_a)]:
            if self_pid not in seg_tables:
                continue
            others = [p for p in pids if p != self_pid and p != partner_pid]
            if not others:
                continue
            rand_pid = int(rng.choice(others))
            if rand_pid not in seg_tables:
                continue
            df_self = seg_tables[self_pid]
            df_rand = seg_tables[rand_pid]
            common  = sorted(set(df_self["seconds"]) & set(df_rand["seconds"]))
            if len(common) < min_len:
                continue
            da = df_self.set_index("seconds").loc[common]
            db = df_rand.set_index("seconds").loc[common]
            feat_cols = _feature_cols(da.reset_index())

            Xa_imp = _impute_cols(da[feat_cols].values.astype(float))
            Xb_imp = _impute_cols(db[feat_cols].values.astype(float))
            X_sync, sync_names = _compute_synchrony_features(Xa_imp, Xb_imp, feat_cols, windows)

            for target in TARGETS:
                y = da[target].values.astype(float)
                valid = ~np.isnan(y)
                x_use, y_use = X_sync[valid], y[valid]
                if len(y_use) < 5:
                    continue
                records.append({
                    "session_id":    session_idx,
                    "pid_A":         self_pid,
                    "pid_B":         rand_pid,
                    "target":        target,
                    "condition":     "sync_rnd",
                    "X":             x_use,
                    "y":             y_use,
                    "feature_names": sync_names,
                })
    return records


def make_circular_shift_synchrony_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    windows: tuple = (3, 6, 12),
    min_shift: int = 6,
    seed: int = 42,
) -> list[dict]:
    """
    Synchrony control: compute synchrony after circularly shifting partner features ≥30s.
    Preserves signal distribution but destroys real-time interpersonal coupling.
    condition = "sync_circ"
    """
    rng = np.random.default_rng(seed)
    records = []
    min_len = max(windows, default=3) + min_shift + 5

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < min_len:
            continue
        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]
        feat_cols = _feature_cols(da.reset_index())

        Xa_imp = _impute_cols(da[feat_cols].values.astype(float))
        Xb_imp = _impute_cols(db[feat_cols].values.astype(float))
        n = len(Xa_imp)
        shift = int(rng.integers(min_shift, max(min_shift + 1, n // 2)))

        for listener_df, listener_imp, speaker_imp, p_l, p_s in [
            (da, Xa_imp, np.roll(Xb_imp, shift, axis=0), pid_a, pid_b),
            (db, Xb_imp, np.roll(Xa_imp, shift, axis=0), pid_b, pid_a),
        ]:
            X_sync, sync_names = _compute_synchrony_features(listener_imp, speaker_imp, feat_cols, windows)
            for target in TARGETS:
                y = listener_df[target].values.astype(float)
                valid = ~np.isnan(y)
                x_use, y_use = X_sync[valid], y[valid]
                if len(y_use) < 5:
                    continue
                records.append({
                    "session_id":    session_idx,
                    "pid_A":         p_l,
                    "pid_B":         p_s,
                    "target":        target,
                    "condition":     "sync_circ",
                    "X":             x_use,
                    "y":             y_use,
                    "feature_names": sync_names,
                })
    return records


def make_circular_shift_pairs(
    seg_tables: dict[int, pd.DataFrame],
    dyads: list[tuple[int, int]],
    min_shift: int = 6,
    conditions: list[tuple[int, str]] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Negative control: shift partner features by a random large offset within session.
    Preserves the statistical properties of partner features but destroys temporal coupling.
    min_shift: minimum shift in segments (default 6 = 30 s, well beyond any real contagion lag).
    """
    if conditions is None:
        conditions = [(0, "circ_lag_0")]

    rng = np.random.default_rng(seed)
    records = []

    for session_idx, (pid_a, pid_b) in enumerate(dyads, start=1):
        if pid_a not in seg_tables or pid_b not in seg_tables:
            continue
        df_a = seg_tables[pid_a]
        df_b = seg_tables[pid_b]
        common = sorted(set(df_a["seconds"]) & set(df_b["seconds"]))
        if len(common) < max(10, min_shift + 5):
            continue

        da = df_a.set_index("seconds").loc[common]
        db = df_b.set_index("seconds").loc[common]

        for (speaker_df, listener_df, p_s, p_l) in [
            (db, da, pid_b, pid_a),
            (da, db, pid_a, pid_b),
        ]:
            feat_cols = _feature_cols(speaker_df.reset_index())
            X_sync = speaker_df[feat_cols].values.astype(float)
            n = len(X_sync)
            # Random circular shift >= min_shift
            shift = int(rng.integers(min_shift, max(min_shift + 1, n // 2)))
            X_shifted = np.roll(X_sync, shift, axis=0)
            delta_cols = [f"delta_{c}" for c in feat_cols]
            aug_cols = feat_cols + delta_cols

            for target in TARGETS:
                y = listener_df[target].values.astype(float)
                for lag, cond_name in conditions:
                    if lag == 0:
                        X_lag, y_lag = X_shifted, y
                    else:
                        X_lag, y_lag = X_shifted[:-lag], y[lag:]
                    X_delta = np.vstack([np.zeros((1, X_lag.shape[1])),
                                         np.diff(X_lag, axis=0)])
                    X_aug = np.hstack([X_lag, X_delta])
                    valid = ~np.isnan(y_lag)
                    x_use, y_use = X_aug[valid], y_lag[valid]
                    if len(y_use) < 5:
                        continue
                    records.append({
                        "session_id":    session_idx,
                        "pid_A":         p_l,
                        "pid_B":         p_s,
                        "target":        target,
                        "condition":     cond_name,
                        "X":             x_use,
                        "y":             y_use,
                        "feature_names": aug_cols,
                    })
    return records
