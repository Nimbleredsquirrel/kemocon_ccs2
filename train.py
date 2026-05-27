import numpy as np
import pandas as pd
import models as _models_module
from evaluate import compute_metrics, compute_ci, compute_pred_dispersion
from config import TARGETS, LAGS


def run_loso(pairs: list[dict]) -> pd.DataFrame:
    session_ids = sorted({p["session_id"] for p in pairs})
    all_conds = sorted({p["condition"] for p in pairs})
    records = []

    n_sessions = len(session_ids)
    for fold_i, held_out in enumerate(session_ids, 1):
        print(f"  fold {fold_i}/{n_sessions} (held-out session {held_out}) …", flush=True)
        train_pairs = [p for p in pairs if p["session_id"] != held_out]
        test_pairs = [p for p in pairs if p["session_id"] == held_out]

        assert not any(p["session_id"] == held_out for p in train_pairs), \
            f"LEAKAGE: session {held_out} appears in train set!"

        for target in TARGETS:
            for cond in all_conds:
                tr = [p for p in train_pairs if p["target"] == target and p["condition"] == cond]
                te = [p for p in test_pairs if p["target"] == target and p["condition"] == cond]

                if not tr or not te:
                    continue

                X_train = np.vstack([p["X"] for p in tr])
                y_train = np.concatenate([p["y"] for p in tr])
                groups_train = np.concatenate([
                    np.full(len(p["y"]), p["session_id"]) for p in tr
                ])
                X_test = np.vstack([p["X"] for p in te])
                y_test = np.concatenate([p["y"] for p in te])

                if len(y_test) < 2:
                    continue

                for model in _models_module.all_models():
                    print(f"    {model.name} ({target} {cond}) …", flush=True)
                    try:
                        if hasattr(model, "fit") and "groups" in model.fit.__code__.co_varnames:
                            model.fit(X_train, y_train, groups=groups_train)
                        else:
                            model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        metrics = compute_metrics(y_test, y_pred)
                        metrics["pred_dispersion"] = compute_pred_dispersion(y_test, y_pred)
                        print(f"    {model.name} CCC={metrics['ccc']:.4f}", flush=True)
                    except Exception as exc:
                        print(f"  [WARN] {model.name} {target} {cond} session={held_out}: {exc}")
                        metrics = {"ccc": np.nan, "pearson_r": np.nan,
                                   "mse": np.nan, "rmse": np.nan,
                                   "pred_dispersion": np.nan}

                    records.append({
                        "held_out_session": held_out,
                        "model": model.name,
                        "target": target,
                        "condition": cond,
                        **metrics,
                    })

    return pd.DataFrame(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()
    rows = []
    for (model, target, condition), grp in df.groupby(["model", "target", "condition"]):
        ci_lo, ci_hi = compute_ci(grp["ccc"].values)
        disp_col = grp["pred_dispersion"] if "pred_dispersion" in grp.columns else None
        rows.append({
            "model":          model,
            "target":         target,
            "condition":      condition,
            "ccc_mean":       grp["ccc"].mean(),
            "ccc_std":        grp["ccc"].std(),
            "ccc_ci_lo":      ci_lo,
            "ccc_ci_hi":      ci_hi,
            "r_mean":         grp["pearson_r"].mean(),
            "rmse_mean":      grp["rmse"].mean(),
            "pred_disp_mean": float(disp_col.mean()) if disp_col is not None else np.nan,
            "n_sessions":     len(grp),
        })
    return (pd.DataFrame(rows)
            .sort_values(["target", "condition", "ccc_mean"],
                         ascending=[True, True, False])
            .reset_index(drop=True))
