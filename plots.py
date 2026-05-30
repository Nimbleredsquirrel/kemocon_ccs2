import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

FIG_DIR = Path(__file__).parent / "results" / "figures"

PALETTE = {"sync": "#4C72B0", "retro": "#DD8452"}
MODEL_ORDER = ["MeanBaseline", "Ridge", "SVR", "CatBoost", "LSTM"]

def plot_ccc_comparison(summary: pd.DataFrame, tag: str = "") -> None:
    """
    Grouped bar chart: mean CCC ± 95% CI per model, one bar group per condition.
    One subplot per target.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    targets = summary["target"].unique()
    conditions = sorted(summary["condition"].unique())
    cmap = plt.cm.get_cmap("tab10", len(conditions))
    colors = {c: cmap(i) for i, c in enumerate(conditions)}

    fig, axes = plt.subplots(1, len(targets), figsize=(max(14, len(conditions)*3), 5),
                              sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub = summary[summary["target"] == target]
        models_present = sub["model"].unique().tolist()
        x = np.arange(len(models_present))
        n_conds = len(conditions)
        w = 0.8 / max(n_conds, 1)

        for i, cond in enumerate(conditions):
            vals, yerr_lo, yerr_hi = [], [], []
            for m in models_present:
                row = sub[(sub["model"] == m) & (sub["condition"] == cond)]
                if len(row):
                    mu = float(row["ccc_mean"].values[0])
                    ci_lo = row["ccc_ci_lo"].values[0] if "ccc_ci_lo" in row.columns else np.nan
                    ci_hi = row["ccc_ci_hi"].values[0] if "ccc_ci_hi" in row.columns else np.nan
                    vals.append(mu)
                    yerr_lo.append(mu - ci_lo if not np.isnan(ci_lo) else 0)
                    yerr_hi.append(ci_hi - mu if not np.isnan(ci_hi) else 0)
                else:
                    vals.append(0); yerr_lo.append(0); yerr_hi.append(0)
            offset = (i - n_conds / 2 + 0.5) * w
            ax.bar(x + offset, vals, w,
                   yerr=[yerr_lo, yerr_hi], capsize=3,
                   color=colors[cond], alpha=0.85, label=cond)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models_present, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"CCC — {target}", fontsize=12)
        ax.set_ylabel("Mean CCC ± 95% CI")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Cross-Person Emotion Prediction: Model × Condition" +
                 (f" ({tag})" if tag else ""), fontsize=13)
    plt.tight_layout()
    path = FIG_DIR / f"ccc_comparison{'_' + tag if tag else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_feature_importance(json_path: Path, target: str, top_n: int = 20) -> None:
    if not json_path.exists():
        return
    with open(json_path) as f:
        imp = json.load(f)

    items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [_pretty(k) for k, _ in items]
    vals = [v for _, v in items]

    colors = [_feat_color(k) for k, _ in items]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(names)), vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — {target}", fontsize=12)

    # Legend for modality colours
    patches = [
        mpatches.Patch(color="#4C72B0", label="Physiological"),
        mpatches.Patch(color="#DD8452", label="Audio"),
        mpatches.Patch(color="#55A868", label="Video"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / f"feature_importance_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_per_session_ccc(results_df: pd.DataFrame, target: str,
                          condition: str = "lag_0") -> None:
    sub = results_df[(results_df["target"] == target) &
                     (results_df["condition"] == condition)]
    if sub.empty:
        return
    models = sub["model"].unique().tolist()
    data = [sub[sub["model"] == m]["ccc"].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(max(10, len(models)*1.5), 5))
    parts = ax.violinplot(data, positions=range(len(models)),
                           showmeans=True, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.5)
    ax.scatter(range(len(models)),
               [np.nanmean(d) for d in data],
               color="white", s=30, zorder=3)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", label="Chance")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("CCC per session")
    ax.set_title(f"Per-Session CCC — {target} ({condition})", fontsize=12)
    ax.legend()
    plt.tight_layout()
    path = FIG_DIR / f"per_session_ccc_{target}_{condition}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_h2_delta(summary: pd.DataFrame) -> None:
    """Bar chart: CCC delta of each lag condition vs lag_0, per model × target."""
    targets = summary["target"].unique()
    models  = [m for m in summary["model"].unique() if m != "MeanBaseline"]
    lag_conds = sorted([c for c in summary["condition"].unique()
                         if c.startswith("lag_") and c != "lag_0"])
    if not lag_conds:
        return

    fig, axes = plt.subplots(len(targets), len(lag_conds),
                              figsize=(max(10, 4*len(lag_conds)), 4*len(targets)),
                              sharey="row", squeeze=False)

    for row_i, target in enumerate(targets):
        for col_i, lag_cond in enumerate(lag_conds):
            ax = axes[row_i][col_i]
            deltas, colors = [], []
            for m in models:
                sub = summary[(summary["model"]==m) & (summary["target"]==target)]
                s = sub[sub["condition"]=="lag_0"]["ccc_mean"].values
                r = sub[sub["condition"]==lag_cond]["ccc_mean"].values
                d = float(r[0] - s[0]) if len(s) and len(r) else 0
                deltas.append(d)
                colors.append("#55A868" if d > 0 else "#C44E52")
            ax.bar(range(len(models)), deltas, color=colors, alpha=0.85)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{target} | Δ vs lag_0: {lag_cond}", fontsize=9)
            ax.set_ylabel("Δ CCC")

    fig.suptitle("H₂: Lag Conditions vs Synchronous (lag_0)", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / "h2_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_secondary_analysis(summary_ext: pd.DataFrame,
                             summary_par: pd.DataFrame) -> None:
    """Compare best model CCC under external vs partner annotations for lag_0."""
    targets = ["arousal", "valence"]
    conds_available = sorted(set(summary_ext["condition"].unique()) &
                              set(summary_par["condition"].unique()))
    # Only show lag_0 and lag_1 to keep the plot readable
    conds = [c for c in ["lag_0", "lag_1", "lag_400ms_av"] if c in conds_available]
    if not conds:
        conds = conds_available[:2]

    models = [m for m in summary_ext["model"].unique() if m != "MeanBaseline"]
    fig, axes = plt.subplots(len(targets), len(conds),
                              figsize=(5*len(conds), 4*len(targets)),
                              sharey=False, squeeze=False)

    for row_i, target in enumerate(targets):
        for col_i, cond in enumerate(conds):
            ax = axes[row_i][col_i]
            ext_vals, par_vals = [], []
            for m in models:
                def _get(df):
                    r = df[(df["model"]==m) & (df["target"]==target) &
                           (df["condition"]==cond)]
                    return float(r["ccc_mean"].values[0]) if len(r) else np.nan
                ext_vals.append(_get(summary_ext))
                par_vals.append(_get(summary_par))
            x = np.arange(len(models))
            w = 0.35
            ax.bar(x - w/2, ext_vals, w, label="External obs.", color="#4C72B0", alpha=0.8)
            ax.bar(x + w/2, par_vals, w, label="Partner",       color="#DD8452", alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax.set_title(f"{target} / {cond}", fontsize=10)
            ax.set_ylabel("CCC")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8)

    fig.suptitle("External Observer vs Partner Annotations", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / "secondary_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_dyad_heatmap(results_df: "pd.DataFrame", target: str,
                      condition: str = "lag_0") -> None:
    """
    Heatmap: rows = session, columns = model, cell = CCC.
    Reveals whether 2–3 dyads are driving all the signal.
    """
    sub = results_df[(results_df["target"] == target) &
                     (results_df["condition"] == condition)]
    if sub.empty:
        return
    models_ordered = [m for m in
                      ["MeanBaseline", "RidgeCV", "SVR", "CatBoost", "Ensemble"]
                      if m in sub["model"].unique()]
    extra = [m for m in sub["model"].unique() if m not in models_ordered]
    models_ordered += extra

    pivot = sub.pivot_table(index="held_out_session", columns="model",
                            values="ccc")[models_ordered]
    n_models = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2), max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, label="CCC", fraction=0.03)

    # Annotate cells
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Sess {s}" for s in pivot.index], fontsize=9)
    ax.set_title(f"Per-Dyad CCC — {target} / {condition}", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / f"dyad_heatmap_{target}_{condition}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_annotation_agreement(agreement_dfs: dict, target: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    """
    Box plot of Pearson r between annotation perspectives.
    Shows inter-rater ceiling — model CCC cannot be expected to exceed this.
    """
    labels, data = [], []
    for name, df in agreement_dfs.items():
        if df is not None and not df.empty and "r" in df.columns:
            vals = df["r"].dropna().values
            if len(vals) > 0:
                labels.append(name)
                data.append(vals)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.8), 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="r = 0")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Inter-Perspective Agreement — {target}\n"
                 "(model CCC should not exceed this ceiling)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = FIG_DIR / f"annotation_agreement_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_incremental(summary: "pd.DataFrame", target: str) -> None:
    """
    Bar chart for incremental model comparison M1→M4.
    Shows ΔCCC at each step; the key bar is M4 − M2 (partner physio added value).
    """
    m_order = ["M1_own_label_ar", "M2_own_signal",
               "M3_label_coupling", "M4_full"]
    m_labels = ["M1: own label AR", "M2: own signal",
                 "M3: label coupling", "M4: full (own+partner)"]
    sub = summary[summary["target"] == target]
    models = [m for m in summary["model"].unique()
              if m in ("RidgeCV", "CatBoost")]
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4),
                              sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        ccc_vals, ci_lo_vals, ci_hi_vals = [], [], []
        for cond in m_order:
            row = sub[(sub["model"] == model) & (sub["condition"] == cond)]
            if len(row):
                mu   = float(row["ccc_mean"].values[0])
                ci_l = row["ccc_ci_lo"].values[0] if "ccc_ci_lo" in row.columns else mu
                ci_h = row["ccc_ci_hi"].values[0] if "ccc_ci_hi" in row.columns else mu
                ccc_vals.append(mu)
                ci_lo_vals.append(mu - ci_l)
                ci_hi_vals.append(ci_h - mu)
            else:
                ccc_vals.append(np.nan)
                ci_lo_vals.append(0)
                ci_hi_vals.append(0)
        x = np.arange(len(m_order))
        colors = ["#9EC6E0", "#4C72B0", "#F4A460", "#C0392B"]
        ax.bar(x, ccc_vals, color=colors, alpha=0.85,
               yerr=[ci_lo_vals, ci_hi_vals], capsize=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(m_labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{model} — {target}", fontsize=10)
        ax.set_ylabel("Mean CCC ± 95% CI")

    fig.suptitle(f"Incremental Model Comparison — {target}\n"
                 "M4−M2 = added value of partner multimodal features", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / f"incremental_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_ablation(ablation_summaries: dict, results_dir: Path) -> None:
    """Bar chart comparing CCC across modality subsets."""
    targets = ["arousal", "valence"]
    modalities = list(ablation_summaries.keys())
    models = [m for m in ablation_summaries[modalities[0]]["model"].unique()
               if m not in ("MeanBaseline",)]

    fig, axes = plt.subplots(1, len(targets), figsize=(12, 5))
    if len(targets) == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("Set2", len(modalities))
    for ax, target in zip(axes, targets):
        x = np.arange(len(models))
        w = 0.8 / len(modalities)
        for i, (mod_name, summ) in enumerate(ablation_summaries.items()):
            vals = []
            for m in models:
                row = summ[(summ["model"]==m) & (summ["target"]==target) &
                            (summ["condition"]=="lag_0")]
                vals.append(float(row["ccc_mean"].values[0]) if len(row) else 0)
            offset = (i - len(modalities)/2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=mod_name, color=cmap(i), alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"Modality Ablation — {target}", fontsize=11)
        ax.set_ylabel("Mean CCC (lag_0)")
        ax.legend(fontsize=8)
    fig.suptitle("Modality Ablation: Physio vs Audio vs Video vs All", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / "ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

def plot_per_session_ccc_bar(results_df: pd.DataFrame, target: str,
                              condition: str = "lag_0",
                              model: str = None) -> None:
    """
    Bar chart of per-session CCC for a given model and condition.
    Sorted by CCC to reveal session heterogeneity.
    """
    sub = results_df[(results_df["target"] == target) &
                     (results_df["condition"] == condition)]
    if sub.empty:
        return
    if model is None:
        # Pick the model with the highest mean CCC
        model = sub.groupby("model")["ccc"].mean().idxmax()
    sub_m = sub[sub["model"] == model].sort_values("ccc")
    if sub_m.empty:
        return

    sessions = sub_m["held_out_session"].values
    ccc_vals = sub_m["ccc"].values
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in ccc_vals]

    fig, ax = plt.subplots(figsize=(max(8, len(sessions) * 0.5), 4))
    ax.bar(range(len(sessions)), ccc_vals, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([f"S{s}" for s in sessions], fontsize=8)
    ax.set_ylabel("CCC")
    ax.set_title(f"Per-Session CCC — {model} | {target} | {condition}", fontsize=11)
    ax.set_xlabel("Session (sorted by CCC)")
    plt.tight_layout()
    path = FIG_DIR / f"per_session_ccc_bar_{target}_{condition}_{model}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_stat_table(stat_df: pd.DataFrame, title: str, tag: str = "") -> None:
    """
    Heatmap of Wilcoxon p-values (FDR-corrected) from a statistical comparison table.
    Rows = target, columns = comparison. Green = significant (p < 0.05).
    """
    if stat_df is None or stat_df.empty:
        return
    needed = {"target", "wilcoxon_p_fdr", "comparison"}
    if not needed.issubset(stat_df.columns):
        return

    targets = sorted(stat_df["target"].unique())
    comparisons = stat_df["comparison"].unique().tolist()

    matrix = np.full((len(targets), len(comparisons)), np.nan)
    for i, t in enumerate(targets):
        for j, c in enumerate(comparisons):
            row = stat_df[(stat_df["target"] == t) & (stat_df["comparison"] == c)]
            if len(row):
                matrix[i, j] = float(row["wilcoxon_p_fdr"].values[0])

    fig, ax = plt.subplots(figsize=(max(8, len(comparisons) * 1.5), max(3, len(targets) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.15)
    plt.colorbar(im, ax=ax, label="Wilcoxon p (FDR)", fraction=0.04)

    for i in range(len(targets)):
        for j in range(len(comparisons)):
            val = matrix[i, j]
            if np.isfinite(val):
                sig = "**" if val < 0.01 else ("*" if val < 0.05 else "ns")
                ax.text(j, i, f"{val:.3f}\n{sig}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(comparisons, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / f"stat_table{'_' + tag if tag else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_synchrony_comparison(summary_sync: pd.DataFrame,
                               summary_ext: pd.DataFrame) -> None:
    """
    Compare synchrony-only (sync_lag_0) and synchrony-augmented (sync_aug_lag_0)
    against raw partner features (lag_0), per target.
    """
    targets = ["arousal", "valence"]
    models  = [m for m in summary_ext["model"].unique() if m != "MeanBaseline"]
    cond_map = {
        "lag_0":          "Raw partner\n(M3)",
        "sync_lag_0":     "Sync only",
        "sync_aug_lag_0": "Partner + Sync\n(M4)",
        "sync_rnd":       "Random-dyad\nsync",
        "sync_circ":      "Circ-shift\nsync",
    }
    all_summary = pd.concat([summary_ext, summary_sync], ignore_index=True)

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), sharey=False)
    if len(targets) == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("Set2", len(cond_map))
    colors = {c: cmap(i) for i, c in enumerate(cond_map)}

    for ax, target in zip(axes, targets):
        sub = all_summary[all_summary["target"] == target]
        x = np.arange(len(models))
        n_conds = len(cond_map)
        w = 0.8 / max(n_conds, 1)
        for i, (cond, label) in enumerate(cond_map.items()):
            vals, yerr_lo, yerr_hi = [], [], []
            for m in models:
                row = sub[(sub["model"] == m) & (sub["condition"] == cond)]
                if len(row):
                    mu   = float(row["ccc_mean"].values[0])
                    ci_l = row["ccc_ci_lo"].values[0] if "ccc_ci_lo" in row.columns else mu
                    ci_h = row["ccc_ci_hi"].values[0] if "ccc_ci_hi" in row.columns else mu
                    vals.append(mu)
                    yerr_lo.append(mu - ci_l)
                    yerr_hi.append(ci_h - mu)
                else:
                    vals.append(0); yerr_lo.append(0); yerr_hi.append(0)
            offset = (i - n_conds / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, yerr=[yerr_lo, yerr_hi], capsize=3,
                   color=colors[cond], alpha=0.85, label=label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Synchrony vs Raw Partner Features — {target}", fontsize=11)
        ax.set_ylabel("Mean CCC ± 95% CI")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Synchrony Extension: Do Dyadic Coupling Features Help?", fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / "synchrony_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_h1_summary(h1a_df: "pd.DataFrame", h1b_df: "pd.DataFrame" = None) -> None:
    """
    Horizontal bar chart of median Δ CCC for H1a and H1b comparisons.
    Positive = real partner lag_0 outperforms the baseline.
    Color: green if FDR p < 0.05, gray otherwise.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    label_map = {
        "lag_0 vs MeanBaseline":      "vs Mean baseline",
        "lag_0 vs random_dyad":       "vs Random dyad",
        "lag_0 vs circ_shift":        "vs Circ. shift",
        "lag_0 vs missingness":       "vs Missingness",
        "lag_0 vs own_signal":        "vs Own signal",
        "lag_0 vs label_ar_own":      "vs Label AR (own)",
        "lag_0 vs label_ar_combined": "vs Label AR (combined)",
    }

    targets = list(h1a_df["target"].unique()) if h1a_df is not None and len(h1a_df) else []
    if not targets:
        return

    fig, axes = plt.subplots(1, len(targets),
                              figsize=(max(10, 6 * len(targets)), 5),
                              squeeze=False)

    for col_i, target in enumerate(targets):
        ax = axes[0][col_i]
        rows = []

        if h1a_df is not None:
            sub = h1a_df[h1a_df["target"] == target]
            for _, r in sub.iterrows():
                rows.append({
                    "label":  label_map.get(r["comparison"], r["comparison"]),
                    "delta":  float(r["median_delta"]) if "median_delta" in r.index else np.nan,
                    "p_fdr":  float(r.get("wilcoxon_p_fdr", 1.0)),
                    "family": "H1a",
                })

        n_h1a = len(rows)
        rows.append({"label": "", "delta": np.nan, "p_fdr": 1.0, "family": "sep"})

        if h1b_df is not None:
            sub = h1b_df[h1b_df["target"] == target]
            for _, r in sub.iterrows():
                rows.append({
                    "label":  label_map.get(r["comparison"], r["comparison"]),
                    "delta":  float(r["median_delta"]) if "median_delta" in r.index else np.nan,
                    "p_fdr":  float(r.get("wilcoxon_p_fdr", 1.0)),
                    "family": "H1b",
                })

        if not rows:
            continue

        labels = [r["label"] for r in rows]
        deltas = [r["delta"] if np.isfinite(r.get("delta", np.nan)) else 0.0 for r in rows]
        colors = []
        for r in rows:
            if r["family"] == "sep":
                colors.append("none")
            elif r["p_fdr"] < 0.05:
                colors.append("#27ae60")
            else:
                colors.append("#bdc3c7")

        y = np.arange(len(rows))
        ax.barh(y, deltas, color=colors, alpha=0.88, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=1.0)

        for i, r in enumerate(rows):
            if r["family"] == "sep" or np.isnan(r.get("delta", np.nan)):
                continue
            p = r["p_fdr"]
            d = r["delta"]
            marker = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            offset = 0.004 if d >= 0 else -0.004
            ha = "left" if d >= 0 else "right"
            ax.text(d + offset, i, marker, va="center", ha=ha, fontsize=8, fontweight="bold")

        # Family labels
        if n_h1a > 0:
            ax.text(ax.get_xlim()[0], -0.6, "H1a (primary)",
                    fontsize=7, color="#555555", style="italic")
        if h1b_df is not None and len(h1b_df[h1b_df["target"] == target]):
            ax.text(ax.get_xlim()[0], n_h1a + 0.6, "H1b (stricter)",
                    fontsize=7, color="#555555", style="italic")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Median Δ CCC  (lag_0 − baseline)")
        ax.set_title(f"{target}", fontsize=11)
        ax.invert_yaxis()

    patches = [
        mpatches.Patch(color="#27ae60", label="FDR p < 0.05"),
        mpatches.Patch(color="#bdc3c7", label="ns"),
    ]
    axes[0][-1].legend(handles=patches, loc="lower right", fontsize=8)

    fig.suptitle("H1: Real Partner Features vs Control Baselines\n"
                 "(positive = real partner outperforms baseline)", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "h1_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _pretty(col: str) -> str:
    return (col
        .replace("aud_F0semitoneFrom27.5Hz_sma3nz_amean", "F0 mean")
        .replace("aud_F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "F0 std")
        .replace("aud_loudness_sma3_amean", "Loudness mean")
        .replace("aud_loudness_sma3_stddevNorm", "Loudness std")
        .replace("aud_jitterLocal_sma3nz_amean", "Jitter")
        .replace("aud_shimmerLocaldB_sma3nz_amean", "Shimmer")
        .replace("aud_HNRdBACF_sma3nz_amean", "HNR")
        .replace("e4_hr_mean", "HR mean")
        .replace("e4_hr_std",  "HR std")
        .replace("e4_eda_mean", "EDA mean")
        .replace("e4_eda_std",  "EDA std")
        .replace("e4_eda_slope","EDA slope")
        .replace("e4_temp_mean","Skin temp mean")
        .replace("e4_temp_std", "Skin temp std")
        .replace("e4_temp_slope","Skin temp slope")
        .replace("e4_ibi_mean", "IBI mean")
        .replace("e4_bvp_rms",  "BVP rms")
        .replace("e4_acc_mean", "ACC mean")
        .replace("polar_hr_mean","Polar HR mean")
        .replace("attention_mean","Attention mean")
        .replace("meditation_mean","Meditation mean")
        .replace("vid_au4_mean", "AU4 mean (brow)")
        .replace("vid_au6_mean", "AU6 mean (cheek)")
        .replace("vid_au12_mean","AU12 mean (lip)")
        .replace("vid_au17_mean","AU17 mean (chin)")
        .replace("vid_yaw_mean", "Head yaw mean")
        .replace("vid_pitch_mean","Head pitch mean")
        .replace("vid_gaze_mean","Gaze mean")
        .replace("_mean", " mean").replace("_std", " std"))

def _feat_color(col: str) -> str:
    if col.startswith("aud_"):   return "#DD8452"
    if col.startswith("vid_"):   return "#55A868"
    return "#4C72B0"

def generate_all(results_df: pd.DataFrame, summary: pd.DataFrame,
                 results_dir: Path,
                 summary_partner: pd.DataFrame = None,
                 agreement_dfs: dict = None,
                 summary_incremental: pd.DataFrame = None,
                 summary_sync: pd.DataFrame = None,
                 stat_model_df: pd.DataFrame = None,
                 stat_lag_df: pd.DataFrame = None,
                 stat_h1a_df: pd.DataFrame = None,
                 stat_h1b_df: pd.DataFrame = None) -> None:
    """Generate all standard plots from a completed run."""
    global FIG_DIR
    FIG_DIR = results_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots …")

    plot_ccc_comparison(summary)
    plot_h2_delta(summary)

    for target in results_df["target"].unique():
        for cond in ["lag_0", "lag_1", "lag_400ms_av"]:
            if cond in results_df["condition"].values:
                plot_per_session_ccc(results_df, target, cond)
                plot_per_session_ccc_bar(results_df, target, cond)
        imp_path = results_dir / f"feature_importance_{target}.json"
        plot_feature_importance(imp_path, target)
        plot_dyad_heatmap(results_df, target, "lag_0")

    if summary_partner is not None:
        plot_secondary_analysis(summary, summary_partner)

    if agreement_dfs is not None:
        for target in results_df["target"].unique():
            plot_annotation_agreement(agreement_dfs, target)

    if summary_incremental is not None:
        for target in results_df["target"].unique():
            if target in summary_incremental["target"].values:
                plot_incremental(summary_incremental, target)

    if summary_sync is not None:
        plot_synchrony_comparison(summary_sync, summary)

    if stat_model_df is not None:
        plot_stat_table(stat_model_df, "Model Comparisons (Wilcoxon FDR)", "model")
    if stat_lag_df is not None:
        plot_stat_table(stat_lag_df,   "Lag Comparisons vs lag_0 (Wilcoxon FDR)", "lag")
    if stat_h1a_df is not None or stat_h1b_df is not None:
        plot_h1_summary(stat_h1a_df, stat_h1b_df)
    if stat_h1a_df is not None:
        plot_stat_table(stat_h1a_df, "H1a: Partner vs Controls (Wilcoxon FDR)", "h1a")
    if stat_h1b_df is not None:
        plot_stat_table(stat_h1b_df, "H1b: Partner vs Own-History (Wilcoxon FDR)", "h1b")

    print(f"All figures saved to {FIG_DIR}/")
