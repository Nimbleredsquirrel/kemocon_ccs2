import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import config as _config
from config import DYADS, TARGETS, DATA_ROOT, LAGS, SEGMENT_SEC
from data_loader import (load_metadata, get_start_time,
                          load_e4_all, load_neuro_all,
                          load_annotations, load_partner_annotations,
                          load_self_annotations, load_baseline_stats)
from features import build_segment_features
import audio_features as aud
import video_features as vid
from dataset import (make_pairs, make_own_signal_pairs, make_random_dyad_pairs,
                     make_circular_shift_pairs, make_label_ar_pairs,
                     make_label_delta_pairs, make_missingness_pairs,
                     make_incremental_pairs,
                     make_synchrony_pairs, make_synchrony_augmented_pairs,
                     make_random_synchrony_pairs, make_circular_shift_synchrony_pairs)
from train import run_loso, summarise
from evaluate import (annotation_agreement, compare_conditions, compare_models,
                      fdr_correction, bootstrap_ci)
from data_quality import print_quality_report, usable_dyads, nan_fraction_per_session
import plots

RESULTS_ROOT = Path(__file__).parent / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

OUT_DIR: Path = RESULTS_ROOT  # overridden in main() once args are known

def load_physio(subjects, use_baseline_norm: bool = True, win_offset: float = 0.0):
    seg_tables = {}
    for pid in sorted(subjects["pid"].unique()):
        try:
            start_ms = get_start_time(pid, subjects)
            e4 = load_e4_all(pid, start_ms)
            neuro = load_neuro_all(pid, start_ms)
            annot = load_annotations(pid)
            if annot is None or len(annot) < 5:
                continue

            bl_stats = load_baseline_stats(pid, subjects) if use_baseline_norm else None
            seg = build_segment_features(pid, e4, neuro, annot, bl_stats,
                                         win_offset=win_offset)
            seg_tables[pid] = seg
            bl_tag = f"  [BL-norm: {len(bl_stats)} signals]" if bl_stats else ""
            print(f"  P{pid}: {len(seg)} segs{bl_tag}")
        except Exception as exc:
            print(f"  P{pid}: ERROR — {exc}")
    return seg_tables

def merge_audio(seg_tables, offset: float = 0.0):
    for pid_a, pid_b in DYADS:
        if pid_a not in seg_tables and pid_b not in seg_tables:
            continue
        n_a = len(seg_tables.get(pid_a, pd.DataFrame()))
        n_b = len(seg_tables.get(pid_b, pd.DataFrame()))
        df_aud_a, df_aud_b = aud.load_or_extract(pid_a, pid_b, n_a, n_b, offset=offset)
        for pid, df_aud in [(pid_a, df_aud_a), (pid_b, df_aud_b)]:
            if pid in seg_tables:
                seg_tables[pid] = aud.merge_audio_into_seg_table(seg_tables[pid], df_aud)
    return seg_tables

def merge_video(seg_tables, offset: float = 0.0):
    pids_with_video = sorted(vid._VIDEO_MAP.keys())
    print(f"\n  Video available for {len(pids_with_video)}/32 participants")
    for pid in pids_with_video:
        if pid not in seg_tables:
            continue
        df_vid = vid.load_or_extract(pid, len(seg_tables[pid]), offset=offset)
        if df_vid is not None:
            seg_tables[pid] = vid.merge_video_into_seg_table(seg_tables[pid], df_vid)
    return seg_tables

def standardise_columns(seg_tables):
    """Ensure all tables have the same feature columns in the same order."""
    all_cols = set()
    for df in seg_tables.values():
        all_cols.update(df.columns)
    meta = {"pid", "seconds", "arousal", "valence"}
    ordered = ["pid", "seconds", "arousal", "valence"] + sorted(all_cols - meta)
    for pid in seg_tables:
        for col in sorted(all_cols - meta):
            if col not in seg_tables[pid].columns:
                seg_tables[pid][col] = np.nan
        seg_tables[pid] = seg_tables[pid][ordered]
    return seg_tables

def _swap_labels(seg_tables_ext: dict, loader_fn) -> dict:
    """
    Return seg_tables with arousal/valence replaced by another perspective's labels.
    Physiological & audio/video features remain the same.
    """
    out = {}
    for pid, df in seg_tables_ext.items():
        annot = loader_fn(pid)
        if annot is None or len(annot) < 5:
            continue
        df_new = df.copy()
        indexed = annot.set_index("seconds")
        secs = df_new["seconds"].values
        df_new["arousal"] = [float(indexed.loc[s, "arousal"])
                              if s in indexed.index else np.nan for s in secs]
        df_new["valence"] = [float(indexed.loc[s, "valence"])
                              if s in indexed.index else np.nan for s in secs]
        df_new = df_new.dropna(subset=["arousal", "valence"]).reset_index(drop=True)
        if len(df_new) >= 5:
            out[pid] = df_new
    return out

def build_partner_seg_tables(seg_tables_ext: dict, subjects: pd.DataFrame) -> dict:
    return _swap_labels(seg_tables_ext, load_partner_annotations)

def build_self_seg_tables(seg_tables_ext: dict) -> dict:
    return _swap_labels(seg_tables_ext, load_self_annotations)

def save_feature_importances(pairs, tag: str = ""):
    from models import CatBoostModel
    import json

    feat_names = None
    for p in pairs:
        if p["target"] == "arousal" and p["condition"] == "sync":
            feat_names = p["feature_names"]
            break
    if feat_names is None:
        return

    for ModelClass in [CatBoostModel]:
        m_instance = ModelClass()
        for target in TARGETS:
            sel = [p for p in pairs if p["target"]==target and p["condition"]=="sync"]
            if not sel:
                continue
            X = np.vstack([p["X"] for p in sel])
            y = np.concatenate([p["y"] for p in sel])
            try:
                m_instance.fit(X, y)
            except Exception:
                continue
            if m_instance.feature_importances_ is not None:
                imp = dict(zip(feat_names, m_instance.feature_importances_.tolist()))
                imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20])
                suffix = f"_{m_instance.name.lower()}{tag}"
                path = OUT_DIR / f"feature_importance_{target}{suffix}.json"
                with open(path, "w") as f:
                    json.dump(imp_sorted, f, indent=2)
                top5 = list(imp_sorted.keys())[:5]
                print(f"  {m_instance.name} top-5 ({target}): {', '.join(top5)}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="K-EmoCon cross-person emotion prediction pipeline")
    parser.add_argument("--no-video",      action="store_true",
                        help="Skip video features")
    parser.add_argument("--no-audio",      action="store_true",
                        help="Skip audio features")
    parser.add_argument("--physio-only",   action="store_true",
                        help="Use physiological features only")
    parser.add_argument("--no-optuna",     action="store_true",
                        help="Skip CatBoostOptuna (much faster)")
    parser.add_argument("--no-lstm",       action="store_true",
                        help="Skip LSTMModel")
    parser.add_argument("--ablation",      action="store_true",
                        help="Run modality ablation (physio/audio/video/all)")
    parser.add_argument("--no-controls",   action="store_true",
                        help="Skip negative-control baselines (own-signal, random-dyad, circular-shift)")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Skip incremental model comparison (M1→M4)")
    parser.add_argument("--no-delta",      action="store_true",
                        help="Skip label-delta (ΔA_label) target analysis")
    parser.add_argument("--no-agreement",  action="store_true",
                        help="Skip annotation agreement analysis")
    parser.add_argument("--no-synchrony",  action="store_true",
                        help="Skip synchrony feature analysis")
    return parser.parse_args()

def make_run_dir(args) -> Path:
    """
    Create and return a timestamped, descriptive subdirectory under results/.

    Naming convention:
      run_YYYYMMDD_HHMM_<modalities>_<optuna>_<controls>

    Examples:
      run_20250527_1430_physio+audio+video_optuna_controls
      run_20250527_1530_physio-only_no-optuna_no-controls
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Modalities tag
    if args.physio_only:
        mods = "physio-only"
    else:
        parts = ["physio"]
        if not args.no_audio:
            parts.append("audio")
        if not args.no_video:
            parts.append("video")
        mods = "+".join(parts)

    opts = "no-optuna" if args.no_optuna else "optuna"
    ctrl = "no-controls" if args.no_controls else "controls"

    name = f"run_{ts}_{mods}_{opts}_{ctrl}"
    out  = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    # figures/ lives inside the run dir, not the global results/
    (out / "figures").mkdir(exist_ok=True)
    return out

def smoke_test() -> bool:
    """Verify required data directories exist before training."""
    from config import DATA_ROOT, E4_DIR, ANNOT_DIR, METADATA_DIR
    required = {
        "E4 data":          E4_DIR,
        "Annotations":      ANNOT_DIR,
        "Metadata":         METADATA_DIR,
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        print("\nERROR: Missing required data directories:")
        for name, path in missing.items():
            print(f"  {name}: {path}")
        print("\nRun:  python3 setup_data.py --kemocon-root /path/to/kemocon_extracted")
        return False
    return True

def save_config(out_dir: Path) -> None:
    """Save a snapshot of config values alongside results."""
    snap = {}
    for k in dir(_config):
        if k.startswith("_"):
            continue
        v = getattr(_config, k)
        if isinstance(v, (int, float, str, list, bool)):
            snap[k] = v
        elif hasattr(v, "__fspath__"):   # Path
            snap[k] = str(v)
    with open(out_dir / "config_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2, default=str)
    print(f"  Config saved → {out_dir}/config_snapshot.json")

def main():
    args = parse_args()

    print("K-EmoCon: Cross-Person Multimodal Emotion Prediction")
    print("=" * 55)

    if not smoke_test():
        return

    # Create run-specific output directory and redirect plots there
    global OUT_DIR
    OUT_DIR = make_run_dir(args)
    plots.FIG_DIR = OUT_DIR / "figures"
    plots.FIG_DIR.mkdir(exist_ok=True)
    print(f"\nRun output → {OUT_DIR.name}/")

    subjects, _ = load_metadata()
    print(f"Participants: {len(subjects)}")

    use_audio = not (args.no_audio or args.physio_only)
    use_video = not (args.no_video or args.physio_only)
    print(f"Modalities: physio=ON  audio={'ON' if use_audio else 'OFF'}  "
          f"video={'ON' if use_video else 'OFF'}")

    # Optionally disable slow models
    if args.no_optuna:
        import models as _models
        _orig = _models.all_models
        _models.all_models = lambda: [m for m in _orig()
                                       if m.name != "CatBoostOptuna"]
        print("  [--no-optuna] CatBoostOptuna disabled")

    if args.no_lstm:
        import models as _models_lstm
        _orig_lstm = _models_lstm.all_models
        _models_lstm.all_models = lambda: [m for m in _orig_lstm()
                                            if m.name != "LSTM"]
        print("  [--no-lstm] LSTMModel disabled")

    # ── 1. Load features ──────────────────────────────────────────────────
    print("\n[1/5] Loading physio + baseline normalisation …")
    seg_tables = load_physio(subjects, use_baseline_norm=True, win_offset=0.0)
    print(f"Loaded: {len(seg_tables)} participants")

    if use_audio:
        print("\n[2/5] Merging audio features (eGeMAPS) …")
        seg_tables = merge_audio(seg_tables, offset=0.0)
    else:
        print("\n[2/5] Audio skipped.")

    if use_video:
        print("\n[3/5] Merging video features …")
        seg_tables = merge_video(seg_tables, offset=0.0)
    else:
        print("\n[3/5] Video skipped.")

    seg_tables = standardise_columns(seg_tables)

    # 400ms-offset tables: physio at 0ms (400ms meaningless at 5s resolution),
    # audio/video re-extracted at 400ms offset.
    print("\n[1b] Building 400ms-offset tables (audio/video only; physio unchanged) …")
    seg_tables_400ms = {pid: df.copy() for pid, df in seg_tables.items()}
    if use_audio or use_video:
        # Strip old audio/video from copies; re-add with 400ms offset
        for pid in seg_tables_400ms:
            av_cols = [c for c in seg_tables_400ms[pid].columns
                       if c.startswith(("aud_", "vid_"))]
            seg_tables_400ms[pid] = seg_tables_400ms[pid].drop(columns=av_cols)
        if use_audio:
            seg_tables_400ms = merge_audio(seg_tables_400ms, offset=0.4)
        if use_video:
            seg_tables_400ms = merge_video(seg_tables_400ms, offset=0.4)
        seg_tables_400ms = standardise_columns(seg_tables_400ms)
    print(f"  400ms tables: {len(seg_tables_400ms)} participants")

    # Feature summary
    sample = next(iter(seg_tables.values()))
    feat_cols = [c for c in sample.columns
                 if c not in {"pid", "seconds", "arousal", "valence"}]
    n_p = sum(1 for c in feat_cols if c.startswith(("e4_","bw_","attention","meditation","polar")))
    n_a = sum(1 for c in feat_cols if c.startswith("aud_"))
    n_v = sum(1 for c in feat_cols if c.startswith("vid_"))
    print(f"\nFeature set: {len(feat_cols)} total  ({n_p} physio | {n_a} audio | {n_v} video)")

    # ── 2. Quality & usability ─────────────────────────────────────────────
    print("\n[4/5] Data quality …")
    print_quality_report(seg_tables, DYADS)
    nan_df = nan_fraction_per_session(seg_tables)
    good_dyads = usable_dyads(nan_df, DYADS, max_nan=0.8)
    print(f"Usable dyads: {len(good_dyads)} / {len(DYADS)}")

    if not good_dyads:
        print("No usable dyads — check DATA_ROOT in config.py"); return

    # ── 3. Primary LOSO (cross-person + 400ms + negative controls) ─────────
    print("\n[5/5] Cross-person prediction — PRIMARY (external observer) …")
    pairs_ext = make_pairs(seg_tables, good_dyads)
    pairs_ext += make_pairs(seg_tables_400ms, good_dyads,
                             conditions=[(0, "lag_400ms_av")])

    if not args.no_controls:
        print("  Adding negative-control pairs …")
        pairs_ext += make_own_signal_pairs(seg_tables, good_dyads,
                                            conditions=[(0, "own_signal")])
        pairs_ext += make_random_dyad_pairs(seg_tables, good_dyads,
                                             conditions=[(0, "random_dyad")])
        pairs_ext += make_circular_shift_pairs(seg_tables, good_dyads,
                                                conditions=[(0, "circ_shift")])
        # Label autoregression — most important missing baseline
        print("  Adding label autoregression baselines …")
        pairs_ext += make_label_ar_pairs(seg_tables, good_dyads, mode="own")
        pairs_ext += make_label_ar_pairs(seg_tables, good_dyads, mode="partner")
        pairs_ext += make_label_ar_pairs(seg_tables, good_dyads, mode="combined")
        # Missingness red-flag baseline
        pairs_ext += make_missingness_pairs(seg_tables, good_dyads)

    _print_pair_summary(pairs_ext)

    print("\nRunning LOSO CV (primary) …")
    results_ext = run_loso(pairs_ext)
    summary_ext = summarise(results_ext)

    print("\n=== Primary Results (External Observer) ===")
    print(summary_ext.to_string(index=False))
    _print_hypothesis_tests(summary_ext)

    # Permutation tests: real partner (lag_0) vs controls
    print("\n--- Permutation Tests (lag_0 vs baselines) ---")
    _run_permutation_tests(results_ext, summary_ext)

    results_ext.to_csv(OUT_DIR / "loso_results_primary.csv", index=False)
    summary_ext.to_csv(OUT_DIR / "loso_summary_primary.csv", index=False)
    save_config(OUT_DIR)
    print(f"\nPrimary results saved → {OUT_DIR}/")

    print("\nFeature importances …")
    save_feature_importances(pairs_ext)

    # ── 4. Secondary: partner + self-report annotations ────────────────────
    print("\n--- Secondary Analysis: All Annotation Perspectives ---")
    seg_tables_par = build_partner_seg_tables(seg_tables, subjects)
    seg_tables_self = build_self_seg_tables(seg_tables)
    print(f"  Partner annotations: {len(seg_tables_par)} participants")
    print(f"  Self-report:         {len(seg_tables_self)} participants")

    summary_par = _run_annotation_perspective(seg_tables_par, good_dyads, "partner", OUT_DIR)
    summary_self = _run_annotation_perspective(seg_tables_self, good_dyads, "self",     OUT_DIR)

    # ── 5. Annotation agreement (reliability ceiling) ──────────────────────
    agreement_dfs = None
    if not args.no_agreement:
        print("\n--- Annotation Agreement (inter-perspective reliability ceiling) ---")
        agreement_dfs = {}
        for target in TARGETS:
            agreement_dfs[target] = {}
            for pair_name, (st_a, st_b) in [
                ("self_vs_ext",     (seg_tables_self, seg_tables)),
                ("partner_vs_ext",  (seg_tables_par,  seg_tables)),
                ("self_vs_partner", (seg_tables_self, seg_tables_par)),
            ]:
                ag = annotation_agreement(st_a, st_b, good_dyads, target)
                agreement_dfs[target][pair_name] = ag
                if not ag.empty:
                    print(f"  {target:8s} {pair_name:20s}: "
                          f"median r = {ag['r'].median():.3f}  "
                          f"(n={len(ag)} participants)")
        print("\n  NOTE: Model CCC ceiling ≈ self/external agreement (above).")

    # ── 6. Label delta targets ─────────────────────────────────────────────
    summary_delta = None
    if not args.no_delta:
        print("\n--- Label Delta Analysis (predict ΔA_label[t]) ---")
        pairs_delta = make_label_delta_pairs(seg_tables, good_dyads)
        if pairs_delta:
            results_delta = run_loso(pairs_delta)
            summary_delta = summarise(results_delta)
            if not summary_delta.empty:
                print("=== Delta Results ===")
                print(summary_delta.to_string(index=False))
                results_delta.to_csv(OUT_DIR / "loso_results_delta.csv", index=False)
                summary_delta.to_csv(OUT_DIR / "loso_summary_delta.csv", index=False)
            else:
                print("  [SKIP] delta run produced no results")
        else:
            print("  [SKIP] no delta pairs built")

    # ── 7. Incremental model comparison (M1→M4) ────────────────────────────
    summary_incr = None
    if not args.no_incremental:
        print("\n--- Incremental Model Comparison (M1→M4) ---")
        print("  Key: ΔCCC(M4 − M2) = added value of partner multimodal features")
        pairs_incr = make_incremental_pairs(seg_tables, good_dyads)
        if pairs_incr:
            # Fast-only: skip Optuna for incremental
            import models as _mods
            _orig_all = _mods.all_models
            _mods.all_models = lambda: [m for m in _orig_all()
                                         if m.name in ("RidgeCV", "CatBoost",
                                                        "MeanBaseline")]
            results_incr = run_loso(pairs_incr)
            _mods.all_models = _orig_all
            summary_incr = summarise(results_incr)
            print("\n=== Incremental Results ===")
            print(summary_incr.to_string(index=False))
            _print_incremental_delta(summary_incr)
            results_incr.to_csv(OUT_DIR / "loso_results_incremental.csv", index=False)
            summary_incr.to_csv(OUT_DIR / "loso_summary_incremental.csv", index=False)
        else:
            print("  [SKIP] no incremental pairs built")

    # ── 8. Synchrony feature analysis ─────────────────────────────────────
    summary_sync = None
    if not args.no_synchrony:
        print("\n--- Synchrony Feature Analysis ---")
        print("  Dyadic coupling features: |A-B|, z-score dist, product, cos-sim, rolling corr")
        summary_sync = _run_synchrony_analysis(seg_tables, good_dyads, OUT_DIR)

    # ── 8b. Modality ablation ──────────────────────────────────────────────
    if args.ablation:
        print("\n--- Modality Ablation ---")
        _run_ablation(subjects, good_dyads, OUT_DIR)

    # ── 9. Plots ───────────────────────────────────────────────────────────
    ag_for_plots = None
    if agreement_dfs:
        ag_for_plots = agreement_dfs.get(TARGETS[0], None)

    stat_model_df, stat_lag_df, stat_h1a_df, stat_h1b_df = _build_stat_tables(results_ext, OUT_DIR)

    plots.generate_all(results_ext, summary_ext, OUT_DIR, summary_par,
                       agreement_dfs=ag_for_plots,
                       summary_incremental=summary_incr,
                       summary_sync=summary_sync,
                       stat_model_df=stat_model_df,
                       stat_lag_df=stat_lag_df,
                       stat_h1a_df=stat_h1a_df,
                       stat_h1b_df=stat_h1b_df)
    if summary_par is not None:
        plots.plot_ccc_comparison(summary_par, tag="partner")
    if summary_self is not None:
        plots.plot_ccc_comparison(summary_self, tag="self")

    if agreement_dfs:
        for target in TARGETS:
            plots.plot_annotation_agreement(agreement_dfs[target], target)

    print(f"\nDone. All outputs in {OUT_DIR}/")

def _run_annotation_perspective(seg_tables_persp, good_dyads, tag, out_dir):
    """Run LOSO on a given annotation perspective (lag_0 only for speed)."""
    pairs = make_pairs(seg_tables_persp, good_dyads, conditions=[(0, "lag_0")])
    if not pairs:
        print(f"  [SKIP] {tag}: no pairs built")
        return None
    print(f"\nRunning LOSO CV ({tag} annotations) …")
    results = run_loso(pairs)
    summary = summarise(results)
    print(f"=== {tag.title()} Annotation Results ===")
    # Print only CCC columns to keep output readable
    cols = ["model", "target", "condition", "ccc_mean", "ccc_std",
            "ccc_ci_lo", "ccc_ci_hi"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    results.to_csv(out_dir / f"loso_results_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"loso_summary_{tag}.csv", index=False)
    return summary

def _pick_ref_model(results_df) -> str:
    """Select the best available reference model for statistical tests."""
    for candidate in ["CatBoostOptuna", "CatBoost", "RidgeCV"]:
        if candidate in results_df["model"].values:
            return candidate
    return results_df["model"].unique()[0]


def _run_permutation_tests(results_df, summary_df):
    """
    Comprehensive statistical tests:
      1. Control comparisons: ref_model lag_0 vs each baseline condition
      2. Lag comparisons:     ref_model lag_k vs lag_0 (FDR-corrected)
      3. Model comparisons:   ref_model vs RidgeCV on lag_0 (FDR-corrected)

    Prints compact tables for each family; saves permutation_tests.csv.
    """
    ref_model = _pick_ref_model(results_df)
    print(f"  Reference model for tests: {ref_model}")

    all_rows = []

    # ── H1a: primary cross-person signal test ────────────────────────────
    h1a_rows = []
    if "MeanBaseline" in results_df["model"].values and ref_model != "MeanBaseline":
        for target in TARGETS:
            row = compare_models(results_df, target, "lag_0", ref_model, "MeanBaseline")
            row["comparison"] = "lag_0 vs MeanBaseline"
            row["cond_a"] = "lag_0"
            row["cond_b"] = "MeanBaseline"
            row["family"] = "h1a"
            h1a_rows.append(row)
    for target in TARGETS:
        for baseline in ["random_dyad", "circ_shift", "missingness"]:
            if baseline not in results_df["condition"].values:
                continue
            row = compare_conditions(results_df, ref_model, target, "lag_0", baseline)
            row["family"] = "h1a"
            row["comparison"] = f"lag_0 vs {baseline}"
            h1a_rows.append(row)
    if h1a_rows:
        h1a_df = pd.DataFrame(h1a_rows)
        ps = fdr_correction(h1a_df["wilcoxon_p"].fillna(1.0).tolist())
        h1a_df["wilcoxon_p_fdr"] = ps
        print("\n  H1a) Partner signal vs primary controls (one-sided, FDR-corrected):")
        print(h1a_df[["target", "comparison", "median_delta",
                       "wilcoxon_p", "wilcoxon_p_fdr"]].to_string(index=False))
        all_rows.append(h1a_df)

    # ── H1b: stricter added-value tests ──────────────────────────────────
    h1b_rows = []
    for target in TARGETS:
        for baseline in ["own_signal", "label_ar_own", "label_ar_combined"]:
            if baseline not in results_df["condition"].values:
                continue
            row = compare_conditions(results_df, ref_model, target, "lag_0", baseline)
            row["family"] = "h1b"
            row["comparison"] = f"lag_0 vs {baseline}"
            h1b_rows.append(row)
    if h1b_rows:
        h1b_df = pd.DataFrame(h1b_rows)
        ps = fdr_correction(h1b_df["wilcoxon_p"].fillna(1.0).tolist())
        h1b_df["wilcoxon_p_fdr"] = ps
        print("\n  H1b) Partner vs own-history baselines [stricter, secondary]:")
        print(h1b_df[["target", "comparison", "median_delta",
                       "wilcoxon_p", "wilcoxon_p_fdr"]].to_string(index=False))
        all_rows.append(h1b_df)

    # ── Family B: lag comparisons ─────────────────────────────────────────
    lag_conds = [c for c in ["lag_400ms_av", "lag_1", "lag_2", "lag_3", "lag_4"]
                 if c in results_df["condition"].values]
    lag_rows = []
    for target in TARGETS:
        for lag_cond in lag_conds:
            row = compare_conditions(results_df, ref_model, target, lag_cond, "lag_0")
            row["family"] = "lag"
            row["comparison"] = f"{lag_cond} vs lag_0"
            lag_rows.append(row)
    if lag_rows:
        lag_df = pd.DataFrame(lag_rows)
        ps = fdr_correction(lag_df["wilcoxon_p"].fillna(1.0).tolist())
        lag_df["wilcoxon_p_fdr"] = ps
        ps_sf = fdr_correction(lag_df["signflip_p"].fillna(1.0).tolist())
        lag_df["signflip_p_fdr"] = ps_sf
        print(f"\n  B) Lag comparisons ({ref_model}, FDR-corrected):")
        print(lag_df[["target", "cond_a", "median_delta",
                       "wilcoxon_p", "wilcoxon_p_fdr"]].to_string(index=False))
        all_rows.append(lag_df)

    # ── Family C: model comparisons ───────────────────────────────────────
    comparison_models = [m for m in ["CatBoostOptuna", "CatBoost", "Ensemble", "RidgeCV"]
                         if m in results_df["model"].values and m != ref_model]
    model_rows = []
    for target in TARGETS:
        for other_model in comparison_models:
            row = compare_models(results_df, target, "lag_0", ref_model, other_model)
            row["family"] = "model"
            model_rows.append(row)
    if model_rows:
        model_df = pd.DataFrame(model_rows)
        ps = fdr_correction(model_df["wilcoxon_p"].fillna(1.0).tolist())
        model_df["wilcoxon_p_fdr"] = ps
        ps_sf = fdr_correction(model_df["signflip_p"].fillna(1.0).tolist())
        model_df["signflip_p_fdr"] = ps_sf
        print(f"\n  C) Model comparisons (lag_0, {ref_model} vs others, FDR-corrected):")
        print(model_df[["target", "comparison", "median_delta",
                         "wilcoxon_p", "wilcoxon_p_fdr",
                         "boot_ci_lo", "boot_ci_hi"]].to_string(index=False))
        all_rows.append(model_df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(OUT_DIR / "permutation_tests.csv", index=False)
        print("\n  (One-sided tests: lag_0 > baseline; H1a = primary, H1b = stricter reference; FDR via Benjamini-Hochberg)")
        print(f"  Saved: permutation_tests.csv")


def _build_stat_tables(results_df, out_dir) -> tuple:
    """Build separate stat tables for model/lag/control comparisons for plotting."""
    ref_model = _pick_ref_model(results_df)

    # Model comparison table
    model_rows = []
    comp_models = [m for m in ["CatBoostOptuna", "CatBoost", "Ensemble", "SVR", "RidgeCV"]
                   if m in results_df["model"].values and m != ref_model]
    for target in TARGETS:
        for other in comp_models:
            row = compare_models(results_df, target, "lag_0", ref_model, other)
            row["family"] = "model"
            model_rows.append(row)
    stat_model = None
    if model_rows:
        stat_model = pd.DataFrame(model_rows)
        ps = fdr_correction(stat_model["wilcoxon_p"].fillna(1.0).tolist())
        stat_model["wilcoxon_p_fdr"] = ps
        stat_model.to_csv(out_dir / "stat_model_comparisons.csv", index=False)

    # Lag comparison table
    lag_conds = [c for c in ["lag_400ms_av", "lag_1", "lag_2", "lag_3", "lag_4"]
                 if c in results_df["condition"].values]
    lag_rows = []
    for target in TARGETS:
        for lag_cond in lag_conds:
            row = compare_conditions(results_df, ref_model, target, lag_cond, "lag_0")
            row["comparison"] = f"{lag_cond} vs lag_0"
            row["family"] = "lag"
            lag_rows.append(row)
    stat_lag = None
    if lag_rows:
        stat_lag = pd.DataFrame(lag_rows)
        ps = fdr_correction(stat_lag["wilcoxon_p"].fillna(1.0).tolist())
        stat_lag["wilcoxon_p_fdr"] = ps
        stat_lag.to_csv(out_dir / "stat_lag_comparisons.csv", index=False)

    # H1a comparison table
    h1a_rows = []
    if "MeanBaseline" in results_df["model"].values and ref_model != "MeanBaseline":
        for target in TARGETS:
            row = compare_models(results_df, target, "lag_0", ref_model, "MeanBaseline")
            row["comparison"] = "lag_0 vs MeanBaseline"
            row["cond_a"] = "lag_0"
            row["cond_b"] = "MeanBaseline"
            row["family"] = "h1a"
            h1a_rows.append(row)
    h1a_baselines = [b for b in ["random_dyad", "circ_shift", "missingness"]
                     if b in results_df["condition"].values]
    for target in TARGETS:
        for baseline in h1a_baselines:
            row = compare_conditions(results_df, ref_model, target, "lag_0", baseline)
            row["comparison"] = f"lag_0 vs {baseline}"
            row["family"] = "h1a"
            h1a_rows.append(row)
    stat_h1a = None
    if h1a_rows:
        stat_h1a = pd.DataFrame(h1a_rows)
        stat_h1a["wilcoxon_p_fdr"] = fdr_correction(stat_h1a["wilcoxon_p"].fillna(1.0).tolist())
        stat_h1a.to_csv(out_dir / "stat_h1a_comparisons.csv", index=False)

    # H1b comparison table
    h1b_baselines = [b for b in ["own_signal", "label_ar_own", "label_ar_combined"]
                     if b in results_df["condition"].values]
    h1b_rows = []
    for target in TARGETS:
        for baseline in h1b_baselines:
            row = compare_conditions(results_df, ref_model, target, "lag_0", baseline)
            row["comparison"] = f"lag_0 vs {baseline}"
            row["family"] = "h1b"
            h1b_rows.append(row)
    stat_h1b = None
    if h1b_rows:
        stat_h1b = pd.DataFrame(h1b_rows)
        stat_h1b["wilcoxon_p_fdr"] = fdr_correction(stat_h1b["wilcoxon_p"].fillna(1.0).tolist())
        stat_h1b.to_csv(out_dir / "stat_h1b_comparisons.csv", index=False)

    return stat_model, stat_lag, stat_h1a, stat_h1b

def _print_incremental_delta(summary):
    """Print ΔCCC table for M1→M4."""
    print("\n  Model      Target    Δ(M2−M1) own signal  Δ(M3−M1) partner labels  "
          "Δ(M4−M2) partner features")
    for target in TARGETS:
        sub = summary[summary["target"] == target]
        for model in ["RidgeCV", "CatBoost"]:
            m_sub = sub[sub["model"] == model]
            def _ccc(cond):
                r = m_sub[m_sub["condition"] == cond]["ccc_mean"].values
                return float(r[0]) if len(r) else np.nan
            m1 = _ccc("M1_own_label_ar")
            m2 = _ccc("M2_own_signal")
            m3 = _ccc("M3_label_coupling")
            m4 = _ccc("M4_full")
            print(f"  {model:10s} {target:8s}  "
                  f"Δ(M2−M1)={m2-m1:+.4f}  "
                  f"Δ(M3−M1)={m3-m1:+.4f}  "
                  f"Δ(M4−M2)={m4-m2:+.4f}  "
                  f"[M4={m4:.4f}]")

def _run_synchrony_analysis(seg_tables, good_dyads, out_dir) -> pd.DataFrame:
    """
    Synchrony extension: test whether dyadic coupling features predict affect
    beyond raw partner features.

    Conditions produced:
      sync_lag_0 / sync_lag_1   — pure synchrony features
      sync_aug_lag_0            — raw partner + synchrony (M4)
      sync_rnd                  — random-dyad synchrony control
      sync_circ                 — circular-shift synchrony control

    Key test: sync_aug_lag_0 (M4) vs lag_0 (M3, from primary analysis).
    """
    import models as _mods
    from config import SYNCHRONY_WINDOWS

    # Fast models only: synchrony features are high-dimensional
    _orig_all = _mods.all_models
    _mods.all_models = lambda: [m for m in _orig_all()
                                 if m.name in ("MeanBaseline", "RidgeCV", "CatBoost")]

    pairs_sync = make_synchrony_pairs(seg_tables, good_dyads,
                                       windows=SYNCHRONY_WINDOWS, lags=(0, 1))
    pairs_sync_aug = make_synchrony_augmented_pairs(seg_tables, good_dyads,
                                                     windows=SYNCHRONY_WINDOWS, lags=(0,))
    pairs_sync_rnd = make_random_synchrony_pairs(seg_tables, good_dyads,
                                                  windows=SYNCHRONY_WINDOWS)
    pairs_sync_circ = make_circular_shift_synchrony_pairs(seg_tables, good_dyads,
                                                            windows=SYNCHRONY_WINDOWS)
    all_sync_pairs = pairs_sync + pairs_sync_aug + pairs_sync_rnd + pairs_sync_circ

    if not all_sync_pairs:
        print("  [SKIP] no synchrony pairs built")
        _mods.all_models = _orig_all
        return None

    _print_pair_summary(all_sync_pairs)
    print("\nRunning LOSO CV (synchrony) …")
    results_sync = run_loso(all_sync_pairs)
    summary_sync = summarise(results_sync)

    print("\n=== Synchrony Results ===")
    print(summary_sync.to_string(index=False))

    # Key test: sync_aug_lag_0 vs lag_0 (from primary results — not available here)
    # Print what we have: sync vs sync controls
    print("\n  Synchrony control tests (sync_lag_0 vs sync_rnd, sync_circ):")
    ref_model = _pick_ref_model(results_sync)
    sync_rows = []
    for target in TARGETS:
        for ctrl in ["sync_rnd", "sync_circ"]:
            if ctrl in results_sync["condition"].values:
                row = compare_conditions(results_sync, ref_model, target, "sync_lag_0", ctrl)
                row["comparison"] = f"sync_lag_0 vs {ctrl}"
                sync_rows.append(row)
    if sync_rows:
        sync_stat = pd.DataFrame(sync_rows)
        ps = fdr_correction(sync_stat["wilcoxon_p"].fillna(1.0).tolist())
        sync_stat["wilcoxon_p_fdr"] = ps
        print(sync_stat[["target", "comparison", "median_delta",
                          "wilcoxon_p", "wilcoxon_p_fdr"]].to_string(index=False))
        sync_stat.to_csv(out_dir / "stat_synchrony_comparisons.csv", index=False)

    results_sync.to_csv(out_dir / "loso_results_synchrony.csv", index=False)
    summary_sync.to_csv(out_dir / "loso_summary_synchrony.csv", index=False)
    print(f"  Synchrony results saved → {out_dir}/")

    _mods.all_models = _orig_all
    return summary_sync


def _run_ablation(subjects, good_dyads, out_dir):
    """Run LOSO for each modality subset and save results."""
    from models import all_models as _all_models
    import models as _mods

    # Fast ablation: use only RidgeCV + CatBoost (skip Optuna)
    _orig_all = _mods.all_models
    _mods.all_models = lambda: [m for m in _orig_all()
                                 if m.name in ("RidgeCV", "CatBoost", "MeanBaseline")]

    modalities = {
        "physio":     (False, False),   # (use_audio, use_video)
        "audio":      (True,  False),
        "video":      (False, True),
        "all":        (True,  True),
    }

    ablation_summaries = {}
    for name, (ua, uv) in modalities.items():
        print(f"\n  Ablation: {name} …")
        st = load_physio(subjects, use_baseline_norm=True)
        if ua:
            st = merge_audio(st)
        if uv:
            st = merge_video(st)
        st = standardise_columns(st)
        pairs = make_pairs(st, good_dyads, conditions=[(0, "lag_0")])
        if not pairs:
            continue
        res = run_loso(pairs)
        summ = summarise(res)
        summ["modality"] = name
        ablation_summaries[name] = summ
        res.to_csv(out_dir / f"ablation_{name}_results.csv", index=False)
        summ.to_csv(out_dir / f"ablation_{name}_summary.csv", index=False)

    _mods.all_models = _orig_all

    if ablation_summaries:
        combined = pd.concat(ablation_summaries.values(), ignore_index=True)
        print("\n=== Modality Ablation Summary (lag_0, CCC mean) ===")
        pivot = combined.pivot_table(
            index=["modality", "target"], columns="model",
            values="ccc_mean", aggfunc="first"
        )
        print(pivot.to_string())
        plots.plot_ablation(ablation_summaries, out_dir)

def _print_pair_summary(pairs):
    conds = sorted({p["condition"] for p in pairs})
    for t in TARGETS:
        for c in conds:
            n = sum(1 for p in pairs if p["target"]==t and p["condition"]==c)
            s = sum(len(p["y"]) for p in pairs if p["target"]==t and p["condition"]==c)
            print(f"  {t:8s} {c:15s}: {n:3d} pairs, {s:5d} segments")

def _print_hypothesis_tests(summary):
    print("\n--- H₁a (primary): partner lag_0 vs control baselines ---")
    for t in TARGETS:
        sub = summary[summary["target"] == t]
        lag0 = sub[sub["condition"] == "lag_0"].sort_values("ccc_mean", ascending=False)
        if lag0.empty:
            continue
        best = lag0.iloc[0]
        lag0_ccc = float(best["ccc_mean"])
        mb_row  = sub[(sub["condition"] == "lag_0") & (sub["model"] == "MeanBaseline")]["ccc_mean"]
        rd_row  = sub[sub["condition"] == "random_dyad"].sort_values("ccc_mean", ascending=False)
        cs_row  = sub[sub["condition"] == "circ_shift"].sort_values("ccc_mean", ascending=False)
        mb_ccc  = float(mb_row.values[0]) if len(mb_row) else np.nan
        rd_ccc  = float(rd_row.iloc[0]["ccc_mean"]) if len(rd_row) else np.nan
        cs_ccc  = float(cs_row.iloc[0]["ccc_mean"]) if len(cs_row) else np.nan
        excl_mb = "" if np.isnan(mb_ccc) else f"  Δ_vs_mean={lag0_ccc - mb_ccc:+.4f}"
        excl_rd = "" if np.isnan(rd_ccc) else f"  Δ_vs_rnd={lag0_ccc - rd_ccc:+.4f}"
        excl_cs = "" if np.isnan(cs_ccc) else f"  Δ_vs_circ={lag0_ccc - cs_ccc:+.4f}"
        print(f"  {t:8s}  best={best['model']:20s}  CCC={lag0_ccc:.4f}"
              f"{excl_mb}{excl_rd}{excl_cs}")

    print("\n--- H₁b (stricter, secondary): partner lag_0 vs own-history baselines ---")
    for t in TARGETS:
        sub = summary[summary["target"] == t]
        lag0 = sub[sub["condition"] == "lag_0"].sort_values("ccc_mean", ascending=False)
        if lag0.empty:
            continue
        m = lag0.iloc[0]["model"]
        lag0_ccc = float(lag0.iloc[0]["ccc_mean"])
        ar_row  = sub[(sub["condition"] == "label_ar_own") & (sub["model"] == m)]["ccc_mean"]
        ow_row  = sub[(sub["condition"] == "own_signal")   & (sub["model"] == m)]["ccc_mean"]
        ar_ccc  = float(ar_row.values[0])  if len(ar_row)  else np.nan
        ow_ccc  = float(ow_row.values[0])  if len(ow_row)  else np.nan
        ar_str  = f"  label_ar_own={ar_ccc:.4f}  Δ={lag0_ccc - ar_ccc:+.4f}" if not np.isnan(ar_ccc) else ""
        ow_str  = f"  own_signal={ow_ccc:.4f}  Δ={lag0_ccc - ow_ccc:+.4f}" if not np.isnan(ow_ccc) else ""
        print(f"  {t:8s}  lag_0({m})={lag0_ccc:.4f}{ar_str}{ow_str}")

    print("\n--- H₂: best lag vs synchronous (lag_0)? ---")
    for t in TARGETS:
        print(f"\n  {t}:")
        for m in summary["model"].unique():
            sub = summary[(summary["model"] == m) & (summary["target"] == t)]
            lag0 = sub[sub["condition"] == "lag_0"]["ccc_mean"].values
            if not len(lag0):
                continue
            best_lag = sub.sort_values("ccc_mean", ascending=False).iloc[0]
            d = float(best_lag["ccc_mean"] - lag0[0])
            print(f"    {m:20s}  sync={lag0[0]:.4f}  "
                  f"best={best_lag['ccc_mean']:.4f} @ {best_lag['condition']}  Δ={d:+.4f}  "
                  f"{'✓' if d > 0 else '✗'}")

    print("\n--- Best lag per target (averaged across models) ---")
    for t in TARGETS:
        sub = summary[summary["target"] == t]
        lag_means = sub.groupby("condition")["ccc_mean"].mean().sort_values(ascending=False)
        print(f"  {t}:")
        for cond, val in lag_means.items():
            print(f"    {cond}: CCC={val:.4f}")

if __name__ == "__main__":
    main()
