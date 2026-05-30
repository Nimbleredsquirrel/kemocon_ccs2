# K-EmoCon: Cross-Person Multimodal Emotion Prediction Baseline

Cross-person emotion prediction on the [K-EmoCon dataset](https://www.nature.com/articles/s41597-020-00630-y) (Park et al., 2020, *Scientific Data*).

**Core research question:** can one participant's multimodal signals (physiology, voice, facial expressions) predict the other participant's emotion — and if so, at what temporal offset and for which annotation perspective?

> **Framing note:** This project tests *cross-person predictive association*, not affective contagion in the causal sense. A positive result means the signal exists; it does not establish a mechanism. The pipeline is designed to rule out as many confounds as possible before making any stronger claim.

---

## Research Hypotheses

**H₁** — Cross-person prediction exceeds non-dyadic and time-shuffled baselines under LOSO evaluation, *and* exceeds own-label autoregression (emotional inertia alone).

**H₂** — Specific theoretically motivated temporal offsets outperform synchronous prediction, with direction depending on modality: short lags (≤ 400ms) for audio/video, longer lags (5–20s) for physiology.

Both are tested separately for **arousal** and **valence**, across all three annotation perspectives.

---

## What Would Count as Evidence

Cross-person signal is treated as real only if it:

1. Exceeds MeanBaseline, random-dyad, and circular-shift controls;
2. Survives LOSO evaluation (no session leakage);
3. Improves over own-label autoregression (emotional inertia);
4. Is consistent across dyads, not driven by 1–2 sessions;
5. Shows stronger effects for theoretically plausible modalities and lags.

---

## Dataset

**K-EmoCon** — 32 participants in 16 dyadic debates, 172.92 minutes total ([Park et al., 2020](https://www.nature.com/articles/s41597-020-00630-y)). Supplementary code: [Kaist-ICLab/K-EmoCon_SupplementaryCodes](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes).

| Signal | Device | Rate |
|--------|--------|------|
| HR, EDA, TEMP, IBI, BVP | Empatica E4 | 1–64 Hz |
| ACC | E4 | 32 Hz |
| Brainwaves (8 bands), Attention, Meditation | NeuroSky MindWave | 1 Hz |
| Heart rate | Polar H7 | 1 Hz |
| Voice (eGeMAPS functionals) | Stereo WAV | per 5s segment |
| Facial AUs + head pose | Video + py-feat | per frame |

Annotations at **5-second granularity** from three perspectives.

### Annotation perspectives

K-EmoCon's key novelty is three independent annotation types, each measuring something different:

| Type | Measures | Role in this project |
|------|----------|----------------------|
| External observer | Perceived emotional *display* | Primary target |
| Partner | Interaction partner's *perception* | Secondary target |
| Self-report | Felt affect | Tertiary target |
| `external − self` | Expressive suppression / display gap | Exploratory |
| `partner − external` | Partner-specific perception bias | Exploratory |

None of these is "ground truth." Valence CCCs near zero may reflect annotation disagreement, not model failure — see annotation agreement analysis.

---

## Project Structure

```
kemocon_baseline/
├── config.py               # all paths and hyperparameters
├── main.py                 # pipeline entry point (CLI)
├── data_loader.py          # loads raw E4, NeuroSky, all annotation types
├── features.py             # physio feature extraction per 5s segment
├── audio_features.py       # eGeMAPS extraction via opensmile
├── video_features.py       # facial AU / head pose from frame-level CSVs
├── dataset.py              # builds all (X, y) pair types (incl. synchrony)
├── models.py               # all ML models
├── train.py                # LOSO cross-validation loop
├── evaluate.py             # metrics + statistical tests (CCC, Wilcoxon, FDR, bootstrap CI)
├── data_quality.py         # NaN audit, usable-dyad selection
├── plots.py                # figures
├── extract_all.py          # one-time data extraction
├── setup_data.py           # rebuild symlinks on a new machine
├── notebooks/
│   ├── eda.ipynb
│   └── kaggle_video_features.ipynb
├── data/
│   ├── e4_data/               → symlink
│   ├── neurosky_polar_data/   → symlink
│   ├── emotion_annotations/   → symlink (external + partner + self)
│   ├── metadata/              → symlink
│   ├── debate_audios/         → symlink
│   ├── debate_recordings/     → symlink
│   ├── audio_cache/           ← cached eGeMAPS CSVs
│   ├── video_features/        ← frame-level AU + pose (from Kaggle)
│   └── video_features_cache/  ← per-segment aggregated video (fallback)
└── results/
    └── run_YYYYMMDD_HHMM_<modalities>_<optuna>_<controls>/
        ├── config_snapshot.json
        ├── loso_results_primary.csv
        ├── loso_summary_primary.csv
        ├── loso_results_partner.csv
        ├── loso_summary_partner.csv
        ├── loso_results_self.csv
        ├── loso_summary_self.csv
        ├── loso_results_incremental.csv
        ├── loso_summary_incremental.csv
        ├── loso_results_synchrony.csv      (unless --no-synchrony)
        ├── loso_summary_synchrony.csv      (unless --no-synchrony)
        ├── permutation_tests.csv
        ├── stat_model_comparisons.csv
        ├── stat_lag_comparisons.csv
        ├── stat_control_comparisons.csv
        ├── stat_synchrony_comparisons.csv  (unless --no-synchrony)
        └── figures/
```

Each run creates its own timestamped subfolder. The name encodes the key settings (e.g. `run_20250527_1430_physio+audio+video_no-optuna_controls`).

> **Reproducibility:** `data/` uses symlinks to the local K-EmoCon archive. On a new machine, run `python3 setup_data.py --kemocon-root /path/to/kemocon_extracted` to rebuild them. Cached audio/video CSVs are portable.

---

## Features

### Physiological (35 features per 5s window, z-scored against resting baseline)

| Feature | Description |
|---------|-------------|
| `e4_hr_mean/std` | Heart rate |
| `e4_eda_mean/std/slope/range/npeaks` | EDA level, trend, SCR count |
| `e4_temp_mean/std/slope` | Skin temperature |
| `e4_ibi_mean/std/rmssd` | Inter-beat intervals + HRV (RMSSD) |
| `e4_bvp_mean/std/rms/range` | Blood volume pulse |
| `e4_acc_mean/std` | Accelerometer magnitude |
| `bw_{band}_mean` | 8 brainwave bands (delta → gamma) |
| `bw_theta_alpha_ratio` | Relaxation index (θ/α) |
| `bw_engagement_idx` | β / (α + θ) |
| `attention_mean/std`, `meditation_mean/std` | NeuroSky scores |
| `polar_hr_mean/std` | Polar H7 heart rate |

Baseline normalization skips derived features (slope, RMSSD, range, npeaks, ratios) — subtracting a raw-signal mean from a slope is not meaningful.

### Audio (7 features)

eGeMAPS functionals: F0 mean/std, loudness mean/std, jitter, shimmer, HNR.

### Video (12 features, 21/32 participants)

AU04, AU06, AU12, AU17 intensity mean/std; head pitch/yaw mean/std.

> AU detection requires py-feat with SVM model. Frame-level data extracted on Kaggle (`notebooks/kaggle_video_features.ipynb`). For most participants only Pitch/Yaw are populated; AU columns are NaN where py-feat SVM could not initialize.

### Delta features

For every feature `f`, `delta_f = f[t] − f[t−1]` is appended, doubling the vector. This captures rate-of-change. Enabled with `--delta` (off by default, slow).

---

## Prediction Setup

**Cross-person:** B's features → predict A's emotion label (both role directions, giving 32 pairs per lag).

### Lag conditions

| Condition | Shift | Modality note |
|-----------|-------|---------------|
| `lag_0` | 0 — synchronous | all |
| `lag_400ms_av` | −400 ms anticipatory | audio/video only; physio window unchanged |
| `lag_1` … `lag_4` | −5s … −20s (5s steps) | physio-scale lags |

The 400ms shift is applied only to audio/video features (re-extracted with a 400ms window offset). Physiology operates at 5s resolution — a 400ms offset has no meaningful effect on HR/EDA/temperature and is not applied.

### Pair types in `dataset.py`

| Function | Condition tag | Purpose |
|----------|---------------|---------|
| `make_pairs` | `lag_0` … `lag_4`, `lag_400ms_av` | Main cross-person pairs |
| `make_own_signal_pairs` | `own_signal` | Predict A from A's own features |
| `make_random_dyad_pairs` | `random_dyad` | A paired with random non-partner |
| `make_circular_shift_pairs` | `circ_shift` | Partner features time-shifted ≥ 30s |
| `make_label_ar_pairs` (mode=own) | `label_ar_own` | A's own past labels only |
| `make_label_ar_pairs` (mode=partner) | `label_ar_partner` | Partner's past labels |
| `make_label_ar_pairs` (mode=combined) | `label_ar_combined` | Both past-label histories |
| `make_missingness_pairs` | `missingness` | NaN-indicator features only (red-flag) |
| `make_incremental_pairs` | `M1`–`M4` | Incremental feature ablation |
| `make_synchrony_pairs` | `sync_lag_0`, `sync_lag_1` | Dyadic synchrony features only |
| `make_synchrony_augmented_pairs` | `sync_aug_lag_0` | Own features + synchrony features |
| `make_random_synchrony_pairs` | `sync_rnd` | Synchrony from random-dyad pairing |
| `make_circular_shift_synchrony_pairs` | `sync_circ` | Synchrony from circular-shifted pairing |

---

## Models

| Model | Notes |
|-------|-------|
| `MeanBaseline` | Training mean — chance |
| `RidgeCVModel` | Ridge, alpha by leave-one-out CV |
| `SVRModel` | LinearSVR |
| `GradientBoostingModel` | XGBoost (sklearn fallback) |
| `CatBoostModel` | Fixed hyperparameters |
| `CatBoostOptuna` | Optuna inner 3-fold × 10 trials; uses GroupKFold by session |
| `EnsembleModel` | Average of CatBoost + RidgeCV + SVR |

Synchrony LOSO uses a reduced set (MeanBaseline, RidgeCV, CatBoost) for speed.

---

## Evaluation

Primary metric: **CCC** (Lin's Concordance Correlation Coefficient).

Additional metrics reported per fold: Pearson r, RMSE, **pred_dispersion** (std_pred / std_true — detects over-smoothed predictions; < 1 means the model predicts a narrow band).

**Cross-validation:** Leave-One-Session-Out (LOSO), 16 folds. Summary table reports mean ± std ± 95% CI (t-distribution, df=15).

### Statistical tests (`evaluate.py`)

Three test families are run and FDR-corrected independently:

| Family | Comparisons | Tests |
|--------|-------------|-------|
| A — Control | `lag_0` vs each control condition | Wilcoxon signed-rank (one-sided) + sign-flip permutation (10 000 flips) |
| B — Lag | `lag_1`–`lag_4`, `lag_400ms_av` vs `lag_0` | Same |
| C — Model | CatBoost vs Ensemble/RidgeCV/SVR on `lag_0` | Same + bootstrap CI on mean Δ |

**FDR correction:** Benjamini-Hochberg applied within each family separately. Raw and adjusted p-values saved in `stat_*.csv` files.

Parametric t-tests are not used: with N=16 sessions, distribution assumptions are not reliable.

### Annotation agreement analysis

Before trusting low model CCC, the pipeline can report Pearson r between annotation perspectives per participant (enabled with `--agreement`):

| Comparison | Interpretation |
|-----------|----------------|
| `self_vs_ext` | How visible is felt affect? (expressivity) |
| `partner_vs_ext` | Does the partner agree with external observers? |
| `self_vs_partner` | Does the partner understand felt affect? |

Model CCC cannot be expected to exceed this human–human ceiling.

---

## Incremental Model Comparison (M1 → M4)

The key analysis to distinguish partner signal from emotional inertia:

| Config | Features | Added vs previous |
|--------|----------|-------------------|
| M1 `own_label_ar` | A_label[t-1..t-4] | — |
| M2 `own_signal` | M1 + A_physio+audio+video[t] | own multimodal features |
| M3 `label_coupling` | M1 + B_label[t-1..t-4] | partner labels |
| M4 `full` | M2 + B_physio+audio+video[t] | partner multimodal features |

**ΔCCC(M4 − M2)** is the direct test of whether partner features add anything beyond own emotional inertia and own multimodal features. If this delta is near zero, the cross-person model adds nothing meaningful.

---

## Synchrony Extension

Beyond raw partner features, the pipeline also tests **dyadic synchrony features** computed from the listener–speaker feature pair:

| Feature | Formula |
|---------|---------|
| Absolute difference | \|A − B\| per feature |
| Z-score distance | \|z_A − z_B\| per feature |
| Product | A × B per feature |
| Cosine similarity | cos(A, B) scalar per timestep |
| Rolling Pearson r | Per feature, windows of 15s / 30s / 60s (3/6/12 segments) |

Window sizes are set by `SYNCHRONY_WINDOWS = (3, 6, 12)` in `config.py`. Rolling correlation uses `numpy.lib.stride_tricks.sliding_window_view` (vectorized, no Python loops over timesteps).

Synchrony controls (`sync_rnd`, `sync_circ`) use the same feature computation but on randomly-paired or time-shifted dyads, allowing a direct test of whether true-dyad synchrony carries information beyond chance coupling.

---

## Annotation Perspectives (Secondary Analyses)

LOSO is run independently for all three targets:

| Run | Target labels | Saved as |
|-----|--------------|----------|
| Primary | External observer | `loso_*_primary.csv` |
| Secondary | Partner's perception | `loso_*_partner.csv` |
| Tertiary | Self-report | `loso_*_self.csv` |

---

## CLI

```bash
# Full run (physio + audio + video, all analyses)
python3 main.py

# Fast run (skips CatBoostOptuna, much faster)
python3 main.py --no-optuna

# Skip synchrony LOSO (saves ~20-40 min)
python3 main.py --no-optuna --no-synchrony

# Physio features only
python3 main.py --physio-only --no-optuna

# Skip slow secondary analyses
python3 main.py --no-optuna --no-delta --no-agreement

# Modality ablation (physio / audio / video / all)
python3 main.py --ablation --no-optuna
```

Each run saves to `results/run_YYYYMMDD_HHMM_<modalities>_<optuna>_<controls>/`.

### First-time setup

```bash
pip install -r requirements.txt
python3 setup_data.py --kemocon-root /path/to/kemocon_extracted
python3 extract_all.py   # extract audio cache
# Place P{pid}_frames.csv files in data/video_features/ (from Kaggle notebook)
python3 main.py --no-optuna
```

### Video features (Kaggle)

AU and head-pose extraction requires py-feat + GPU. Run `notebooks/kaggle_video_features.ipynb` on Kaggle, download the `P{pid}_frames.csv` outputs, place in `data/video_features/`. The pipeline then aggregates frame-level data per 5s segment at runtime (supports any time offset).

---

## Known Limitations

| Limitation | Status |
|-----------|--------|
| AU columns mostly NaN (py-feat SVM didn't initialize) | Data issue; only Pitch/Yaw reliable for most participants |
| Video available for only 21/32 participants | Missing participants treated as NaN features |
| N=16 dyads — very small sample; results may not replicate | All reported CIs and permutation tests reflect this uncertainty |
| No speaker/listener state features (VAD) | Partner audio may predict emotion simply because partner is speaking |
| Synchrony features implemented but show no benefit over raw partner features | `sync_lag_0` indistinguishable from `sync_rnd` and `sync_circ` controls (all FDR p > 0.98) |
| Categorical emotions (18 labels in self-reports) not yet modeled | Dimensional valence may be too coarse; category models not implemented |
| Random-dyad baseline uses a single random draw | Multiple draws (N=100) would give a proper null CCC distribution |

---

## Key Config (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SEGMENT_SEC` | 5 | Annotation window size (s) |
| `LAGS` | [0,1,2,3,4] | Segment lags (×5s each) |
| `WIN_OFFSET_400MS` | 0.4 | 400ms audio/video anticipatory offset |
| `TARGETS` | ["arousal","valence"] | Prediction targets |
| `RANDOM_SEED` | 42 | Global seed for reproducibility |
| `OPTUNA_TRIALS` | 10 | Optuna trials per LOSO fold |
| `OPTUNA_INNER_SPLITS` | 3 | Inner GroupKFold splits for Optuna |
| `SYNCHRONY_WINDOWS` | (3, 6, 12) | Rolling correlation window sizes in segments (15s, 30s, 60s) |

---

## References

- Park, S. et al. (2020). K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations. *Scientific Data*, 7, 293. https://doi.org/10.1038/s41597-020-00630-y
- Kaist-ICLab. K-EmoCon Supplementary Codes. https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes
- Eyben, F. et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing. *IEEE Transactions on Affective Computing*, 7(2), 190–202.
- Lin, L. I. (1989). A Concordance Correlation Coefficient to Evaluate Reproducibility. *Biometrics*, 45(1), 255–268.
