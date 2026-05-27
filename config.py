from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"

E4_DIR = DATA_ROOT / "e4_data"
NEURO_DIR = DATA_ROOT / "neurosky_polar_data"
ANNOT_DIR = DATA_ROOT / "emotion_annotations" / "aggregated_external_annotations"
METADATA_DIR = DATA_ROOT / "metadata"

DYADS = [(i, i + 1) for i in range(1, 32, 2)]

SEGMENT_SEC = 5

RIDGE_ALPHA = 1.0
SVR_C = 1.0
SVR_GAMMA = "scale"
XGB_N = 200
XGB_DEPTH = 3
XGB_LR = 0.05
LSTM_HIDDEN = 32
LSTM_EPOCHS = 50
LSTM_LR = 1e-3
LSTM_BATCH = 16

OPTUNA_TRIALS = 10
OPTUNA_INNER_SPLITS = 3

TARGETS = ["arousal", "valence"]

# lag=0 → synchronous, lag=1 → 5s back, etc.
LAGS = [0, 1, 2, 3, 4]
RANDOM_SEED = 42
WIN_OFFSET_400MS = 0.4
