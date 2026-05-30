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
CB_ITERS = 200
CB_DEPTH = 3
CB_LR    = 0.05
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.2
LSTM_SEQ_LEN = 10    # sliding window in segments (10 × 5s = 50s context)
LSTM_EPOCHS  = 50
LSTM_LR      = 1e-3
LSTM_BATCH   = 32

OPTUNA_TRIALS = 10
OPTUNA_INNER_SPLITS = 3

TARGETS = ["arousal", "valence"]

# lag=0 → synchronous, lag=1 → 5s back, etc.
LAGS = [0, 1, 2, 3, 4]
RANDOM_SEED = 42
WIN_OFFSET_400MS = 0.4

# Synchrony feature windows in segments (3→15s, 6→30s, 12→60s)
SYNCHRONY_WINDOWS = (3, 6, 12)
