from typing import Optional, List, Dict
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import LinearSVR as _SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor as _XGB
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _XGB
    _HAS_XGB = False

from config import (RIDGE_ALPHA, SVR_C, SVR_GAMMA,
                    XGB_N, XGB_DEPTH, XGB_LR,
                    LSTM_HIDDEN, LSTM_EPOCHS, LSTM_LR, LSTM_BATCH,
                    OPTUNA_TRIALS, OPTUNA_INNER_SPLITS)


def _impute_median(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    idx = np.where(np.isnan(X))
    X = X.copy()
    X[idx] = np.take(medians, idx[1])
    return X, medians


def _apply_impute(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    X = X.copy()
    idx = np.where(np.isnan(X))
    X[idx] = np.take(medians, idx[1])
    return X


class MeanBaseline:
    name = "MeanBaseline"
    feature_importances_ = None

    def fit(self, X_train, y_train):
        self._mean = float(np.nanmean(y_train))
        return self

    def predict(self, X_test):
        return np.full(len(X_test), self._mean)


class RidgeRegression:
    name = "Ridge"
    feature_importances_ = None

    def __init__(self):
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ])
        self._medians = None

    def fit(self, X_train, y_train):
        X, self._medians = _impute_median(X_train)
        self._pipe.fit(X, y_train)
        return self

    def predict(self, X_test):
        X = _apply_impute(X_test, self._medians)
        return self._pipe.predict(X)


class SVRModel:
    name = "SVR"
    feature_importances_ = None

    def __init__(self):
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", _SVR(C=SVR_C, max_iter=2000, dual="auto")),
        ])
        self._medians = None

    def fit(self, X_train, y_train):
        X, self._medians = _impute_median(X_train)
        self._pipe.fit(X, y_train)
        return self

    def predict(self, X_test):
        X = _apply_impute(X_test, self._medians)
        return self._pipe.predict(X)


class GradientBoostingModel:
    name = "XGBoost" if _HAS_XGB else "GradientBoosting"
    feature_importances_: Optional[np.ndarray] = None

    def __init__(self):
        if _HAS_XGB:
            self._model = _XGB(
                n_estimators=XGB_N,
                max_depth=XGB_DEPTH,
                learning_rate=XGB_LR,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        else:
            self._model = _XGB(
                n_estimators=XGB_N,
                max_depth=XGB_DEPTH,
                learning_rate=XGB_LR,
                random_state=42,
            )
        self._medians = None

    def fit(self, X_train, y_train):
        X, self._medians = _impute_median(X_train)
        self._model.fit(X, y_train)
        if hasattr(self._model, "feature_importances_"):
            self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X_test):
        X = _apply_impute(X_test, self._medians)
        return self._model.predict(X)


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _LSTMNet(object if not _HAS_TORCH else object):
    pass


if _HAS_TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size,
                                num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    name = "LSTM"
    feature_importances_ = None

    def __init__(self):
        self._net = None
        self._medians = None
        self._scaler_mean = None
        self._scaler_std = None

    def _scale(self, X):
        return (X - self._scaler_mean) / (self._scaler_std + 1e-8)

    def fit(self, X_train, y_train):
        if not _HAS_TORCH:
            self._fallback = RidgeRegression().fit(X_train, y_train)
            return self

        X, self._medians = _impute_median(X_train)
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0)
        X_s = self._scale(X)

        n_feat = X_s.shape[1]
        self._net = _LSTMNet(n_feat, LSTM_HIDDEN)
        opt = torch.optim.Adam(self._net.parameters(), lr=LSTM_LR)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(X_s, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y_train, dtype=torch.float32)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)

        self._net.train()
        for _ in range(LSTM_EPOCHS):
            for xb, yb in dl:
                opt.zero_grad()
                pred = self._net(xb)
                loss_fn(pred, yb).backward()
                opt.step()
        return self

    def predict(self, X_test):
        if not _HAS_TORCH or self._net is None:
            return self._fallback.predict(X_test)
        X = _apply_impute(X_test, self._medians)
        X_s = self._scale(X)
        Xt = torch.tensor(X_s, dtype=torch.float32).unsqueeze(1)
        self._net.eval()
        with torch.no_grad():
            return self._net(Xt).numpy()


try:
    from catboost import CatBoostRegressor as _CB
    _HAS_CB = True
except ImportError:
    _HAS_CB = False

from config import XGB_N, XGB_DEPTH, XGB_LR


class CatBoostModel:
    name = "CatBoost"
    feature_importances_: Optional[np.ndarray] = None

    def __init__(self):
        if _HAS_CB:
            self._model = _CB(
                iterations=XGB_N,
                depth=XGB_DEPTH,
                learning_rate=XGB_LR,
                loss_function="RMSE",
                random_seed=42,
                verbose=0,
                thread_count=-1,
                allow_writing_files=False,
            )
        else:
            print("[WARN] catboost not installed — CatBoost disabled")
            self._model = None

    def fit(self, X_train, y_train):
        if self._model is None:
            return self
        self._model.fit(X_train, y_train)
        if hasattr(self._model, "feature_importances_"):
            self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X_test):
        if self._model is None:
            return np.full(len(X_test), np.nan)
        return self._model.predict(X_test)


class RidgeCVModel:
    name = "RidgeCV"
    feature_importances_ = None

    _ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    def __init__(self):
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=self._ALPHAS, cv=5)),
        ])
        self._medians = None

    def fit(self, X_train, y_train):
        X, self._medians = _impute_median(X_train)
        self._pipe.fit(X, y_train)
        return self

    def predict(self, X_test):
        X = _apply_impute(X_test, self._medians)
        return self._pipe.predict(X)


class CatBoostOptuna:
    name = "CatBoostOptuna"
    feature_importances_: Optional[np.ndarray] = None

    def __init__(self, n_trials: int = OPTUNA_TRIALS,
                 n_splits: int = OPTUNA_INNER_SPLITS):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self._model = None
        self._fallback = None

    def fit(self, X_train, y_train, groups=None):
        if not _HAS_CB:
            print("[WARN] catboost not installed — CatBoostOptuna disabled")
            return self

        from config import RANDOM_SEED

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            self._model = _CB(
                iterations=400, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
                loss_function="RMSE", random_seed=RANDOM_SEED,
                verbose=0, allow_writing_files=False,
            )
            self._model.fit(X_train, y_train)
            self.feature_importances_ = self._model.feature_importances_
            return self

        from sklearn.model_selection import KFold, GroupKFold
        from evaluate import ccc as _ccc_fn

        def objective(trial):
            params = {
                "iterations":        trial.suggest_int("iterations", 100, 600),
                "depth":             trial.suggest_int("depth", 3, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "loss_function":     trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
                "random_seed": RANDOM_SEED, "verbose": 0, "allow_writing_files": False,
                "thread_count": -1,
            }
            if groups is not None:
                kf = GroupKFold(n_splits=self.n_splits)
                splits = kf.split(X_train, y_train, groups)
            else:
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
                splits = kf.split(X_train)
            scores = []
            for tr_i, va_i in splits:
                m = _CB(**params)
                m.fit(X_train[tr_i], y_train[tr_i])
                scores.append(_ccc_fn(y_train[va_i], m.predict(X_train[va_i])))
            return float(np.nanmean(scores))

        def _cb(study, trial):
            print(f"      optuna trial {trial.number+1}/{self.n_trials}"
                  f"  CCC={trial.value:.4f}", flush=True)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=self.n_trials,
                       show_progress_bar=False, callbacks=[_cb])

        best = dict(study.best_params)
        best.update({"random_seed": RANDOM_SEED, "verbose": 0, "allow_writing_files": False})

        self._model = _CB(**best)
        self._model.fit(X_train, y_train)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X_test):
        if self._model is None:
            return np.full(len(X_test), np.nan)
        return self._model.predict(X_test)


class EnsembleModel:
    name = "Ensemble"
    feature_importances_ = None

    def fit(self, X_train, y_train):
        self._sub = [CatBoostModel(), RidgeCVModel(), SVRModel()]
        for m in self._sub:
            m.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        preds = np.stack([m.predict(X_test) for m in self._sub], axis=0)
        return preds.mean(axis=0)


def all_models():
    return [
        MeanBaseline(),
        RidgeCVModel(),
        SVRModel(),
        GradientBoostingModel(),
        CatBoostModel(),
        CatBoostOptuna(),
        EnsembleModel(),
        # LSTMModel(),
    ]
