"""
modelo_bagging.py
Simplified Bagging model for volatility forecasting (faster version).
Keeps compatibility with main.py and MCS evaluation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================
# Configuração
# =============================
@dataclass
class BaggingConfig:
    train_min: int = 750
    n_estimators: int = 100
    max_samples: float = 0.8
    max_features: float = 0.8
    max_depth: int = 6
    random_state: int = 123
    feature_window: int = 22


# =============================
# Features e Target
# =============================
def _build_target(ret: pd.Series) -> pd.Series:
    return (ret.astype(float)) ** 2


def _har_features(y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "HAR_d": y.shift(1),
        "HAR_w": y.shift(1).rolling(5).mean(),
        "HAR_m": y.shift(1).rolling(22).mean(),
    }, index=y.index)


def _lag_features(ret: pd.Series, window: int = 10) -> pd.DataFrame:
    """Simple and fast lag features."""
    r2 = ret ** 2
    abs_ret = ret.abs()
    df = pd.DataFrame({
        "ret_lag1": ret.shift(1),
        "ret_lag2": ret.shift(2),
        "absret_lag1": abs_ret.shift(1),
        "absret_lag2": abs_ret.shift(2),
        "r2_lag1": r2.shift(1),
        "r2_lag2": r2.shift(2),
        "vol_roll_5": r2.rolling(5).mean().shift(1),
        "vol_roll_22": r2.rolling(22).mean().shift(1)
    }, index=ret.index)
    return df


def _build_X(ret: pd.Series, exo: Optional[pd.DataFrame], window: int) -> Tuple[pd.Series, pd.DataFrame]:
    y = _build_target(ret)
    X = pd.concat([
        _har_features(y),
        _lag_features(ret, window),
        exo.reindex(ret.index).shift(1) if exo is not None else pd.DataFrame(index=ret.index)
    ], axis=1)
    return y, X


# =============================
# Backtest
# =============================
def run_bagging_mse_backtest(
    ret: pd.Series,
    exo: Optional[pd.DataFrame] = None,
    cfg: BaggingConfig = BaggingConfig()
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Runs a 1-step-ahead expanding-window backtest with BaggingRegressor.
    Returns losses and predictions compatible with other models.
    """
    ret = ret.sort_index().astype(float)
    y, X = _build_X(ret, exo, window=cfg.feature_window)

    y_true_list, y_hat_list, idx_list = [], [], []

    iterator = tqdm(range(cfg.train_min, len(ret)), desc="Bagging rolling forecast", ncols=80)

    for i in iterator:
        X_train = X.iloc[:i].dropna()
        y_train = y.loc[X_train.index]

        if i >= len(X) or X.iloc[i].isna().any():
            continue

        x_next = X.iloc[i:i+1]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        x_next_scaled = scaler.transform(x_next)

        model = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=cfg.max_depth, random_state=cfg.random_state),
            n_estimators=cfg.n_estimators,
            max_samples=cfg.max_samples,
            max_features=cfg.max_features,
            bootstrap=True,
            random_state=cfg.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        y_hat = float(model.predict(x_next_scaled)[0])
        y_true = float(y.iloc[i])

        y_hat_list.append(max(y_hat, 1e-12))
        y_true_list.append(y_true)
        idx_list.append(ret.index[i])

    preds_df = pd.DataFrame({"yhat": y_hat_list}, index=idx_list)
    y_true_ser = pd.Series(y_true_list, index=idx_list, name="y_true")

    mse = (y_true_ser - preds_df["yhat"]) ** 2
    losses_df = pd.DataFrame({"BAGGING_MSE": mse}, index=idx_list)
    losses_array = losses_df.to_numpy(dtype=float)

    meta = {"index": losses_df.index, "model_names": ["BAGGING_MSE"], "loss_func": "MSE"}

    print(f"\n✅ Finished Bagging backtest. {len(losses_df)} valid predictions.")
    return losses_array, losses_df, preds_df, meta


# =============================
# Teste rápido
# =============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=700, freq="D")
    ret_sim = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

    losses_array, losses_df, preds_df, meta = run_bagging_mse_backtest(ret_sim)

    print(losses_df.tail())
    plt.plot(preds_df.index[-100:], preds_df["yhat"][-100:], label="Predicted")
    plt.plot(preds_df.index[-100:], (ret_sim[-100:]**2), label="Realized")
    plt.legend()
    plt.show()
