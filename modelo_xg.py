"""
modelo_xg.py
Simplified and faster XGBoost model for volatility forecasting.
Keeps compatibility with main.py and grid_search.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from tqdm import tqdm


# =============================
# Configurações do modelo
# =============================
@dataclass
class XGBConfig:
    train_min: int = 750
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 2
    reg_lambda: float = 1.0
    random_state: int = 123


# =============================
# Criação de features
# =============================
def _build_target(ret: pd.Series) -> pd.Series:
    """Target = r_t^2 (proxy for realized variance)."""
    return ret.astype(float) ** 2


def _har_features(y: pd.Series) -> pd.DataFrame:
    """HAR-style features: daily, weekly, monthly averages."""
    return pd.DataFrame({
        "HAR_d": y.shift(1),
        "HAR_w": y.shift(1).rolling(5).mean(),
        "HAR_m": y.shift(1).rolling(22).mean()
    }, index=y.index)


def _lag_features(ret: pd.Series) -> pd.DataFrame:
    """Lag features of |r| and r²."""
    a = ret.abs()
    r2 = ret ** 2
    return pd.DataFrame({
        "absret_lag1": a.shift(1),
        "absret_lag2": a.shift(2),
        "r2_lag1": r2.shift(1),
        "r2_lag2": r2.shift(2),
    }, index=ret.index)


def _join_exo(idx: pd.DatetimeIndex, exo: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Optional exogenous variables."""
    if exo is None:
        return pd.DataFrame(index=idx)
    return exo.reindex(idx).sort_index().shift(1)


def _build_X(ret: pd.Series, exo: Optional[pd.DataFrame]) -> Tuple[pd.Series, pd.DataFrame]:
    y = _build_target(ret)
    X = pd.concat([
        _har_features(y),
        _lag_features(ret),
        _join_exo(ret.index, exo)
    ], axis=1)
    return y, X


# =============================
# Backtest 1-step-ahead
# =============================
def run_xgb_mse_backtest(
    ret: pd.Series,
    exo: Optional[pd.DataFrame] = None,
    cfg: XGBConfig = XGBConfig()
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict]:
    """
    One-step-ahead expanding window forecast with XGBoost.

    Returns:
        losses_array: np.ndarray (T x 1)
        losses_df: pd.DataFrame
        preds_df: pd.DataFrame
        meta: Dict with metadata
    """
    ret = ret.sort_index().astype(float)
    y, X = _build_X(ret, exo)

    y_true_list, y_hat_list, idx_list = [], [], []

    iterator = tqdm(
        range(cfg.train_min, len(ret)),
        desc="XGB rolling forecast",
        ncols=80
    )

    # Pre-define model to reuse parameters (faster than re-creating each time)
    for i in iterator:
        X_train = X.iloc[:i].dropna()
        y_train = y.loc[X_train.index]

        # Skip if not enough data
        if len(X_train) < cfg.train_min:
            continue

        x_next = X.iloc[i]
        if x_next.isna().any():
            continue

        model = XGBRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

        y_pred = float(model.predict(x_next.values.reshape(1, -1))[0])
        y_true = float(y.iloc[i])

        y_hat_list.append(max(y_pred, 1e-12))  # avoid negatives
        y_true_list.append(y_true)
        idx_list.append(y.index[i])

    preds_df = pd.DataFrame({"yhat": y_hat_list}, index=idx_list)
    y_true_ser = pd.Series(y_true_list, index=idx_list, name="y_true")

    mse = (y_true_ser - preds_df["yhat"]) ** 2
    losses_df = pd.DataFrame({"XGB_MSE": mse}, index=idx_list)

    losses_array = losses_df.to_numpy(dtype=float)
    meta = {"index": losses_df.index, "model_names": ["XGB_MSE"], "loss_func": "MSE"}
    print(f"\n✅ Finished XGB backtest. {len(losses_df)} valid predictions.")
    return losses_array, losses_df, preds_df, meta


# =============================
# Teste rápido (execução direta)
# =============================
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=600, freq="D")
    ret_sim = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

    losses_array, losses_df, preds_df, meta = run_xgb_mse_backtest(ret_sim)
    print(losses_df.tail())
