"""
grid_search.py
Lightweight grid search for XGBoost, Bagging, and LSTM models.
Designed for simplicity, speed, and integration with the MCS framework.
"""

import numpy as np
import pandas as pd
from yahooquery import Ticker
from sklearn.metrics import mean_squared_error
from itertools import product

# Import model modules
from models.modelo_xg import run_xgb_mse_backtest, XGBConfig
from models.modelo_bagging import run_bagging_mse_backtest, BaggingConfig
from models.modelo_lstm import run_lstm_mse_backtest, LSTMConfig


# =========================================================
# 1. Load data
# =========================================================
def load_returns():
    """Load Brent Crude Oil returns."""
    ticker = Ticker("BZ=F")
    df = ticker.history(start="2000-01-01", end="2024-12-31")
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        df = df[df["symbol"] == "BZ=F"]
        df = df.set_index("date").sort_index()

    precos = df["adjclose"].astype(float)
    retornos = 100 * precos.pct_change().dropna()
    retornos.index = pd.to_datetime(retornos.index).tz_localize(None)
    return retornos


# =========================================================
# 2. Utility: evaluate model quickly (mean MSE)
# =========================================================
def evaluate_model(losses_array):
    """Compute mean MSE for model evaluation."""
    return np.nanmean(losses_array)


# =========================================================
# 3. Grid search: XGBoost
# =========================================================
def grid_search_xgb(retornos):
    print("\n🔍 Grid Search: XGBoost")
    param_grid = {
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 4],
        "subsample": [0.8, 1.0],
        "n_estimators": [300, 500],
    }

    best_score = np.inf
    best_params = None

    for params in product(*param_grid.values()):
        cfg = XGBConfig(
            learning_rate=params[0],
            max_depth=params[1],
            subsample=params[2],
            n_estimators=params[3]
        )
        losses_array, _, _, _ = run_xgb_mse_backtest(retornos, cfg=cfg)
        mse = evaluate_model(losses_array)
        print(f"Params {cfg}: Mean MSE={mse:.6f}")

        if mse < best_score:
            best_score = mse
            best_params = cfg

    print(f"\n✅ Best XGBoost config: {best_params}")
    print(f"Lowest MSE: {best_score:.6f}")
    return best_params


# =========================================================
# 4. Grid search: Bagging
# =========================================================
def grid_search_bagging(retornos):
    print("\n🔍 Grid Search: Bagging")
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, 8],
        "max_samples": [0.7, 0.9],
    }

    best_score = np.inf
    best_params = None

    for params in product(*param_grid.values()):
        cfg = BaggingConfig(
            n_estimators=params[0],
            max_depth=params[1],
            max_samples=params[2]
        )
        losses_array, _, _, _ = run_bagging_mse_backtest(retornos, cfg=cfg)
        mse = evaluate_model(losses_array)
        print(f"Params {cfg}: Mean MSE={mse:.6f}")

        if mse < best_score:
            best_score = mse
            best_params = cfg

    print(f"\n✅ Best Bagging config: {best_params}")
    print(f"Lowest MSE: {best_score:.6f}")
    return best_params


# =========================================================
# 5. Grid search: LSTM
# =========================================================
def grid_search_lstm(retornos):
    print("\n🔍 Grid Search: LSTM (light version)")
    param_grid = {
        "lookback": [10, 20],
        "hidden_size": [8, 16],
        "epochs": [3, 5],
    }

    # Sample subset for faster tuning
    ret_sample = retornos[-1500:]

    best_score = np.inf
    best_params = None

    for params in product(*param_grid.values()):
        cfg = LSTMConfig(
            lookback=params[0],
            hidden_size=params[1],
            epochs=params[2],
            train_min=500
        )
        losses_array, _, _, _ = run_lstm_mse_backtest(ret_sample, cfg=cfg)
        mse = evaluate_model(losses_array)
        print(f"Params {cfg}: Mean MSE={mse:.6f}")

        if mse < best_score:
            best_score = mse
            best_params = cfg

    print(f"\n✅ Best LSTM config: {best_params}")
    print(f"Lowest MSE: {best_score:.6f}")
    return best_params


# =========================================================
# 6. Main execution
# =========================================================
if __name__ == "__main__":
    retornos = load_returns()

    best_xgb = grid_search_xgb(retornos)
    best_bag = grid_search_bagging(retornos)
    best_lstm = grid_search_lstm(retornos)

    print("\n====== SUMMARY ======")
    print("Best XGBoost:", best_xgb)
    print("Best Bagging:", best_bag)
    print("Best LSTM:", best_lstm)
