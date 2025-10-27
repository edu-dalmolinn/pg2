"""
main.py
Final integrated version
Evaluates multiple volatility forecasting models and compares them using the Model Confidence Set (MCS) procedure.
"""

import numpy as np
import pandas as pd
from yahooquery import Ticker

# Local imports
from forecasting_errors import calculate_forecast_errors
from models.modelo_xg import run_xgb_mse_backtest, XGBConfig
from models.modelo_bagging import run_bagging_mse_backtest, BaggingConfig
from models.modelo_lstm import run_lstm_mse_backtest, LSTMConfig
from grid_search import grid_search_xgb, grid_search_bagging, grid_search_lstm


# =========================================================
# 1. Load data
# =========================================================
# ===== Quick test toggle =====
QUICK_TEST = False   # set False for full experiment
QUICK_LIMIT = 125   # how many last obs to keep in quick mode


def load_data():
    ticker = Ticker("BZ=F")
    df = ticker.history(start="2000-01-01", end="2024-12-31")
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        df = df[df["symbol"] == "BZ=F"]
        df = df.set_index("date").sort_index()

    precos = df["adjclose"]
    retornos = 100 * precos.pct_change().dropna()
    retornos.index = pd.to_datetime(retornos.index).tz_localize(None)

    if QUICK_TEST:
        retornos = retornos.iloc[-QUICK_LIMIT:]
        print(f"[QUICK] Using only last {len(retornos)} observations.")

    return retornos


# =========================================================
# 2. Define econometric models
# =========================================================
def define_models():
    """Returns a dict with simple ARCH/GARCH/EGARCH model specs."""
    return {
        "ARCH(1)": {"vol": "ARCH", "p": 1},
        "GARCH(1,1)": {"vol": "GARCH", "p": 1, "q": 1},
        "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "q": 1},
    }


# =========================================================
# 3. Run all models
# =========================================================
def run_all_models(retornos: pd.Series, use_gridsearch: bool = False):
    """
    Runs all forecasting models, aligns their loss arrays, and returns:
      - losses_all (np.ndarray): T x K matrix of losses
      - model_labels (list[str]): names for each column/model
    """
    # ---- Econometric (ARCH/GARCH/EGARCH) ----
    rolling_win = 1000
    if QUICK_TEST:
        rolling_win = min(60, max(20, len(retornos) // 3))  # smaller window on quick runs

    errors_econometric = calculate_forecast_errors(
        retornos,
        define_models(),
        rolling_window_size=rolling_win,
    )  # shape: T x 3  (ARCH, GARCH, EGARCH)

    # ---- ML configs ----
    if use_gridsearch and not QUICK_TEST:
        xgb_best = grid_search_xgb(retornos)
        bag_best = grid_search_bagging(retornos)
        lstm_best = grid_search_lstm(retornos)
    else:
        if QUICK_TEST:
            xgb_best = XGBConfig(
                train_min=max(5, len(retornos) // 4),
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            bag_best = BaggingConfig(
                train_min=max(5, len(retornos) // 4),
                n_estimators=30,
                max_depth=4,
                max_samples=0.8,
                max_features=0.8,
            )
            lstm_best = LSTMConfig(
                train_min=max(8, len(retornos) // 3),
                lookback=min(10, max(5, len(retornos) // 6)),
                epochs=2,
                hidden_size=8,
                min_sequences=20 
            )
        else:
            xgb_best = XGBConfig()
            bag_best = BaggingConfig()
            lstm_best = LSTMConfig()

    # ---- Run each ML model; skip gracefully if too little data ----
    xgb_losses_array = None
    bag_losses_array = None
    lstm_losses_array = None

    try:
        xgb_losses_array, _, _, _ = run_xgb_mse_backtest(retornos, cfg=xgb_best)
    except Exception as e:
        print(f"[WARN] XGB failed in this run: {e}")

    try:
        bag_losses_array, _, _, _ = run_bagging_mse_backtest(retornos, cfg=bag_best)
    except Exception as e:
        print(f"[WARN] Bagging failed in this run: {e}")

    try:
        lstm_losses_array, _, _, _ = run_lstm_mse_backtest(retornos, cfg=lstm_best)
    except Exception as e:
        print(f"[WARN] LSTM failed in this run: {e}")

    # ---- Collect available arrays and names (order defines labels) ----
    candidate_arrays = [
        ("ARCH(1)",       errors_econometric[:, [0]] if errors_econometric.size else None),
        ("GARCH(1,1)",    errors_econometric[:, [1]] if errors_econometric.size else None),
        ("EGARCH(1,1)",   errors_econometric[:, [2]] if errors_econometric.size else None),
        ("XGBoost",       xgb_losses_array),
        ("Bagging",       bag_losses_array),
        ("LSTM",          lstm_losses_array),
    ]

    available = [(name, arr) for (name, arr) in candidate_arrays if arr is not None and arr.size > 0]
    if len(available) < 2:
        raise ValueError("Too few models produced losses. Increase data or relax quick-test settings.")

    # ---- Align by shortest T and hstack ----
    min_T = min(arr.shape[0] for _, arr in available)
    losses_all = np.hstack([arr[-min_T:] for _, arr in available])
    model_labels = [name for name, _ in available]

    print(f"\n✅ Final losses array shape: {losses_all.shape}")
    print(f"Models: {model_labels}")
    return losses_all, model_labels


# =========================================================
# 4. Run MCS
# =========================================================
def run_mcs(losses_all: np.ndarray, model_labels, alpha: float = 0.05, n_boot: int = 2000) -> pd.DataFrame:
    from model_confidence_set import ModelConfidenceSet

    losses_all = np.asarray(losses_all, dtype=float)
    mask = np.isfinite(losses_all).all(axis=1)
    losses_all = losses_all[mask]

    T, K = losses_all.shape
    if T < 5 or K < 2:
        raise ValueError(f"Too few observations for MCS after cleaning (T={T}, K={K}).")

    # Small but positive block length for short samples
    block_len = max(1, min(10, int(round(T / 5))))
    if QUICK_TEST and n_boot > 500:
        n_boot = 300

    print(f"Running MCS with T={T}, K={K}, block_len={block_len}, n_boot={n_boot}, alpha={alpha}")
    mcs = ModelConfidenceSet(
        losses_all,
        n_boot=n_boot,
        alpha=alpha,
        block_len=block_len,
        show_progress=True,
    )
    mcs.compute()
    results = mcs.results(as_dataframe=True)
    # Attach readable model names
    if len(model_labels) == K:
        results.index = model_labels
    return results


# =========================================================
# 5. Main
# =========================================================
if __name__ == "__main__":
    retornos = load_data()
    losses_all, model_labels = run_all_models(retornos, use_gridsearch=True)
    results = run_mcs(losses_all, model_labels, alpha=0.05, n_boot=2000)
    print("\n===== MCS RESULTS =====")
    print(results)
