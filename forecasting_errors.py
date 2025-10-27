"""
forecasting_errors.py
Calculates out-of-sample forecast errors for volatility models (ARCH, GARCH, EGARCH, etc.).
"""

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm import tqdm  # progress bar (optional, nice for long loops)


def calculate_forecast_errors(retornos: pd.Series, models: dict, rolling_window_size: int = 1000) -> np.ndarray:
    """
    Calculates out-of-sample forecast errors for a set of volatility models using rolling estimation.

    Args:
        retornos (pd.Series): Time series of asset returns.
        models (dict): Dict of models to test, e.g., {'GARCH(1,1)': {'vol': 'GARCH', 'p': 1, 'q': 1}}.
        rolling_window_size (int): Size of rolling window for training.

    Returns:
        np.ndarray: Matrix (T x N) with squared forecast errors for each model.
    """

    model_names = list(models.keys())
    losses = {name: [] for name in model_names}

    # Use tqdm to visualize progress if you’re running long samples
    iterator = tqdm(
        range(rolling_window_size, len(retornos)),
        desc="Backtesting models",
        ncols=80
    )

    for i in iterator:
        train_window = retornos.iloc[i - rolling_window_size:i]
        realized_vol = retornos.iloc[i] ** 2

        for name, params in models.items():
            try:
                model = arch_model(train_window, **params, rescale=False)
                res = model.fit(disp="off")
                forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
                loss = (realized_vol - forecast_var) ** 2
                losses[name].append(loss)
            except Exception as e:
                # Fill with NaN instead of breaking the loop
                losses[name].append(np.nan)
                print(f"⚠️ Model {name} failed at t={i}: {e}")

    # Convert to DataFrame and clean
    losses_df = pd.DataFrame(losses, index=retornos.index[rolling_window_size:])
    losses_df = losses_df.dropna(how="any")

    print(f"\n✅ Finished backtest. Final sample size: {len(losses_df)} obs")
    return losses_df.to_numpy(dtype=float)
