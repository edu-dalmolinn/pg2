"""
modelo_lstm.py
Simplified LSTM volatility forecasting for MCS evaluation.
Prioritizes speed and simplicity over deep hyperparameter tuning.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =============================
# Configuração
# =============================
@dataclass
class LSTMConfig:
    train_min: int = 750
    lookback: int = 20
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 5   # drastically reduced to make faster
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 123
    min_sequences: int = 200  # << used below (no more hardcoded 200)


# =============================
# Funções auxiliares
# =============================
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_target(ret: pd.Series) -> pd.Series:
    """Target = squared returns (proxy for volatility)."""
    return (ret.astype(float)) ** 2


def _create_sequences(series: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping sequences of length `lookback`."""
    series = series.dropna().astype(np.float32).values
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# =============================
# Estrutura do modelo LSTM
# =============================
class LSTMVolatilityModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =============================
# Função de treino
# =============================
def _train_lstm(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, cfg: LSTMConfig, device: torch.device) -> None:
    X_tensor = torch.from_numpy(X_train[:, :, None]).to(device)
    y_tensor = torch.from_numpy(y_train[:, None]).to(device)

    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=cfg.batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()


# =============================
# Backtest 1-step-ahead
# =============================
def run_lstm_mse_backtest(ret: pd.Series, cfg: LSTMConfig = LSTMConfig()) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Runs a simplified one-step-ahead backtest for volatility forecasting.
    Returns: (losses_array, losses_df, preds_df, meta)
    """
    ret = ret.sort_index().astype(float)
    _set_seed(cfg.random_state)

    device = torch.device(cfg.device)
    vol = _build_target(ret)
    y_true_list: List[float] = []
    y_hat_list: List[float] = []
    idx_list: List[pd.Timestamp] = []

    iterator = tqdm(range(cfg.train_min, len(ret)), desc="LSTM rolling forecast", ncols=80)

    for i in iterator:
        train_vol = vol.iloc[:i]
        # need enough total observations to build sequences
        if len(train_vol) < cfg.lookback + cfg.train_min:
            continue

        X_train, y_train = _create_sequences(train_vol, cfg.lookback)
        # >>> use cfg.min_sequences (no hardcoded 200)
        if len(X_train) < cfg.min_sequences:
            continue

        # Normalize (scalar mean/std over all entries; simple and fast)
        x_mean, x_std = X_train.mean(), X_train.std() + 1e-8
        y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
        X_train_n = ((X_train - x_mean) / x_std).astype(np.float32)
        y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)

        model = LSTMVolatilityModel(cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
        _train_lstm(model, X_train_n, y_train_n, cfg, device)

        # Predict next step
        seq = vol.iloc[i - cfg.lookback: i].values.astype(np.float32)
        if np.isnan(seq).any():
            continue
        seq_n = ((seq - x_mean) / x_std).astype(np.float32)

        model.eval()
        with torch.no_grad():
            x_next = torch.from_numpy(seq_n.reshape(1, cfg.lookback, 1)).to(device)
            y_pred_n = model(x_next).item()

        y_pred = float(max(y_pred_n * y_std + y_mean, 1e-12))
        y_true = float(vol.iloc[i])

        idx_list.append(ret.index[i])
        y_true_list.append(y_true)
        y_hat_list.append(y_pred)

    preds_df = pd.DataFrame({"yhat": y_hat_list}, index=idx_list)
    y_true_ser = pd.Series(y_true_list, index=idx_list, name="y_true")

    mse = (y_true_ser - preds_df["yhat"]) ** 2
    losses_df = pd.DataFrame({"LSTM_MSE": mse}, index=idx_list)
    losses_array = losses_df.to_numpy(dtype=float)
    meta = {"index": losses_df.index, "model_names": ["LSTM_MSE"], "loss_func": "MSE"}

    print(f"\n✅ Finished LSTM backtest. {len(losses_df)} valid predictions.")
    return losses_array, losses_df, preds_df, meta


# =============================
# Teste rápido
# =============================
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=800, freq="D")
    ret_sim = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

    losses_array, losses_df, preds_df, meta = run_lstm_mse_backtest(
        ret_sim,
        cfg=LSTMConfig(train_min=500, lookback=20, epochs=3, min_sequences=200)
    )
    print(losses_df.tail())
