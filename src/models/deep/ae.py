"""PyTorch AutoEncoder anomaly detector.

Requires ``torch``.  If torch is not installed, importing this module raises
``ImportError`` with a clear message.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.models.base import BaseDetector


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for AutoEncoderDetector. "
            "Install it with: pip install torch"
        )


class _MLP(nn.Module if _TORCH_AVAILABLE else object):
    """Simple symmetric MLP autoencoder."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims
        encoder_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            encoder_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_dims = list(reversed(dims))
        decoder_layers: list[nn.Module] = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(self.encoder(x))


class AutoEncoderDetector(BaseDetector):
    """Sliding-window MLP autoencoder.  Reconstruction MSE = anomaly score.

    The model is trained to reconstruct **normal** windows.  Anomalous
    windows produce higher reconstruction error.

    Parameters
    ----------
    window_size:
        Sliding window length.
    hidden_dims:
        Encoder hidden layer sizes (decoder is the mirror).
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    device:
        ``'cpu'`` or ``'cuda'``.  Auto-selects CUDA if available when None.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        window_size: int = 50,
        hidden_dims: Optional[list[int]] = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        _require_torch()
        self.window_size = window_size
        self.hidden_dims = hidden_dims or [64, 16]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def fit(self, x: np.ndarray) -> "AutoEncoderDetector":
        _require_torch()
        torch.manual_seed(self.seed)
        T, C = x.shape
        input_dim = self.window_size * C

        windows = self._sliding_windows(x, self.window_size).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _MLP(input_dim, self.hidden_dims).to(self.device)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()
                loss_fn(self._model(batch), batch).backward()
                optimiser.step()

        self._C = C
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        _require_torch()
        self._check_fitted("_model", self)
        T = x.shape[0]
        windows = self._sliding_windows(x, self.window_size).astype(np.float32)
        tensor = torch.from_numpy(windows).to(self.device)

        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor)
            mse = ((tensor - recon) ** 2).mean(dim=1).cpu().numpy()

        return self._expand_scores(mse, T, self.window_size)
