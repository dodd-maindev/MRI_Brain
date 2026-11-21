"""Exponential Moving Average (EMA) wrapper with decay warmup and BN buffer sync."""

from contextlib import contextmanager
from typing import Dict, Iterator
import torch
import torch.nn as nn


class ModelExponentialMovingAverage:
    """Maintains shadow EMA parameters and floating-point BN buffers with warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        """Clone initial floating-point parameters and buffers into shadow state."""
        self.decay, self.num_updates = decay, 0
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow state with warmed-up decay after each optimizer step."""
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap model parameters and BN stats with EMA shadow state."""
        state = model.state_dict()
        backup = {k: state[k].detach().clone() for k in self.shadow if k in state}
        for k in backup:
            state[k].copy_(self.shadow[k])
        try:
            yield
        finally:
            cur_state = model.state_dict()
            for k, orig in backup.items():
                cur_state[k].copy_(orig)
