"""Exponential Moving Average (EMA) wrapper for smoother validation convergence."""

from contextlib import contextmanager
from typing import Dict, Iterator
import torch
import torch.nn as nn


class ModelExponentialMovingAverage:
    """Maintains shadow EMA parameters for stable evaluation and checkpointing."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        """Clone initial floating-point parameters into shadow dictionary."""
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters() if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights after each optimizer step."""
        d = self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap model parameters with EMA shadow weights."""
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
