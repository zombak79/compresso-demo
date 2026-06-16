from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class SRPTensor:
    """Tiny SRP tensor reader used by the public Streamlit demo.

    This mirrors the serialized format produced by Compresso but intentionally
    implements only what the demo needs: fixed-k row-packed sparse embeddings.
    """

    __slots__ = ("cols", "vals", "shape", "prefix_shape")

    def __init__(
        self,
        *,
        cols: torch.Tensor,
        vals: torch.Tensor,
        shape: tuple[int, int],
        prefix_shape: tuple[int, ...] | None = None,
        validate: bool = True,
    ) -> None:
        if cols.dtype != torch.long:
            raise ValueError("SRPTensor.cols must be torch.long")
        if cols.dim() != 2 or vals.dim() != 2:
            raise ValueError("SRPTensor.cols and SRPTensor.vals must be 2D")
        if cols.shape != vals.shape:
            raise ValueError(f"cols.shape {tuple(cols.shape)} != vals.shape {tuple(vals.shape)}")
        rows, _ = cols.shape
        if int(shape[0]) != int(rows):
            raise ValueError(f"shape[0]={shape[0]} must equal rows={rows}")
        cols_total = int(shape[1])
        if cols_total <= 0:
            raise ValueError("shape[1] must be positive")
        if validate and cols.numel() > 0:
            cmin = int(cols.min().item())
            cmax = int(cols.max().item())
            if cmin < 0 or cmax >= cols_total:
                raise ValueError(f"cols out of bounds: min={cmin}, max={cmax}, allowed [0, {cols_total - 1}]")
        self.cols = cols
        self.vals = vals
        self.shape = (int(shape[0]), cols_total)
        self.prefix_shape = tuple(prefix_shape) if prefix_shape is not None else None

    @property
    def rows(self) -> int:
        return int(self.shape[0])

    @property
    def cols_total(self) -> int:
        return int(self.shape[1])

    @property
    def k(self) -> int:
        return int(self.cols.size(1))

    @staticmethod
    def from_dict(payload: dict[str, Any], *, validate: bool = True) -> "SRPTensor":
        if not isinstance(payload, dict):
            raise ValueError("SRP payload must be a dict")
        if payload.get("layout", "srp") != "srp":
            raise ValueError(f"Unsupported SRP layout: {payload.get('layout')!r}")
        if int(payload.get("version", 1)) != 1:
            raise ValueError(f"Unsupported SRP version: {payload.get('version')!r}")
        return SRPTensor(
            cols=payload["cols"],
            vals=payload["vals"],
            shape=tuple(payload["shape"]),
            prefix_shape=tuple(payload["prefix_shape"]) if payload.get("prefix_shape") is not None else None,
            validate=validate,
        )


def load_srp_tensor(path: str | Path, map_location=None, *, validate: bool = True) -> SRPTensor:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return SRPTensor.from_dict(payload, validate=validate)
