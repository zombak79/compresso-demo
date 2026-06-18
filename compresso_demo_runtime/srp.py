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
    def device(self):
        return self.vals.device

    @property
    def dtype(self):
        return self.vals.dtype

    @property
    def requires_grad(self) -> bool:
        return bool(self.vals.requires_grad)

    @property
    def is_cuda(self) -> bool:
        return bool(self.vals.is_cuda)

    @property
    def rows(self) -> int:
        return int(self.shape[0])

    @property
    def cols_total(self) -> int:
        return int(self.shape[1])

    @property
    def k(self) -> int:
        return int(self.cols.size(1))

    @property
    def nnz(self) -> int:
        return int(self.cols.numel())

    @property
    def ndim(self) -> int:
        return self.dim()

    def __repr__(self) -> str:
        prefix = f", prefix_shape={self.prefix_shape}" if self.prefix_shape is not None else ""
        return (
            "SRPTensor("
            f"shape={self.shape}, "
            f"k={self.k}, "
            f"vals=Tensor(shape={tuple(self.vals.shape)}, dtype={self.vals.dtype}, device={self.vals.device}), "
            f"cols=Tensor(shape={tuple(self.cols.shape)}, dtype={self.cols.dtype}, device={self.cols.device})"
            f"{prefix}"
            ")"
        )

    def dim(self) -> int:
        return (len(self.prefix_shape) + 1) if self.prefix_shape is not None else 2

    def size(self, dim: int | None = None):
        logical_shape = (*self.prefix_shape, self.cols_total) if self.prefix_shape is not None else self.shape
        if dim is None:
            return torch.Size(logical_shape)
        return torch.Size(logical_shape)[dim]

    def numel(self) -> int:
        out = 1
        for size in self.size():
            out *= int(size)
        return int(out)

    def is_floating_point(self) -> bool:
        return bool(torch.is_floating_point(self.vals))

    def to(self, *args, **kwargs) -> "SRPTensor":
        vals = self.vals.to(*args, **kwargs)
        cols = self.cols.to(device=vals.device)
        return SRPTensor(cols=cols, vals=vals, shape=self.shape, prefix_shape=self.prefix_shape, validate=False)

    def cpu(self) -> "SRPTensor":
        return self.to("cpu")

    def cuda(self, device: int | str | torch.device | None = None) -> "SRPTensor":
        if device is None:
            return self.to("cuda")
        return self.to(torch.device("cuda", device) if isinstance(device, int) else device)

    def detach(self) -> "SRPTensor":
        return SRPTensor(
            cols=self.cols.detach(),
            vals=self.vals.detach(),
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def clone(self) -> "SRPTensor":
        return SRPTensor(
            cols=self.cols.clone(),
            vals=self.vals.clone(),
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def contiguous(self) -> "SRPTensor":
        return SRPTensor(
            cols=self.cols.contiguous(),
            vals=self.vals.contiguous(),
            shape=self.shape,
            prefix_shape=self.prefix_shape,
            validate=False,
        )

    def requires_grad_(self, requires_grad: bool = True) -> "SRPTensor":
        self.vals.requires_grad_(requires_grad)
        return self

    def to_dense(self) -> torch.Tensor:
        rows, cols_total = self.shape
        out = torch.zeros((rows, cols_total), device=self.device, dtype=self.dtype)
        out.scatter_add_(dim=1, index=self.cols, src=self.vals)
        if self.prefix_shape is not None:
            out = out.view(*self.prefix_shape, cols_total)
        return out

    def to_coo(self) -> torch.Tensor:
        row_idx = torch.arange(self.rows, device=self.cols.device, dtype=torch.long).repeat_interleave(self.k)
        indices = torch.stack((row_idx, self.cols.reshape(-1)), dim=0)
        return torch.sparse_coo_tensor(
            indices,
            self.vals.reshape(-1),
            size=self.shape,
            device=self.device,
            dtype=self.dtype,
            check_invariants=False,
        ).coalesce()

    def to_csr(self) -> torch.Tensor:
        return self.to_coo().to_sparse_csr()

    def to_csc(self) -> torch.Tensor:
        return self.to_coo().to_sparse_csc()

    def to_bsr(self, blocksize: tuple[int, int]) -> torch.Tensor:
        return self.to_coo().to_sparse_bsr(blocksize)

    def to_bsc(self, blocksize: tuple[int, int]) -> torch.Tensor:
        return self.to_coo().to_sparse_bsc(blocksize)

    def to_scipy_coo(self):
        from scipy import sparse

        row_idx = torch.arange(self.rows, device=self.cols.device, dtype=torch.long).repeat_interleave(self.k)
        return sparse.coo_matrix(
            (
                self.vals.detach().cpu().numpy().reshape(-1),
                (row_idx.detach().cpu().numpy(), self.cols.detach().cpu().numpy().reshape(-1)),
            ),
            shape=self.shape,
        )

    def to_scipy_csr(self):
        return self.to_scipy_coo().tocsr()

    def to_scipy_csc(self):
        return self.to_scipy_coo().tocsc()

    def to_numpy_dict(self) -> dict[str, Any]:
        return {
            "cols": self.cols.detach().cpu().numpy(),
            "vals": self.vals.detach().cpu().numpy(),
            "shape": self.shape,
            "prefix_shape": self.prefix_shape,
        }

    def numpy(self) -> dict[str, Any]:
        return self.to_numpy_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "layout": "srp",
            "shape": self.shape,
            "prefix_shape": self.prefix_shape,
            "cols": self.cols,
            "vals": self.vals,
        }

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

    @staticmethod
    def from_dense(
        x: torch.Tensor,
        k: int,
        *,
        score_mode: Literal["abs", "raw", "relu"] = "abs",
    ) -> "SRPTensor":
        if x.dim() < 2:
            raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}")
        cols_total = int(x.shape[-1])
        if not (1 <= int(k) <= cols_total):
            raise ValueError(f"k must be in [1, {cols_total}], got {k}")
        prefix_shape = tuple(int(d) for d in x.shape[:-1])
        rows = 1
        for d in prefix_shape:
            rows *= d
        x2d = x.reshape(rows, cols_total)
        if score_mode == "abs":
            scores = x2d.abs()
        elif score_mode == "raw":
            scores = x2d
        elif score_mode == "relu":
            scores = x2d.relu()
        else:
            raise ValueError("score_mode must be one of {'abs', 'raw', 'relu'}")
        idx = torch.topk(scores, k=int(k), dim=-1, largest=True).indices
        vals = x2d.gather(dim=-1, index=idx)
        return SRPTensor(
            cols=idx.to(torch.long),
            vals=vals,
            shape=(rows, cols_total),
            prefix_shape=prefix_shape if len(prefix_shape) > 0 else None,
        )


def load_srp_tensor(path: str | Path, map_location=None, *, validate: bool = True) -> SRPTensor:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return SRPTensor.from_dict(payload, validate=validate)
