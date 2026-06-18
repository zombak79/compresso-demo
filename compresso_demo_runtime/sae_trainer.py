from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .srp import SRPTensor


def topk_ste(
    x: torch.Tensor,
    k: int,
    dim: int = -1,
    score_mode: Literal["abs", "raw", "relu"] = "abs",
    ste_alpha: float = 0.0,
) -> torch.Tensor:
    if score_mode == "abs":
        scores = x.abs()
    elif score_mode == "raw":
        scores = x
    elif score_mode == "relu":
        scores = x.relu()
    else:
        raise ValueError("score_mode must be one of {'abs', 'raw', 'relu'}")
    idx = torch.topk(scores, k=int(k), dim=dim).indices
    mask = torch.zeros_like(x, dtype=torch.bool).scatter_(dim, idx, True)
    masked = x * mask.to(dtype=x.dtype)
    if ste_alpha <= 0.0:
        return masked
    if ste_alpha > 1.0:
        raise ValueError("ste_alpha must be in [0, 1]")
    back = x * (mask.to(dtype=x.dtype) + (~mask).to(dtype=x.dtype) * float(ste_alpha))
    return back + (masked - back).detach()


class TopKSparsify(nn.Module):
    def __init__(
        self,
        k: int,
        dim: int = -1,
        score_mode: Literal["abs", "raw", "relu"] = "abs",
        ste_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.dim = int(dim)
        self.score_mode = score_mode
        self.ste_alpha = float(ste_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return topk_ste(x, self.k, self.dim, self.score_mode, self.ste_alpha)


class TopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        decoder_bias: bool = False,
        pre_act: nn.Module | None = None,
        post_sparsify: nn.Module | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        sparsify_score_mode: Literal["abs", "raw", "relu"] = "abs",
        sparsify_ste_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)
        self.encoder = encoder if encoder is not None else nn.Linear(input_dim, hidden_dim)
        self.sparsify = TopKSparsify(k, score_mode=sparsify_score_mode, ste_alpha=sparsify_ste_alpha)
        self.pre_act = pre_act
        self.post_sparsify = post_sparsify
        self.decoder = decoder if decoder is not None else nn.Linear(hidden_dim, input_dim, bias=decoder_bias)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        if self.pre_act is not None:
            h = self.pre_act(h)
        codes = self.sparsify(h)
        if self.post_sparsify is not None:
            codes = self.post_sparsify(codes)
        reconstruction = self.decoder(codes)
        stats = self._compute_stats(x, reconstruction, codes)
        return reconstruction, codes, stats

    def _compute_stats(self, x: torch.Tensor, recon: torch.Tensor, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        active_mask = codes != 0
        x_flat = x.reshape(x.shape[0], -1)
        recon_flat = recon.reshape(recon.shape[0], -1)
        diff = x_flat - recon_flat
        return {
            "active_count": active_mask.float().sum(dim=-1).mean(),
            "activation_freq": active_mask.float().mean(dim=0),
            "reconstruction_mse": (diff * diff).mean(),
            "cosine_similarity": F.cosine_similarity(x_flat, recon_flat, dim=-1).mean(),
            "dead_features": (active_mask.float().mean(dim=0) == 0).sum(),
        }


class EmbeddingsDataset:
    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        tensor = torch.as_tensor(embeddings)
        if tensor.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {tuple(tensor.shape)}")
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        self.embeddings = tensor.contiguous()
        self.n, self.dim = int(tensor.shape[0]), int(tensor.shape[1])
        self.indices = np.arange(self.n)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(idx) * self.batch_size
        end = min(start + self.batch_size, self.n)
        rows = self.indices[start:end]
        return self.embeddings[torch.as_tensor(rows, dtype=torch.long)].to(self.device)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


class L1Normalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=1.0, dim=-1)


class L2Normalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=-1)


@dataclass(frozen=True)
class TopKSAEConfig:
    hidden_dim: int = 4096
    k: int = 128
    decoder_bias: bool = False
    pre_act: nn.Module | None = None
    post_sparsify: nn.Module | None = None
    encoder: nn.Module | None = None
    decoder: nn.Module | None = None
    sparsify_score_mode: Literal["abs", "raw", "relu"] = "abs"
    sparsify_ste_alpha: float = 0.01
    alpha_loss: float = 0.01
    l1_penalty: float = 0.0
    batch_size: int = 128
    shuffle: bool = True
    seed: int = 42
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    srp_score_mode: Literal["abs", "raw", "relu"] = "abs"


class TopKSAETrainer:
    def __init__(self, config: TopKSAEConfig | None = None) -> None:
        self.cfg = config if config is not None else TopKSAEConfig()
        self.device = torch.device(self.cfg.device)
        self.sae: TopKSAE | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.input_dim: int | None = None
        self.history: list[dict[str, float]] = []

    def build(self, input_dim: int) -> "TopKSAETrainer":
        if self.sae is not None:
            return self
        torch.manual_seed(int(self.cfg.seed))
        self.input_dim = int(input_dim)
        model = TopKSAE(
            input_dim=input_dim,
            hidden_dim=self.cfg.hidden_dim,
            k=self.cfg.k,
            decoder_bias=self.cfg.decoder_bias,
            pre_act=self.cfg.pre_act,
            post_sparsify=self.cfg.post_sparsify,
            encoder=self.cfg.encoder,
            decoder=self.cfg.decoder,
            sparsify_score_mode=self.cfg.sparsify_score_mode,
            sparsify_ste_alpha=self.cfg.sparsify_ste_alpha,
        ).to(self.device)
        if self.cfg.compile:
            model = torch.compile(model)  # type: ignore[assignment]
        self.sae = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return self

    def _progress(self, iterable, *, total: int | None = None):
        if not self.cfg.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:
            return iterable
        return tqdm(iterable, total=total)

    def _set_lr(self, lr: float) -> None:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before setting lr")
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _current_lr(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before reading lr")
        return float(self.optimizer.param_groups[0]["lr"])

    def train_step(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.sae is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")
        self.sae.train()
        self.optimizer.zero_grad(set_to_none=True)
        _recon, sparse, stats = self.sae(batch)
        cosine_loss = 1.0 - stats["cosine_similarity"]
        mse = stats["reconstruction_mse"]
        loss = self.cfg.alpha_loss * cosine_loss + (1.0 - self.cfg.alpha_loss) * mse
        if self.cfg.l1_penalty > 0.0:
            loss = loss + self.cfg.l1_penalty * sparse.abs().mean()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.detach(), "cosine_loss": cosine_loss.detach(), "reconstruction_mse": mse.detach()}

    def fit(self, embeddings: np.ndarray | torch.Tensor) -> "TopKSAETrainer":
        data = EmbeddingsDataset(
            embeddings,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            device=self.device,
        )
        self.build(data.dim)
        epochs = int(self.cfg.epochs)
        self._set_lr(float(self.cfg.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs) if self.cfg.decay else None
        epoch_iter = self._progress(range(1, epochs + 1), total=epochs)
        for epoch in epoch_iter:
            sums: dict[str, float] = {}
            n_batches = 0
            for batch in data:
                stats = self.train_step(batch)
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.0) + float(value.cpu().item())
                n_batches += 1
            data.on_epoch_end()
            record = {key: value / max(1, n_batches) for key, value in sums.items()}
            record["epoch"] = float(epoch)
            record["lr"] = self._current_lr()
            self.history.append(record)
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix({"loss": f"{record['loss']:.4f}", "lr": f"{record['lr']:.2E}"})
            if scheduler is not None:
                scheduler.step()
        return self

    @torch.no_grad()
    def encode(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.sae is None:
            raise RuntimeError("trainer must be fitted before encode")
        data = EmbeddingsDataset(embeddings, batch_size=self.cfg.batch_size, shuffle=False, device=self.device)
        self.sae.eval()
        codes = []
        for batch in self._progress(data, total=len(data)):
            _recon, sparse, _stats = self.sae(batch)
            codes.append(sparse.detach().cpu())
        return torch.cat(codes, dim=0)

    @torch.no_grad()
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor:
        codes = self.encode(embeddings)
        return SRPTensor.from_dense(codes, k=self.cfg.k, score_mode=self.cfg.srp_score_mode)

    def fit_transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor:
        self.fit(embeddings)
        return self.transform(embeddings)
