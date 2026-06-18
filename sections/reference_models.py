from __future__ import annotations

import streamlit as st


def render_signature(code: str) -> None:
    st.code(code.strip(), language="python")


def render() -> None:
    st.title("Models")
    st.caption("Model-level abstractions built from the sparse primitives.")

    st.markdown(
        """
The current core model reference starts with `TopKSAE`: a sparse autoencoder
with a hard Top-k bottleneck. It is intentionally configurable: the default is a
simple linear SAE, but users can provide custom encoder and decoder modules.
"""
    )

    st.subheader("TopKSAE")
    st.markdown("Sparse autoencoder with a Top-k sparse code layer.")
    render_signature(
        """
class TopKSAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        tied: bool = False,
        decoder_bias: bool = False,
        pre_act: torch.nn.Module | None = None,
        post_sparsify: torch.nn.Module | None = None,
        encoder: torch.nn.Module | None = None,
        decoder: torch.nn.Module | None = None,
        sparsify_score_mode: str = "abs",
        sparsify_ste_alpha: float = 0.0,
    ) -> None
"""
    )

    st.markdown(
        """
**Constructor Parameters**

- `input_dim`: dimensionality of the input and reconstruction.
- `hidden_dim`: width of the sparse code layer.
- `k`: number of active sparse code features per sample.
- `tied`: if `True`, the decoder uses the transpose of the encoder weight.
- `decoder_bias`: whether the untied/default decoder has a bias. Default is `False`.
- `pre_act`: optional module applied after encoder and before Top-k sparsification.
- `post_sparsify`: optional module applied after Top-k sparse code selection.
- `encoder`: optional custom encoder module. If omitted, uses `nn.Linear(input_dim, hidden_dim)`.
- `decoder`: optional custom decoder module. If omitted and `tied=False`, uses
  `nn.Linear(hidden_dim, input_dim, bias=decoder_bias)`.
- `sparsify_score_mode`: Top-k score mode passed to `TopKSparsify`: `abs`, `raw`, or `relu`.
- `sparsify_ste_alpha`: gradient scale for non-selected sparse-code positions.

**Important Constraints**

- `tied=True` cannot be combined with a custom `decoder`.
- If custom modules are used, the encoder must output shape `(batch, hidden_dim)`.
- The decoder output must match the per-sample flattened shape of the input for stats computation.
"""
    )

    st.markdown("**Forward Method**")
    render_signature(
        """
def forward(
    self,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
"""
    )
    st.markdown(
        """
`forward(x)` returns three objects:

- `reconstruction`: reconstructed input. For default linear SAE, shape `(batch, input_dim)`.
- `codes`: sparse hidden code, shape `(batch, hidden_dim)`, exactly `k` non-zero values per row.
- `stats`: dictionary with reconstruction and sparsity diagnostics.

`stats` contains:

- `active_count`: average number of active code features per sample.
- `activation_freq`: vector of feature activation frequencies, shape `(hidden_dim,)`.
- `reconstruction_mse`: mean squared reconstruction error.
- `cosine_similarity`: mean cosine similarity between input and reconstruction.
- `dead_features`: number of hidden features with zero activation frequency in this batch.
"""
    )

    st.markdown("**Other Important Methods / Attributes**")
    st.markdown(
        """
- `get_decoder_weight()`: returns the effective decoder weight matrix.
  - tied: `encoder.weight.t()`, shape `(input_dim, hidden_dim)`
  - untied: `decoder.weight`
- `encoder`: encoder module.
- `decoder`: decoder module when `tied=False`.
- `sparsify`: internal `TopKSparsify` module.
- `pre_act`: optional pre-sparsification module.
- `post_sparsify`: optional post-sparsification module.
"""
    )

    st.subheader("Default Linear SAE Example")
    st.markdown("A minimal dense-to-sparse-to-dense autoencoder.")
    render_signature(
        """
import torch
from compresso import TopKSAE

model = TopKSAE(
    input_dim=32,
    hidden_dim=64,
    k=8,
    sparsify_score_mode="abs",
    sparsify_ste_alpha=0.01,
)

x = torch.randn(4, 32)
recon, codes, stats = model(x)

print(recon.shape)
# torch.Size([4, 32])

print(codes.shape)
# torch.Size([4, 64])

print((codes != 0).sum(dim=-1))
# tensor([8, 8, 8, 8])

print(stats.keys())
# dict_keys([
#   'active_count',
#   'activation_freq',
#   'reconstruction_mse',
#   'cosine_similarity',
#   'dead_features',
# ])
"""
    )

    st.subheader("Tied Decoder Example")
    st.markdown(
        """
With `tied=True`, the decoder uses the encoder weight transposed. This reduces
parameters and makes the effective decoder weight directly coupled to the encoder.
"""
    )
    render_signature(
        """
import torch
from compresso import TopKSAE

model = TopKSAE(input_dim=32, hidden_dim=64, k=8, tied=True)

decoder_weight = model.get_decoder_weight()

print(model.encoder.weight.shape)
# torch.Size([64, 32])

print(decoder_weight.shape)
# torch.Size([32, 64])

print(torch.equal(decoder_weight, model.encoder.weight.t()))
# True
"""
    )

    st.subheader("Activation Hooks Example")
    st.markdown(
        """
`pre_act` and `post_sparsify` allow experimentation without rewriting the model.
For example, a ReLU pre-activation makes selected codes non-negative; an L1
normalization post-hook can force every sparse code row to sum to one in L1 norm.
"""
    )
    render_signature(
        """
import torch
import torch.nn as nn
import torch.nn.functional as F
from compresso import TopKSAE

class L1Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=1.0, dim=-1)

model = TopKSAE(
    input_dim=32,
    hidden_dim=64,
    k=8,
    pre_act=nn.ReLU(),
    post_sparsify=L1Normalize(),
)

x = torch.randn(4, 32)
_, codes, _ = model(x)

print((codes >= 0).all())
# tensor(True)

print(codes.abs().sum(dim=-1))
# tensor([1., 1., 1., 1.], grad_fn=...)
"""
    )

    st.subheader("Custom Encoder / Decoder Example")
    st.markdown(
        """
Custom modules are useful when the input is not already flat, or when the SAE
should reuse a domain-specific encoder/decoder. The only requirement is that
the encoder output has width `hidden_dim`, and the decoder reconstructs the
original per-sample shape.
"""
    )
    render_signature(
        """
import torch
import torch.nn as nn
from compresso import TopKSAE

encoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
)

decoder = nn.Sequential(
    nn.Linear(128, 28 * 28),
    nn.Unflatten(1, (28, 28)),
)

model = TopKSAE(
    input_dim=28 * 28,
    hidden_dim=128,
    k=16,
    encoder=encoder,
    decoder=decoder,
)

x = torch.randn(4, 28, 28)
recon, codes, stats = model(x)

print(recon.shape)
# torch.Size([4, 28, 28])

print(codes.shape)
# torch.Size([4, 128])

print((codes != 0).sum(dim=-1))
# tensor([16, 16, 16, 16])
"""
    )

    st.subheader("Training Pattern")
    st.markdown(
        """
`TopKSAE` does not impose a loss function. In practice, use either the provided
`stats["reconstruction_mse"]`, a cosine reconstruction loss, or a domain-specific
objective. The model is an ordinary `torch.nn.Module`, so standard optimizers and
serialization work.
"""
    )
    render_signature(
        """
import torch
from compresso import TopKSAE

model = TopKSAE(input_dim=512, hidden_dim=4096, k=128)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.randn(256, 512)

for step in range(100):
    recon, codes, stats = model(x)
    loss = stats["reconstruction_mse"]

    opt.zero_grad()
    loss.backward()
    opt.step()

print(float(loss))
# Example output: 0.73
"""
    )

    st.subheader("Exporting Sparse Codes")
    st.markdown(
        """
After training, codes can be exported to `SRPTensor` for compact storage,
retrieval experiments, or clustering. `TopKSAE` itself returns dense sparse-code
tensors; `SRPTensor.from_dense(...)` converts those fixed-k codes into row-packed
storage.
"""
    )
    render_signature(
        """
import torch
from compresso import SRPTensor, TopKSAE

model = TopKSAE(input_dim=512, hidden_dim=4096, k=128)
x = torch.randn(1000, 512)

with torch.no_grad():
    _, codes, _ = model(x)
    srp = SRPTensor.from_dense(codes, k=128, score_mode="abs")

print(srp.shape)
# (1000, 4096)

print(srp.k)
# 128

print(srp.cols.shape, srp.vals.shape)
# torch.Size([1000, 128]) torch.Size([1000, 128])
"""
    )

    st.subheader("TopKSAETrainer")
    st.markdown(
        """
`TopKSAETrainer` is a compact, sklearn-style helper for the common workflow:
fit an SAE on a dense embedding matrix, then export sparse codes as an
`SRPTensor`. It is intentionally thin: the underlying model is still `TopKSAE`,
but the trainer handles batching, optimizer setup, optional cosine learning-rate
decay, progress reporting, and SRP export.
"""
    )
    render_signature(
        """
@dataclass(frozen=True)
class TopKSAEConfig:
    hidden_dim: int = 4096
    k: int = 128
    decoder_bias: bool = False
    pre_act: torch.nn.Module | None = None
    post_sparsify: torch.nn.Module | None = None
    encoder: torch.nn.Module | None = None
    decoder: torch.nn.Module | None = None
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
    def fit(self, embeddings: np.ndarray | torch.Tensor) -> TopKSAETrainer
    def encode(self, embeddings: np.ndarray | torch.Tensor) -> torch.Tensor
    def transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor
    def fit_transform(self, embeddings: np.ndarray | torch.Tensor) -> SRPTensor
"""
    )

    st.markdown(
        """
**Configuration Parameters**

- `hidden_dim`: sparse code width.
- `k`: number of non-zero sparse features exported per row.
- `decoder_bias`: whether the default decoder has a bias.
- `pre_act`: optional module applied before Top-k selection.
- `post_sparsify`: optional module applied after Top-k selection, for example L1 normalization.
- `encoder`: optional custom encoder. If omitted, uses a linear encoder.
- `decoder`: optional custom decoder. If omitted, uses a linear decoder.
- `sparsify_score_mode`: score used to choose Top-k features: `abs`, `raw`, or `relu`.
- `sparsify_ste_alpha`: gradient scale for non-selected features. `0.0` means no gradient outside Top-k.
- `alpha_loss`: blends cosine and MSE losses as `alpha_loss * cosine_loss + (1 - alpha_loss) * mse`.
- `l1_penalty`: optional L1 penalty on sparse codes.
- `batch_size`: number of embedding rows per training batch.
- `shuffle`: whether to shuffle rows between epochs.
- `seed`: random seed for initialization and shuffling.
- `epochs`: number of training epochs.
- `lr`: AdamW learning rate.
- `weight_decay`: AdamW weight decay.
- `decay`: if `True`, use cosine learning-rate decay from `lr` across training.
- `compile`: if `True`, wraps the model with `torch.compile`.
- `device`: training device, for example `cpu`, `cuda`, or `mps`.
- `show_progress`: whether to show a tqdm progress bar.
- `srp_score_mode`: score used by `SRPTensor.from_dense(...)` during export.

**Important Methods**

- `fit(embeddings)`: train the SAE and keep the fitted model inside the trainer.
- `encode(embeddings)`: return dense sparse-code tensors from the encoder and Top-k layer.
- `transform(embeddings)`: encode embeddings and return an `SRPTensor`.
- `fit_transform(embeddings)`: train and export sparse codes in one call.
"""
    )

    st.markdown("**Example: Train and Export SRP Codes**")
    render_signature(
        """
import numpy as np
import torch.nn.functional as F
from compresso import TopKSAEConfig, TopKSAETrainer

class L1Normalize(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x, p=1.0, dim=-1)

A = np.random.randn(10_000, 768).astype("float32")

model = TopKSAETrainer(
    TopKSAEConfig(
        hidden_dim=4096,
        k=32,
        batch_size=1024,
        epochs=300,
        post_sparsify=L1Normalize(),
        sparsify_score_mode="abs",
        sparsify_ste_alpha=0.01,
        lr=1e-3,
        decay=True,
        device="cpu",
        compile=False,
    )
)

srp = model.fit_transform(A)

print(srp)
# SRPTensor(shape=(10000, 4096), k=32, vals=..., cols=...)

print(srp.to_csr().shape)
# torch.Size([10000, 4096])
"""
    )
