from __future__ import annotations

import streamlit as st


def render_signature(code: str) -> None:
    st.code(code.strip(), language="python")


def render() -> None:
    st.title("Sparsify")
    st.caption("Core sparse building blocks used throughout Compresso.")

    st.markdown(
        """
Compresso has two related but different sparse-learning paths:

- **Sparse activations / embeddings**: produce sparse vectors, often exported as `SRPTensor`.
- **Sparse parameters**: learn model weights under a sparse mask, often with pruning/rewind schedules.

The objects below are the main low-level pieces used by higher-level models and examples.
"""
    )

    st.subheader("topk_ste")
    st.markdown("Hard Top-k sparsification with optional straight-through gradients.")
    render_signature(
        """
def topk_ste(
    x: torch.Tensor,
    k: int,
    dim: int = -1,
    score_mode: str = "abs",
    ste_alpha: float = 0.0,
) -> torch.Tensor
"""
    )

    st.markdown(
        """
**Parameters**

- `x`: input tensor.
- `k`: number of values to keep along `dim`.
- `dim`: dimension along which Top-k is selected. Default is the last dimension.
- `score_mode`: ranking rule used to choose Top-k positions:
  - `abs`: choose largest `abs(x)`, but keep original signed values.
  - `raw`: choose largest raw values, so negative values are usually disfavored.
  - `relu`: choose largest positive values after `relu`; negative values score as zero.
- `ste_alpha`: gradient scale for non-selected positions. Must be in `[0, 1]`.
  - `0.0`: gradients flow only through selected Top-k positions.
  - `0.01`: selected positions get full gradient, non-selected positions get weak gradient.
  - `1.0`: dense identity-style backward path while forward remains sparse.

**Output**

Returns a tensor with the same shape as `x`. Forward values are hard sparse:
only `k` entries per slice are non-zero.
"""
    )

    st.markdown("**Example**")
    render_signature(
        """
import torch
from compresso import topk_ste

x = torch.tensor([[0.1, -0.9, 0.3, 0.2]])
y = topk_ste(x, k=2, score_mode="abs", ste_alpha=0.01)

print(y)
# tensor([[ 0.0000, -0.9000,  0.3000,  0.0000]])
"""
    )

    st.subheader("TopKSparsify")
    st.markdown("`torch.nn.Module` wrapper around `topk_ste`.")
    render_signature(
        """
class TopKSparsify(torch.nn.Module):
    def __init__(
        self,
        k: int,
        dim: int = -1,
        score_mode: str = "abs",
        ste_alpha: float = 0.0,
    ) -> None
"""
    )

    st.markdown(
        """
**Constructor Parameters**

- `k`: number of active values to keep along `dim`.
- `dim`: dimension along which Top-k is selected.
- `score_mode`: one of `abs`, `raw`, or `relu`; same semantics as `topk_ste`.
- `ste_alpha`: gradient scale for non-selected values.

**Important Methods**

- `forward(x)`: returns `topk_ste(x, self.k, self.dim, self.score_mode, self.ste_alpha)`.
- `set_k(k)`: changes `k` at runtime. Useful for schedules or ablations.
- `extra_repr()`: returns a concise module description for `print(module)`.
"""
    )

    st.markdown("**Example**")
    render_signature(
        """
import torch
from compresso import TopKSparsify

sparsify = TopKSparsify(k=2, score_mode="abs", ste_alpha=0.01)
x = torch.tensor([[1.0, -4.0, 2.0, 0.5]])

print(sparsify(x))
# tensor([[ 0., -4.,  2.,  0.]])

sparsify.set_k(1)
print(sparsify(x))
# tensor([[ 0., -4.,  0.,  0.]])
"""
    )

    st.subheader("SRPTensor")
    st.markdown("Fixed-k row-packed sparse tensor for sparse activations or embeddings.")
    render_signature(
        """
class SRPTensor:
    def __init__(
        self,
        *,
        cols: torch.Tensor,          # shape: (rows, k), dtype: torch.long
        vals: torch.Tensor,          # shape: (rows, k)
        shape: tuple[int, int],      # (rows, cols_total)
        prefix_shape: tuple[int, ...] | None = None,
        validate: bool = True,
    ) -> None
"""
    )

    st.markdown(
        """
**Constructor Parameters**

- `cols`: integer column indices, shape `(rows, k)`, dtype `torch.long`.
- `vals`: stored values, shape `(rows, k)`. Same row/slot layout as `cols`.
- `shape`: logical dense matrix shape `(rows, cols_total)`.
- `prefix_shape`: optional original prefix shape. For example, dense input `(batch, time, dim)`
  can be flattened to `(batch * time, dim)` while remembering `(batch, time)`.
- `validate`: checks dtype, rank, shape compatibility, and column bounds.

**Properties**

- `rows`: number of rows.
- `cols_total`: logical dense width.
- `k`: stored non-zero slots per row.
- `device`: device of `vals`.
- `dtype`: dtype of `vals`.
- `nnz`: number of stored SRP slots, equal to `rows * k`.
- `requires_grad`: whether `vals` requires gradients.

**Important Methods**

- `to_dense()`: materializes a dense tensor. Duplicate columns within a row are accumulated.
- `to(...)`, `cpu()`, `cuda(...)`: move/cast the SRP tensor while keeping `cols` as `torch.long`.
- `detach()`, `clone()`, `contiguous()`, `requires_grad_(...)`: tensor-like utility methods.
- `size(dim=None)`, `dim()`, `numel()`, `is_floating_point()`: tensor-like metadata helpers.
- `to_coo()`, `to_csr()`, `to_csc()`: convert to PyTorch sparse layouts.
- `to_bsr(blocksize)`, `to_bsc(blocksize)`: convert to PyTorch block sparse layouts. Block size is explicit because `(1, 1)` is rarely useful.
- `to_scipy_coo()`, `to_scipy_csr()`, `to_scipy_csc()`: convert to SciPy sparse matrices.
- `to_numpy_dict()` / `numpy()`: return structural arrays `{cols, vals, shape, prefix_shape}`.
- `to_dict()`: serializes into a `torch.save`-compatible payload.
- `from_dict(payload, validate=True)`: restores an `SRPTensor`.
- `from_dense(x, k, score_mode="abs")`: projects dense `x` to fixed-k SRP.
"""
    )

    st.markdown("**Example**")
    render_signature(
        """
import torch
from compresso import SRPTensor

x = torch.tensor([
    [0.1, -0.9, 0.3, 0.2],
    [2.0,  0.1, 0.0, -3.0],
])

srp = SRPTensor.from_dense(x, k=2, score_mode="abs")

print(srp.cols)
# tensor([[1, 2],
#         [3, 0]])

print(srp.vals)
# tensor([[-0.9000,  0.3000],
#         [-3.0000,  2.0000]])

print(srp.to_dense())
# tensor([[ 0.0000, -0.9000,  0.3000,  0.0000],
#         [ 2.0000,  0.0000,  0.0000, -3.0000]])

print(srp)
# SRPTensor(shape=(2, 4), k=2, vals=Tensor(...), cols=Tensor(...))

coo = srp.to_coo()
csr = srp.to_csr()
scipy_csr = srp.to_scipy_csr()

print(coo.layout)
# torch.sparse_coo
"""
    )

    st.subheader("MaskedParam")
    st.markdown("Dense trainable parameter with Top-k sparse masking and pruning/rewind state.")
    render_signature(
        """
class MaskedParam(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        k_target: int,
        k_schedule: Sequence[int] | None = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        sparsity: Literal["row", "col"] = "row",
        allow_regrowth: bool = True,
        score_mode: Literal["abs", "raw", "relu"] = "abs",
        ste_alpha: float = 1.0,
        post_norm_l1: bool = False,
    ) -> None
"""
    )

    st.markdown(
        """
**Constructor Parameters**

- `weight`: initial dense 2D tensor. Internally copied into trainable `self.weight`.
- `k_target`: final number of active entries per row/column.
- `k_schedule`: optional explicit non-increasing list of `k` values. Last value must equal `k_target`.
- `num_stages`: number of pruning stages when `k_schedule` is not provided.
- `stability_window`: number of recent mask-change measurements used for stability.
- `change_threshold`: average mask-change threshold below which a stage is considered stable.
- `sparsity`: `row` keeps `k` per row; `col` keeps `k` per column.
- `allow_regrowth`: currently retained for API compatibility/state, but the current forward path uses Top-k selection.
- `score_mode`: Top-k score mode: `abs`, `raw`, or `relu`.
- `ste_alpha`: Top-k backward gradient scale for non-selected weights. Default `1.0` gives dense-style gradient flow.
- `post_norm_l1`: if true, applies `torch.nn.functional.normalize(..., p=1.0)` after masking.

**Important Attributes**

- `weight`: trainable dense parameter.
- `mask`: frozen/current binary mask buffer.
- `k_current`: current active count.
- `k_next`: next scheduled active count.
- `stage_idx`: current pruning stage.
- `mask_frozen`: if true, forward uses the stored mask directly.

**Important Methods**

- `forward()`: returns masked/sparsified dense weight.
- `topk_weights(k=None)`: returns Top-k sparse dense weight.
- `topk_mask(k=None)`: returns boolean mask for current or requested `k`.
- `step_mask()`: updates stability statistics and may mark the current stage completed.
- `rewind()`: rewinds weights to initialization under the chosen mask and advances stage if stable.
- `freeze_mask()`: freezes current Top-k mask for deterministic masked forward.
- `srp()`: exports row-wise dense parameter to `SRPTensor` using current `k`.
- `maskedparam_to_srp()`: exports row-wise sparse parameter as `SRPParam`.
- `maskedparam_to_coo()`: exports sparse parameter as packed COO.
- `get_stats()`: returns pruning/schedule state as a dictionary.
"""
    )

    st.markdown("**Example**")
    render_signature(
        """
import torch
from compresso import MaskedParam

weight = torch.tensor([
    [1.0, -4.0, 2.0, 0.5],
    [0.2,  0.1, 3.0, -2.0],
])

param = MaskedParam(
    weight,
    k_target=2,
    k_schedule=[4, 2],
    sparsity="row",
    score_mode="abs",
    ste_alpha=0.01,
)

print(param())       # stage starts dense because k_current = 4
# tensor([[ 1.0000, -4.0000,  2.0000,  0.5000],
#         [ 0.2000,  0.1000,  3.0000, -2.0000]], grad_fn=...)

param.freeze_mask()
print(param())       # now frozen to current mask
# tensor([[ 1.0000, -4.0000,  2.0000,  0.5000],
#         [ 0.2000,  0.1000,  3.0000, -2.0000]], grad_fn=...)

print(param.get_stats()["k_current"])
# 4
"""
    )

    st.info(
        "For direct sparse-parameter learning, MaskedParam is usually driven by a controller/training loop "
        "that calls step_mask(), rewind(), and eventually freeze_mask()."
    )

    st.subheader("SRPParam")
    st.markdown("Trainable fixed-k row-packed sparse parameter.")
    render_signature(
        """
class SRPParam(torch.nn.Module):
    def __init__(
        self,
        cols: torch.Tensor,          # shape: (rows, k), dtype: torch.long
        values: torch.Tensor,        # shape: (rows, k)
        shape: tuple[int, int],      # (rows, cols_total)
        *,
        validate: bool = True,
    ) -> None
"""
    )

    st.markdown(
        """
**Constructor Parameters**

- `cols`: fixed sparse structure, shape `(rows, k)`, dtype `torch.long`.
- `values`: trainable values for each `(row, slot)`, shape `(rows, k)`.
- `shape`: logical dense matrix shape `(rows, cols_total)`.
- `validate`: checks dtype, shape, and column bounds.

**Important Properties**

- `shape`: returns `(rows, cols_total)`.
- `rows`, `cols_total`, `k`: stored dimensions.
- `values`: trainable `torch.nn.Parameter`.
- `cols`: fixed registered buffer.

**Important Methods**

- `forward()`: returns an `SRPTensor` view of the current sparse parameter.
- `to_dense()`: materializes a dense matrix with scatter-add semantics.
- `build_coo(dtype=None, coalesce=True)`: converts to `torch.sparse_coo_tensor`.
- `select_rows(row_indices)`: returns a new `SRPParam` containing selected rows.
- `from_dense(A, k, mode="topk_abs")`: constructs from a dense 2D matrix.
- `from_sparse_coo(sp, k=None, require_row_packed_fixed_k=True)`: constructs from sparse COO.
"""
    )

    st.markdown("**Example**")
    render_signature(
        """
import torch
from compresso import SRPParam

A = torch.tensor([
    [1.0, -4.0, 2.0, 0.5],
    [0.2,  0.1, 3.0, -2.0],
])

param = SRPParam.from_dense(A, k=2, mode="topk_abs")

print(param.cols)
# tensor([[1, 2],
#         [2, 3]])

print(param.to_dense())
# tensor([[ 0., -4.,  2.,  0.],
#         [ 0.,  0.,  3., -2.]], grad_fn=...)

srp = param()
print(srp.shape, srp.k)
# (2, 4) 2
"""
    )
