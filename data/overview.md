# Compresso

<img class="overview-logo" src="{{ compresso_image }}" alt="Compresso logo">

Sparse representations for compact, inspectable deep learning.

Compresso is a PyTorch framework for sparse representation learning. It provides
building blocks for Top-k sparsification, sparse autoencoders, compact sparse
tensor storage, and semantic analysis of learned sparse features.

<div class="overview-logo-clear"></div>

## What this demo shows

This companion demo presents one end-to-end recommender-system workflow:

1. Product metadata is converted into dense semantic embeddings.
2. A top-*k* sparse autoencoder transforms each product into a fixed-*k* sparse representation.
3. Products with related sparse activation patterns are grouped into clusters.
4. Cluster members are aggregated and assigned human-readable labels.
5. The resulting sparse factors and representative products can be inspected across several Amazon product domains.

The displayed results were generated with Compresso and are loaded as precomputed artifacts so that the interface remains fast and responsive.

## What you can explore

- **Methodology** shows the code used for embedding preparation, sparse autoencoder training, clustering, labeling, and export.
- **Domain pages** display discovered product clusters as horizontal galleries of representative items.
- **Latent-factor details** expose the sparse feature and activation direction associated with each cluster.
- **Display controls** let you vary the number of clusters and representative products shown.

## Compresso and Compresso-Recsys

The project is split into two layers:

- **[Compresso](https://github.com/zombak79/compresso)** contains the general-purpose sparse representation framework: sparse autoencoders, `SRPTensor`, differentiable top-*k* operators, sparse and masked parameters, pruning utilities, and clustering components.
- **[Compresso-Recsys](https://github.com/zombak79/compresso-recsys)** contains recommender-specific pipelines, datasets, evaluation code, and examples built on top of the core framework.

This separation keeps Compresso lightweight and reusable beyond recommendation, while domain-specific workflows can evolve independently.

## Installation

Compresso is available from [PyPI](https://pypi.org/project/compresso-pytorch/) and can be installed with:

```bash
pip install compresso-pytorch
```

Recommender-specific addon can be installed as:

```bash
pip install compresso-recsys
```

For tutorials, API documentation, and advanced examples, see the [Compresso documentation](https://zombak79.github.io/compresso/).

## Citation

If you use Compresso in your research, please cite:

```bibtex
@software{vancura2026compresso,
  author = {Vojtěch Vančura and Giacomo Medda and Martin Spišák and Ladislav Peška},
  title = {Compresso: A PyTorch Framework for Sparse Representation Learning},
  year = {2026},
  url = {https://github.com/zombak79/compresso}
}
```
