from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Compresso")
    st.caption("Sparse representations for compact, inspectable deep learning.")

    st.markdown(
        """
Compresso is a PyTorch framework for sparse representation learning. It provides
building blocks for Top-k sparsification, sparse autoencoders, compact sparse
tensor storage, and semantic analysis of learned sparse features.

This site is a lightweight companion demo. Instead of reproducing the full API
reference, it shows what sparse representations can look like in a recommender
system: products are embedded, compressed into sparse codes, grouped by shared
latent factors, and labeled as human-readable segments.
"""
    )

    left, middle, right = st.columns(3)

    with left:
        st.subheader("Learn Sparse Codes")
        st.write("Dense product metadata is encoded and compressed with a Top-k sparse autoencoder.")

    with middle:
        st.subheader("Find Latent Factors")
        st.write("Items that activate the same sparse feature form compact, inspectable product segments.")

    with right:
        st.subheader("Browse the Result")
        st.write("Each example page shows labeled clusters with representative Amazon products.")

    st.divider()

    st.subheader("What to explore")
    st.markdown(
        """
- **Methodology** explains the shared pipeline used for the Amazon examples:
  checkpoint creation, sentence-transformer embeddings, sparse autoencoder
  training, clustering, labeling, and export.
- **Dataset pages** load compact zip artifacts from this repository and render
  discovered product clusters as horizontal galleries.
- **Official docs** remain the best place for installation instructions, API
  reference, and deeper examples.
"""
    )

    st.link_button("Open Compresso Docs", "https://zombak79.github.io/compresso/index.html")
