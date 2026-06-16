from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Compresso")
    st.caption("Sparse representations for compact, inspectable deep learning.")

    st.markdown(
        """
Compresso is a small research-oriented library for building and analyzing sparse neural
representations. It focuses on practical sparse learning tools: Top-k sparsification,
sparse autoencoders, sparse parameter learning, SRP tensors, and clustering methods for
understanding what sparse features represent.

This demo will grow into an interactive guide. The first milestone is cluster exploration:
load sparse entity representations, browse discovered clusters, inspect their items, and
compare different clustering strategies.
"""
    )

    left, middle, right = st.columns(3)

    with left:
        st.subheader("Sparse Learning")
        st.write("Train or apply Top-k sparse representations, sparse autoencoders, and masked sparse parameters.")

    with middle:
        st.subheader("Compact Storage")
        st.write("Store sparse activations and embeddings with SRP tensors, then use them for lightweight inference.")

    with right:
        st.subheader("Interpretability")
        st.write("Cluster sparse representations into human-readable segments and explore the hierarchy.")

    st.divider()

    st.subheader("What is coming next")
    st.markdown(
        """
- Cluster explorer for the GoodBooks demo bundle
- User-based cluster recommendations from sparse item representations
- Short walkthroughs for SAE training, compressed ELSA, and SRP inference
- Links to documentation, paper notes, and source code
"""
    )

    st.info("Links and additional pages will be added as the demo structure stabilizes.")
