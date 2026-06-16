from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("GoodBooks Pipeline")
    st.caption("How the bundled sparse GoodBooks cluster graph was produced.")

    st.markdown(
        """
The demo bundle is an exported artifact, not a live training pipeline. The
high-level process was:

1. Build a GoodBooks data split and metadata table.
2. Encode item metadata into semantic embeddings using an SBERT-style model.
3. Train a Top-k sparse autoencoder on those semantic embeddings.
4. Export sparse item codes as an `SRPTensor`.
5. Run a clustering pipeline over the SRP representations.
6. Save the graph and sparse embeddings into a small standalone zip bundle.
"""
    )

    st.subheader("Artifact Boundary")
    st.markdown(
        """
This public Streamlit app only loads the exported artifact:

```text
data/goodbooks/goodbooks_demo.zip
  manifest.json
  data/entity_metadata.csv
  data/split.npz
  sbert_sae/sparse_embeddings.srp.pt
  semantic_clustering_merged/graph.json
  topm_clustering_merged/graph.json
```

The full training code lives in Compresso/research examples. The public demo
uses a small local runtime only to read the saved artifact.
"""
    )

    st.subheader("Clustering Pipeline")
    st.markdown("The primary graph was produced with a pipeline in this style:")
    st.code(
        """
graph = cc.ClusteringPipeline(
    [
        cc.SRPSimilarityClustering(
            threshold=0.45,
            top_k=100,
            min_cluster_size=5,
            normalize_rows=True,
            min_local_density=None,
            centroid_top_k=4,
            batch_size=32,
        ),
        cc.EntityContainmentLink(
            threshold=1.0,
            child_scope="leaves",
            parent_scope="all",
        ),
        cc.FeatureContainmentLink(
            threshold=1.0,
            child_scope="leaves",
            parent_scope="all",
        ),
        cc.MaterializeLinkMerges(parent_scope="active"),
        cc.PruneRedundantRoots(),
        cc.LabelClusters(
            entity_metadata=metadata,
            text_extractor=item_texts,
            label_fn=label_cluster,
            cluster_scope="active",
        ),
        cc.SemanticSimilarityMerge(
            embed_fn=embed_fn,
            threshold=0.95,
            label_fn=label_cluster_group,
            label_text_fn=semantic_parent_text,
            cluster_scope="active",
            max_rounds=10,
        ),
    ]
).fit(srp)
""".strip(),
        language="python",
    )

    st.markdown(
        """
The important idea is that sparse codes are not only compressed embeddings.
They also become a substrate for browsing, naming, merging, and recommending
clusters.
"""
    )
