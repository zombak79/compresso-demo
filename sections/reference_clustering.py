from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Clustering")
    st.caption("Turning sparse representations into cluster graphs.")

    st.markdown(
        """
Compresso clustering treats sparse representations as interpretable handles.
Clusters are graph nodes with:

- member entities
- a sparse centroid in the same feature space
- optional labels/tags/descriptions
- parent and child links

The result is a `SparseClusterSet`: not just a flat clustering, but a graph that
can preserve hierarchy and overlapping structure.
"""
    )

    st.subheader("Core Data Model")
    st.markdown(
        """
- `SparseCluster`: one cluster node, with `entity_indices`, sparse centroid, label, parents, and children
- `SparseClusterSet`: a graph-like container of clusters
- `active_clusters`: clusters selected as the current visible/end-user set
- `root_clusters`: clusters without parents
- `centroid`: sparse feature vector positioning the cluster in SRP space
"""
    )

    st.subheader("Cluster Builders")
    st.markdown(
        """
- `TopMSignedClustering`: creates feature-driven clusters from top signed activations
- `ComboSignedClustering`: creates clusters from combinations of signed features
- `FeaturePathClustering`: recursively splits clusters by feature paths
- `SRPSimilarityClustering`: builds clusters from similarity neighborhoods in sparse representation space
"""
    )

    st.subheader("Linking and Merging")
    st.markdown(
        """
- `EntityContainmentLink`: links clusters when one entity set is contained in another
- `FeatureContainmentLink`: links clusters when centroid feature support is contained in another
- `MaterializeLinkMerges`: creates non-destructive parent clusters from links
- `PruneRedundantRoots`: hides roots already represented by a larger branch
- `LabelDuplicateMerge`: groups clusters with duplicate labels
- `CompactHiddenClusters`: removes hidden duplicate/intermediate nodes for cleaner display
- `SemanticSimilarityMerge`: creates parent clusters for semantically similar labels
"""
    )

    st.subheader("Annotation")
    st.markdown(
        """
Cluster naming and semantic merging are intentionally callback-based. Users can
provide their own functions for:

- extracting text from cluster members
- calling an LLM or local model to name clusters
- embedding labels/descriptions for semantic similarity

That keeps Compresso independent of API keys, model providers, and domain-specific prompts.
"""
    )
