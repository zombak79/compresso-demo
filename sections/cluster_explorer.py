from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.bundle import DEFAULT_GRAPH_STAGE, load_goodbooks_bundle
from lib.cluster_view import (
    child_cluster_table,
    cluster_features,
    cluster_items,
    cluster_title,
    root_cluster_table,
)
from ui import init_item_row_styles, render_item_row


@st.cache_resource(show_spinner="Loading GoodBooks demo bundle...")
def get_bundle(graph_stage: str) -> dict:
    return load_goodbooks_bundle(graph_stage=graph_stage)


def render_cluster_details(graph, metadata: pd.DataFrame, cluster_id: str) -> None:
    cluster = graph.cluster_by_id[cluster_id]
    title = cluster_title(cluster)

    st.subheader(title)
    st.caption(cluster.cluster_id)

    a, b, c, d = st.columns(4)
    a.metric("Items", cluster.entity_count)
    b.metric("Children", len(cluster.child_cluster_ids))
    c.metric("Parents", len(cluster.parent_cluster_ids))
    d.metric("Features", len(cluster.centroid.indices))

    if cluster.description and cluster.description != cluster.label:
        st.write(cluster.description)

    with st.expander("Sparse centroid features", expanded=False):
        st.dataframe(cluster_features(cluster, limit=20), hide_index=True, width="stretch")

    child_rows = child_cluster_table(graph, cluster.cluster_id)
    if not child_rows.empty:
        st.markdown("#### Child Clusters")
        st.dataframe(child_rows, hide_index=True, width="stretch")

    items = cluster_items(metadata, cluster, limit=80)
    st.markdown("#### Items")
    render_item_row(title, items, row_id=f"cluster_items_{abs(hash(cluster.cluster_id))}", limit=36)

    with st.expander("Item table", expanded=False):
        st.dataframe(items, hide_index=True, width="stretch")


def item_option_label(metadata: pd.DataFrame, row_idx: int) -> str:
    if row_idx < 0:
        return "No item selected"
    row = metadata.iloc[row_idx]
    authors = str(row.get("authors", "") or "")
    suffix = f" | {authors}" if authors else ""
    return f"{row_idx}: {row.get('title', '')}{suffix}"


def clusters_for_item_table(graph, entity_idx: int, *, active_only: bool = True) -> pd.DataFrame:
    mapping = graph.entity_to_cluster_ids if active_only else graph.entity_to_all_cluster_ids
    cluster_ids = mapping.get(int(entity_idx), [])
    rows = []
    by_id = graph.cluster_by_id
    active_ids = set(graph.active_cluster_ids or ())
    for cluster_id in cluster_ids:
        cluster = by_id[cluster_id]
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "label": cluster_title(cluster),
                "n_items": cluster.entity_count,
                "n_children": len(cluster.child_cluster_ids),
                "n_parents": len(cluster.parent_cluster_ids),
                "is_root": len(cluster.parent_cluster_ids) == 0,
                "is_active": cluster.cluster_id in active_ids,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["cluster_id", "label", "n_items", "n_children", "n_parents", "is_root", "is_active"])
    return pd.DataFrame(rows).sort_values(["is_root", "label", "n_items"], ascending=[False, True, False], ignore_index=True)


def filter_clusters_containing_item(clusters: pd.DataFrame, graph, entity_idx: int) -> pd.DataFrame:
    if clusters.empty or entity_idx < 0:
        return clusters
    by_id = graph.cluster_by_id
    keep = []
    for row in clusters.itertuples():
        cluster = by_id[str(row.cluster_id)]
        keep.append(int(entity_idx) in set(cluster.entity_indices.tolist()))
    return clusters.loc[keep].reset_index(drop=True)


def render() -> None:
    init_item_row_styles()

    st.title("Cluster Explorer")
    st.caption("Browse sparse GoodBooks clusters as a hierarchy of interpretable segments.")

    default_bundle = get_bundle(DEFAULT_GRAPH_STAGE)
    stages = default_bundle["graph_stages"]

    with st.sidebar:
        st.subheader("Browse Clusters")
        graph_stage = st.selectbox(
            "Graph stage",
            options=stages,
            index=stages.index(DEFAULT_GRAPH_STAGE) if DEFAULT_GRAPH_STAGE in stages else 0,
        )
        bundle = get_bundle(graph_stage)
        graph = bundle["graph"]
        metadata = bundle["metadata"]

        st.markdown("##### Optional Item Filter")
        item_options = [-1] + list(range(len(metadata)))
        selected_item_idx = st.selectbox(
            "Select item",
            options=item_options,
            format_func=lambda i: item_option_label(metadata, i),
            index=0,
            help="Choose a book to inspect all clusters that contain it.",
        )
        item_scope_all = st.checkbox("Show all graph nodes for item", value=False)
        filter_to_item = st.checkbox("Filter cluster picker to selected item", value=selected_item_idx >= 0, disabled=selected_item_idx < 0)

        min_items = st.slider("Minimum root size", min_value=1, max_value=100, value=5, step=1)
        roots = root_cluster_table(graph, min_items=min_items)
        if selected_item_idx >= 0 and filter_to_item:
            roots = filter_clusters_containing_item(roots, graph, selected_item_idx)
        st.caption(f"{len(roots)} root clusters shown from {len(graph.root_clusters)} roots.")

        if roots.empty:
            st.warning("No root clusters match the selected minimum size.")
            return

        root_labels = [f"{row.label} ({row.n_items})" for row in roots.itertuples()]
        root_choice = st.selectbox("Root cluster", options=list(range(len(roots))), format_func=lambda i: root_labels[i])
        selected_cluster_id = str(roots.iloc[root_choice]["cluster_id"])

        path = [selected_cluster_id]
        max_depth = st.slider("Drill-down depth", min_value=0, max_value=5, value=3, step=1)

        for depth in range(max_depth):
            current = graph.cluster_by_id[path[-1]]
            children = child_cluster_table(graph, current.cluster_id)
            if children.empty:
                break

            child_labels = ["Stay here"] + [f"{row.label} ({row.n_items})" for row in children.itertuples()]
            choice = st.selectbox(
                f"Level {depth + 1}",
                options=list(range(len(child_labels))),
                format_func=lambda i, labels=child_labels: labels[i],
                key=f"cluster_child_{depth}_{current.cluster_id}",
            )
            if choice == 0:
                break
            path.append(str(children.iloc[choice - 1]["cluster_id"]))

    if selected_item_idx >= 0:
        selected = metadata.iloc[[selected_item_idx]]
        render_item_row("Selected Item", selected, row_id="selected_item", limit=1)
        item_clusters = clusters_for_item_table(graph, selected_item_idx, active_only=not item_scope_all)
        st.markdown(f"#### Clusters containing `{metadata.iloc[selected_item_idx].get('title', '')}`")
        st.caption(f"Showing {'all graph nodes' if item_scope_all else 'active clusters only'}.")
        st.dataframe(item_clusters, hide_index=True, width="stretch")
        st.divider()

    breadcrumb = " / ".join(cluster_title(graph.cluster_by_id[cid]) for cid in path)
    st.markdown(f"**Path:** {breadcrumb}")
    render_cluster_details(graph, metadata, path[-1])
