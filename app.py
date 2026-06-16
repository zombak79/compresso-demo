from __future__ import annotations

import html
import pandas as pd
import streamlit as st

from lib.bundle import DEFAULT_GRAPH_STAGE, load_goodbooks_bundle
from lib.cluster_view import (
    child_cluster_table,
    cluster_features,
    cluster_items,
    cluster_title,
    item_card_rows,
    root_cluster_table,
)


st.set_page_config(
    page_title="Compresso Demo",
    page_icon=":",
    layout="wide",
)


PAGES = [
    "Overview",
    "Cluster Explorer",
    "User Recommendations",
    "Model Walkthroughs",
]


@st.cache_resource(show_spinner="Loading GoodBooks demo bundle...")
def get_bundle(graph_stage: str) -> dict:
    return load_goodbooks_bundle(graph_stage=graph_stage)


def init_item_row_styles() -> None:
    st.markdown(
        """
        <style>
        .item-row {
            margin: 0.25rem 0 1.25rem 0;
        }
        .item-row-title {
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 0.65rem 0;
        }
        .item-rail {
            display: flex;
            gap: 0.85rem;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0.15rem 0.1rem 0.75rem 0.1rem;
            scroll-snap-type: x mandatory;
        }
        .item-card {
            flex: 0 0 auto;
            width: 170px;
            min-height: 292px;
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 14px;
            background: #ffffff;
            color: #1f2933;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
            overflow: hidden;
            scroll-snap-align: start;
        }
        .item-card img {
            width: 100%;
            height: 214px;
            object-fit: cover;
            background: #eef2f7;
            display: block;
        }
        .item-card-body {
            padding: 0.6rem 0.7rem 0.7rem 0.7rem;
        }
        .item-card-title {
            font-size: 0.86rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.25rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .item-card-authors {
            font-size: 0.74rem;
            line-height: 1.2;
            color: #667085;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .muted-small {
            color: #667085;
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_item_row(title: str, items: pd.DataFrame, *, row_id: str, limit: int = 24) -> None:
    cards = []
    for item in item_card_rows(items, limit=limit):
        item_title = html.escape(item["title"])
        authors = html.escape(item["authors"])
        image = html.escape(item["image"])
        item_id = html.escape(item["id"])
        fallback = "https://placehold.co/340x428/eef2f7/64748b?text=No+cover"
        cards.append(
            '<div class="item-card" title="{}">'
            '<img src="{}" alt="{}">'
            '<div class="item-card-body">'
            '<div class="item-card-title">{}</div>'
            '<div class="item-card-authors">{}</div>'
            '<div class="muted-small">#{}</div>'
            "</div>"
            "</div>".format(item_title, image or fallback, item_title, item_title, authors, item_id)
        )
    row_html = (
        f'<div class="item-row">'
        f'<div class="item-row-title">{html.escape(title)}</div>'
        f'<div class="item-rail" id="{html.escape(row_id)}">'
        f'{"".join(cards)}'
        f"</div>"
        f"</div>"
    )
    st.markdown(row_html, unsafe_allow_html=True)


def render_overview() -> None:
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
                "is_active": cluster.cluster_id in set(graph.active_cluster_ids or ()),
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


def render_cluster_explorer() -> None:
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


def render_placeholder(page: str) -> None:
    st.title(page)
    st.info("This page is coming next. For now, try the overview or cluster explorer.")


with st.sidebar:
    st.title("Compresso")
    st.caption("Demo navigation")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("More links will land here later.")


if page == "Overview":
    render_overview()
elif page == "Cluster Explorer":
    render_cluster_explorer()
else:
    render_placeholder(page)
