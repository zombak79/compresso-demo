from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from lib.example_bundles import ExampleBundleInfo, load_example_bundle, markdown_without_title
from ui import init_product_gallery_styles, render_product_cluster_row


@st.cache_data(show_spinner="Loading example dataset...")
def _load_bundle(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    _ = (mtime_ns, size)
    return load_example_bundle(path)


def _cluster_indices(value: object) -> list[int]:
    try:
        return [int(v) for v in list(value)]
    except (TypeError, ValueError):
        return []


def _reliable_rating_threshold(rows: pd.DataFrame) -> int | None:
    if "rating_number" not in rows.columns or rows.empty:
        return None
    counts = pd.to_numeric(rows["rating_number"], errors="coerce").dropna()
    if counts.empty:
        return None
    return max(50, int(counts.quantile(0.25)))


def _cluster_items(metadata: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    valid = [idx for idx in indices if 0 <= idx < len(metadata)]
    rows = metadata.iloc[valid].copy()
    threshold = _reliable_rating_threshold(rows)
    if threshold is not None:
        counts = pd.to_numeric(rows["rating_number"], errors="coerce")
        filtered = rows.loc[counts >= threshold]
        if not filtered.empty:
            rows = filtered
    sort_columns = [col for col in ("average_rating", "rating_number") if col in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last", kind="stable")
    return rows


def _covered_item_count(clusters: pd.DataFrame) -> int:
    covered: set[int] = set()
    if "indices" not in clusters.columns:
        return 0
    for value in clusters["indices"]:
        covered.update(_cluster_indices(value))
    return len(covered)


def _cluster_size(value: object) -> int:
    return len(_cluster_indices(value))


def _clean_label(value: object) -> str:
    label = str(value or "").strip()
    return "Untitled Cluster" if not label else " ".join(label.split())


def render_dataset(info: ExampleBundleInfo) -> None:
    init_product_gallery_styles()

    stat = Path(info.path).stat()
    bundle = _load_bundle(str(info.path), stat.st_mtime_ns, stat.st_size)
    metadata = bundle["metadata"]
    clusters = bundle["clusters"].copy()
    title = bundle["title"]

    if "indices" in clusters.columns:
        clusters["_n_items"] = clusters["indices"].map(_cluster_size)
    else:
        clusters["_n_items"] = 0
    if "cluster_label" in clusters.columns:
        clusters["_label"] = clusters["cluster_label"].map(_clean_label)
    else:
        clusters["_label"] = "Untitled Cluster"

    with st.sidebar:
        st.subheader("Dataset Options")
        show_centroid = st.checkbox("Show latent factor details", value=True)
        cards_per_cluster = st.slider("Products per row", min_value=4, max_value=18, value=10, step=1)
        max_clusters = st.slider(
            "Clusters shown",
            min_value=1,
            max_value=max(1, len(clusters)),
            value=min(len(clusters), 24),
            step=1,
        )

    st.title(title)

    covered_items = _covered_item_count(clusters)
    total_assignments = int(clusters["_n_items"].sum())
    a, b, c, d = st.columns(4)
    a.metric("Products", f"{len(metadata):,}")
    b.metric("Clusters", f"{len(clusters):,}")
    #c.metric("Products Covered", f"{covered_items:,}")
    #d.metric("Assignments", f"{total_assignments:,}")

    st.subheader("Discovered Clusters")
    st.caption(
        "Each row is a labeled product segment discovered from sparse Compresso representations. "
        "Cards show representative products from that cluster."
    )

    shown_clusters = clusters.sample(frac=1).head(max_clusters)
    for row_idx, (_, row) in enumerate(shown_clusters.iterrows()):
        indices = _cluster_indices(row.get("indices", []))
        items = _cluster_items(metadata, indices)
        centroid = row.get("centroid")
        render_product_cluster_row(
            str(row.get("_label") or "Untitled Cluster"),
            items,
            row_id=f"{info.dataset_id}_{row_idx}",
            item_count=len(indices),
            centroid=centroid,
            show_centroid=show_centroid,
            limit=cards_per_cluster,
        )

    if len(shown_clusters) == 0:
        st.info("No clusters were found in this example bundle.")


def render_missing_examples() -> None:
    st.title("Examples")
    st.info("No example bundles were found in the data directory.")


def render_methodology(markdown: str) -> None:
    st.title("Methodology")
    if markdown.strip():
        st.markdown(markdown_without_title(markdown))
    else:
        st.info("No methodology description was found.")
