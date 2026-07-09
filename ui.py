from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from lib.cluster_view import item_card_rows


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


def init_product_gallery_styles() -> None:
    st.markdown(
        """
        <style>
        .cluster-row {
            margin: 0.2rem 0 2.1rem 0;
        }
        .cluster-row-header {
            align-items: baseline;
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem 0.75rem;
            margin: 0 0 0.75rem 0;
        }
        .cluster-row-title {
            color: #242938;
            font-size: 1.36rem;
            font-weight: 760;
            line-height: 1.2;
        }
        .cluster-row-meta {
            color: #6b7280;
            font-size: 0.86rem;
            line-height: 1.25;
        }
        .product-rail {
            display: flex;
            gap: 1rem;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0.1rem 0.15rem 0.8rem 0.05rem;
            scroll-snap-type: x mandatory;
        }
        .product-card {
            background: #ffffff;
            border: 1px solid rgba(35, 40, 55, 0.09);
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(20, 26, 38, 0.09);
            color: #242938;
            flex: 0 0 214px;
            height: 338px;
            overflow: hidden;
            scroll-snap-align: start;
        }
        .product-card-image {
            align-items: center;
            background: #f5f6f8;
            display: flex;
            height: 178px;
            justify-content: center;
            width: 100%;
        }
        .product-card-image img {
            display: block;
            height: 100%;
            object-fit: contain;
            width: 100%;
        }
        .product-card-no-image {
            color: #8b95a5;
            font-size: 0.82rem;
            font-weight: 650;
        }
        .product-card-body {
            padding: 0.78rem 0.86rem 0.85rem 0.86rem;
        }
        .product-card-title {
            display: -webkit-box;
            font-size: 0.92rem;
            font-weight: 760;
            line-height: 1.23;
            margin-bottom: 0.52rem;
            overflow: hidden;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 3;
        }
        .product-card-subtle {
            color: #687182;
            font-size: 0.78rem;
            line-height: 1.3;
            margin-top: 0.24rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_rating(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return ""


def _format_count(value: object) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return ""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M ratings"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k ratings"
    return f"{count} ratings"


def render_product_cluster_row(
    title: str,
    items: pd.DataFrame,
    *,
    row_id: str,
    item_count: int,
    centroid: object | None = None,
    show_centroid: bool = False,
    limit: int = 12,
) -> None:
    cards = []
    for _, row in items.head(limit).iterrows():
        item_title = html.escape(str(row.get("title") or "Untitled product"))
        item_id = html.escape(str(row.get("item_id") or ""))
        image = str(row.get("image_url") or "")
        image_html = (
            f'<img src="{html.escape(image)}" alt="{item_title}">'
            if image
            else '<div class="product-card-no-image">No image</div>'
        )
        rating = _format_rating(row.get("average_rating"))
        count = _format_count(row.get("rating_number"))
        rating_bits = []
        if rating:
            rating_bits.append(f"Rating {html.escape(rating)}")
        if count:
            rating_bits.append(html.escape(count))
        rating_text = " · ".join(rating_bits)
        cards.append(
            '<div class="product-card" title="{}">'
            '<div class="product-card-image">{}</div>'
            '<div class="product-card-body">'
            '<div class="product-card-title">{}</div>'
            '<div class="product-card-subtle">{}</div>'
            '<div class="product-card-subtle">#{}</div>'
            "</div>"
            "</div>".format(item_title, image_html, item_title, rating_text, item_id)
        )

    meta = [f"{int(item_count):,} products"]
    if show_centroid and centroid is not None:
        try:
            values = list(centroid)
            if len(values) >= 2:
                factor = int(float(values[0]))
                sign = "positive" if float(values[1]) >= 0 else "negative"
                meta.append(f"latent factor {factor}, {sign} activation")
        except (TypeError, ValueError):
            meta.append(f"centroid {html.escape(str(centroid))}")

    row_html = (
        '<div class="cluster-row">'
        '<div class="cluster-row-header">'
        f'<div class="cluster-row-title">{html.escape(title.strip())}</div>'
        f'<div class="cluster-row-meta">{" · ".join(html.escape(v) for v in meta)}</div>'
        "</div>"
        f'<div class="product-rail" id="{html.escape(row_id)}">'
        f'{"".join(cards)}'
        "</div>"
        "</div>"
    )
    st.markdown(row_html, unsafe_allow_html=True)
