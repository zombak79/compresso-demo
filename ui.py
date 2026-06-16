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
