from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from sections import (
    cluster_explorer,
    goodbooks_pipeline,
    overview,
    placeholder,
    reference_clustering,
    reference_models,
    reference_sparsify,
)


st.set_page_config(
    page_title="Compresso Demo",
    page_icon=":",
    layout="wide",
)


PageFn = Callable[[], None]

PAGES: dict[str, PageFn] = {
    "Overview": overview.render,
    "Sparsify": reference_sparsify.render,
    "Models": reference_models.render,
    "Clustering": reference_clustering.render,
    "GoodBooks Pipeline": goodbooks_pipeline.render,
    "Cluster Explorer": cluster_explorer.render,
    "User Recommendations": lambda: placeholder.render("User Recommendations"),
}

MENU = [
    ("Compresso", [("Overview", "Overview")]),
    ("Reference", [("Sparsify", "Sparsify"), ("Models", "Models"), ("Clustering", "Clustering")]),
    ("Example", [("GoodBooks Pipeline", "GoodBooks Pipeline")]),
    ("Explore", [("Cluster Explorer", "Cluster Explorer"), ("User Recommendations", "User Recommendations")]),
]


def set_page(page: str) -> None:
    st.session_state["page"] = page


def render_menu() -> str:
    if "page" not in st.session_state or st.session_state["page"] not in PAGES:
        st.session_state["page"] = "Overview"
    page = st.session_state["page"]

    with st.sidebar:
        #st.title("Compresso")
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] h1 {
                margin-bottom: 0rem 0 0 0;
            }
            [data-testid="stSidebar"] div.stButton {
                margin: 0rem 0;
            }
            [data-testid="stSidebar"] div.stButton > button {
                background: transparent;
                border: 0;
                box-shadow: none;
                color: inherit;
                display: block;
                font-size: 1rem;
                font-weight: 400;
                line-height: 0.3;
                margin: 0;
                min-height: 0;
                padding: 0.08rem 0 0.08rem 1.0rem;
                text-align: left;
                width: 100%;
            }
            [data-testid="stSidebar"] div.stButton > button:hover {
                background: transparent;
                border: 0;
                color: inherit;
                text-decoration: underline;
            }
            [data-testid="stSidebar"] div.stButton > button:disabled {
                background: transparent;
                border: 0;
                color: inherit;
                opacity: 1;
                font-weight: 750;
            }
            .nav-section {
                font-weight: 750;
                margin: 0.3rem 0 0.3rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for section, entries in MENU:
            st.markdown(f'<div class="nav-section">{section}</div>', unsafe_allow_html=True)
            for label, page_key in entries:
                active = page == page_key
                prefix = "-> " if active else ""
                st.button(
                    f"{prefix}{label}",
                    key=f"nav_{page_key}",
                    disabled=active,
                    on_click=set_page,
                    args=(page_key,),
                )

    return page


page = render_menu()
PAGES[page]()
