from __future__ import annotations

from collections.abc import Callable
import html

import streamlit as st

from lib.example_bundles import discover_example_bundles, markdown_sections
from lib.paths import DATA_DIR
from sections import (
    examples,
    overview,
)


st.set_page_config(
    page_title="Compresso Demo",
    page_icon=":",
    layout="wide",
)


PageFn = Callable[[], None]

EXAMPLE_BUNDLES = discover_example_bundles()
OVERVIEW_PATH = DATA_DIR / "overview.md"
OVERVIEW_MARKDOWN = OVERVIEW_PATH.read_text(encoding="utf-8") if OVERVIEW_PATH.exists() else "# Compresso"
METHODOLOGY_PAGE = "example:methodology"
METHODOLOGY_PATH = DATA_DIR / "description.md"
METHODOLOGY_MARKDOWN = METHODOLOGY_PATH.read_text(encoding="utf-8") if METHODOLOGY_PATH.exists() else ""
METHODOLOGY_SECTIONS = tuple(markdown_sections(METHODOLOGY_MARKDOWN))

PAGES: dict[str, PageFn] = {
    "Overview": lambda: overview.render(OVERVIEW_MARKDOWN),
    METHODOLOGY_PAGE: lambda: examples.render_methodology(METHODOLOGY_MARKDOWN),
}
for bundle in EXAMPLE_BUNDLES:
    PAGES[bundle.page_key] = lambda bundle=bundle: examples.render_dataset(bundle)
if not EXAMPLE_BUNDLES:
    PAGES["Examples"] = examples.render_missing_examples

MENU = [
    ("Compresso", [("Overview", "Overview")]),
    (
        "Examples",
        [("Methodology", METHODOLOGY_PAGE)]
        + ([(bundle.title, bundle.page_key) for bundle in EXAMPLE_BUNDLES] if EXAMPLE_BUNDLES else [("No bundles found", "Examples")]),
    ),
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
            .nav-subitems {
                margin: -0.45rem 0 0.45rem 2.0rem;
            }
            .nav-subitems a {
                color: inherit;
                display: block;
                font-size: 0.92rem;
                line-height: 1.35;
                opacity: 0.82;
                padding: 0.03rem 0;
                text-decoration: none;
            }
            .nav-subitems a:hover {
                opacity: 1;
                text-decoration: underline;
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
                if active and page_key == METHODOLOGY_PAGE and METHODOLOGY_SECTIONS:
                    links = "".join(
                        f'<a href="#{html.escape(section.anchor)}">- {html.escape(section.title)}</a>'
                        for section in METHODOLOGY_SECTIONS
                    )
                    st.markdown(f'<div class="nav-subitems">{links}</div>', unsafe_allow_html=True)

    return page


page = render_menu()
PAGES[page]()
