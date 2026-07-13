from __future__ import annotations

from base64 import b64encode

import streamlit as st

from lib.paths import DATA_DIR


OVERVIEW_IMAGE_PATH = DATA_DIR / "compresso.jpg"


def _overview_image_data_uri() -> str:
    if not OVERVIEW_IMAGE_PATH.exists():
        return ""
    encoded = b64encode(OVERVIEW_IMAGE_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render(markdown: str) -> None:
    st.markdown(
        """
        <style>
        :root {
            --overview-logo-max-width: 100px;
            --overview-logo-preferred-width: 10vw;
        }
        .overview-logo {
            float: left;
            margin: 0 1.5rem 1rem 0;
            height: auto;
            max-width: 100%;
            width: min(var(--overview-logo-preferred-width), var(--overview-logo-max-width));
        }
        .overview-logo-clear {
            clear: both;
        }
        @media (max-width: 700px) {
            .overview-logo {
                display: block;
                float: none;
                margin: 0 auto 1rem;
                max-width: 150px;
                width: 48vw;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    content = markdown.replace("{{ compresso_image }}", _overview_image_data_uri())
    st.markdown(content, unsafe_allow_html=True)
