from __future__ import annotations

import streamlit as st


def render(title: str) -> None:
    st.title(title)
    st.info("This page is coming next. For now, try the overview or cluster explorer.")
