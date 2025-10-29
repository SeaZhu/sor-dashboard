from __future__ import annotations

import streamlit as st


FILTER_STATE_KEY = "classification_filter"


def classification_filter(slide_scores):
    """Return the subset of slide_scores matching the sidebar selection."""

    available_classes = sorted(slide_scores["Classification"].dropna().unique().tolist())
    st.sidebar.header("Classification Filter")
    if not available_classes:
        st.sidebar.info("No classification data available; showing all objectives.")
        return slide_scores, available_classes

    default_selection = st.session_state.get(FILTER_STATE_KEY, available_classes)

    selection = st.sidebar.multiselect(
        "Select categories",
        options=available_classes,
        default=default_selection,
        key=FILTER_STATE_KEY,
    )

    if not selection:
        st.sidebar.info("No categories selected; showing all objectives.")
        selection = available_classes

    filtered_scores = slide_scores[slide_scores["Classification"].isin(selection)]
    return filtered_scores, selection
