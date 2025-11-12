"""UI helper module: contains the classification filter component used by the PCA page."""

# Enable future-style annotations to keep type hints concise.
from __future__ import annotations

# Import Streamlit to render sidebar controls.
import streamlit as st


# Session state key that preserves the multi-select choice across reruns.
FILTER_STATE_KEY = "classification_filter"


def classification_filter(slide_scores):
    """
    Filter the score table according to sidebar classification selections.

    Example argument:
        slide_scores.head()["Classification"] -> ["Noteworthy", "To-Monitor", ...]
    Example return:
        (filtered_scores, selection) where selection = ["Noteworthy", "To-Monitor"]
    """

    # Collect unique classification labels in alphabetical order, skipping missing values.
    available_classes = sorted(slide_scores["Classification"].dropna().unique().tolist())
    # Add a sidebar heading so users recognize the control cluster.
    st.sidebar.header("Classification Filter")
    # When no classification data exists, show all rows and notify the user.
    if not available_classes:
        st.sidebar.info("No classification data available; showing all objectives.")
        return slide_scores, available_classes

    # Use the prior session-state selection if available; otherwise default to all classes.
    default_selection = st.session_state.get(FILTER_STATE_KEY, available_classes)

    # Render the multi-select allowing users to pick the categories they want to inspect.
    selection = st.sidebar.multiselect(
        "Select categories",
        options=available_classes,
        default=default_selection,
        key=FILTER_STATE_KEY,
    )

    # If all options are cleared, inform the user and revert to showing every category.
    if not selection:
        st.sidebar.info("No categories selected; showing all objectives.")
        selection = available_classes

    # Filter the DataFrame to rows whose Classification falls within the selection.
    filtered_scores = slide_scores[slide_scores["Classification"].isin(selection)]
    return filtered_scores, selection
