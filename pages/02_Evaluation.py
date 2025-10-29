from __future__ import annotations

import streamlit as st

from sor_dashboard_core import (
    DATA_PATH,
    PLOTLY_CONFIG,
    SLIDE_WEIGHTS,
    compute_weighted_scores,
    load_dataset,
    plot_weighted_scores,
)
from sor_dashboard_ui import classification_filter


st.title("Evaluation")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

df = load_dataset()
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)

filtered_scores, _ = classification_filter(slide_scores)

st.subheader("Weighted Scoring Framework")
st.markdown(
    """
    Weighted scores prioritize regional office performance (40%) while giving equal emphasis (10% each) to other support areas.
    The resulting composite score remains on a 0–10 scale.
    """
)

st.plotly_chart(
    plot_weighted_scores(filtered_scores),
    config=PLOTLY_CONFIG,
    use_container_width=True,
)

st.markdown("### Classification Summary")
summary = filtered_scores.groupby("Classification").agg(
    Objectives=("SOR_ID", "count"),
    Average_Score=("Weighted.Score", "mean"),
)
summary.index.name = "Classification"
st.dataframe(summary, width="stretch")

st.markdown("### Detailed Results")
st.dataframe(filtered_scores, width="stretch")
