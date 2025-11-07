from __future__ import annotations

import streamlit as st

from sor_dashboard_core import (
    DATA_PATH,
    SLIDE_WEIGHTS,
    compute_weighted_scores,
    load_dataset,
)
st.title("Conclusion")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

df = load_dataset()
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)

st.subheader("Final Report Summary")
st.markdown(
    """
    **Key Findings**

    * Regional office performance strongly influences the composite outcome, elevating objectives with robust field support.
    * Noteworthy objectives align with positive PC1 scores, confirming weighting and PCA agreement.
    * Areas flagged **To-Improve** show below-threshold weighted scores, highlighting priority gaps.
    """
)

st.markdown("### Top Performing Objectives")
top_objectives = slide_scores.nlargest(3, "Weighted.Score")[
    ["SOR_ID", "Goal", "Obj", "Weighted.Score", "Classification"]
]
st.dataframe(top_objectives, width="stretch")

st.markdown("### Improvement Priorities")
low_objectives = slide_scores.nsmallest(3, "Weighted.Score")[
    ["SOR_ID", "Goal", "Obj", "Weighted.Score", "Classification"]
]
st.dataframe(low_objectives, width="stretch")
