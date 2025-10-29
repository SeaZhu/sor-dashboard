from __future__ import annotations

import streamlit as st

from sor_dashboard_core import (
    DATA_PATH,
    SLIDE_WEIGHTS,
    compute_weighted_scores,
    load_dataset,
)
from sor_dashboard_ui import classification_filter


st.title("Conclusion")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

df = load_dataset()
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)

filtered_scores, selection = classification_filter(slide_scores)

st.subheader("Final Report Summary")
st.markdown(
    """
    **Key Findings**

    * Regional office performance strongly influences the composite outcome, elevating objectives with robust field support.
    * Objectives classified as **Noteworthy** align closely with positive PC1 scores, confirming consistency between weighting and PCA.
    * Areas flagged **To-Improve** show below-threshold weighted scores, highlighting priority gaps.
    """
)

if filtered_scores.empty:
    st.info("No objectives match the current selection.")
else:
    st.markdown("### Top Performing Objectives")
    top_objectives = filtered_scores.nlargest(3, "Weighted.Score")[
        ["SOR_ID", "Goal", "Obj", "Weighted.Score", "Classification"]
    ]
    st.dataframe(top_objectives, width="stretch")

    st.markdown("### Improvement Priorities")
    low_objectives = filtered_scores.nsmallest(3, "Weighted.Score")[
        ["SOR_ID", "Goal", "Obj", "Weighted.Score", "Classification"]
    ]
    st.dataframe(low_objectives, width="stretch")

st.markdown(
    """
    **Next Steps**

    * Engage objective owners for targeted action planning around low-scoring criteria.
    * Monitor medium-tier objectives for trend shifts and reassess weights quarterly.
    * Incorporate qualitative feedback alongside the quantitative dashboard for a balanced review.
    """
)
