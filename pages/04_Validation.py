from __future__ import annotations

import numpy as np
import streamlit as st

from sor_dashboard_core import (
    DATA_PATH,
    EQUAL_WEIGHTS,
    NUMERIC_COLUMNS,
    SLIDE_WEIGHTS,
    compute_weighted_scores,
    load_dataset,
    prepare_pca,
    weight_summary_table,
)
from sor_dashboard_ui import classification_filter


st.title("Validation")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

df = load_dataset()
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)
equal_scores = compute_weighted_scores(df, EQUAL_WEIGHTS)
pcs_df, _, _ = prepare_pca(df, NUMERIC_COLUMNS)

filtered_scores, selection = classification_filter(slide_scores)
filtered_ids = filtered_scores.index
filtered_pcs = pcs_df.loc[filtered_ids]

comparison = weight_summary_table(slide_scores, equal_scores)
comparison = comparison[comparison["SOR_ID"].isin(filtered_scores["SOR_ID"])]

st.subheader("Sensitivity & Consistency Checks")

total_objectives = len(filtered_scores)
if total_objectives:
    changed_count = (comparison["Changed"] == "Yes").sum()
    consistency_pct = 1 - changed_count / total_objectives
    if total_objectives > 1:
        correlation = float(np.corrcoef(filtered_scores["Weighted.Score"], filtered_pcs["PC1"])[0, 1])
        correlation_text = f"{correlation:.2f}"
    else:
        correlation_text = "N/A"
    st.markdown(
        "\n".join(
            [
                f"**Filtered objectives:** {total_objectives}",
                f"* Equal-weight classifications align {consistency_pct:.0%} of the time.",
                f"* Weighted score vs. PC1 correlation: {correlation_text}.",
            ]
        )
    )
else:
    st.info("No objectives match the current selection for validation analysis.")

st.markdown("### Classification Comparison")
st.dataframe(comparison, width="stretch")
