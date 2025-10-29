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
filtered_equal = equal_scores.loc[filtered_ids]
filtered_pcs = pcs_df.loc[filtered_ids]

comparison = weight_summary_table(slide_scores, equal_scores)
comparison = comparison[comparison["SOR_ID"].isin(filtered_scores["SOR_ID"])]

st.subheader("Sensitivity & Consistency Checks")
st.markdown(
    """
    This section contrasts the prescribed slide weights with an equal-weight scenario to gauge sensitivity. It also reports
    the correlation between the weighted composite score and the first principal component (PC1).
    """
)

st.markdown("### Classification Comparison")
st.dataframe(comparison, width="stretch")

changed_count = (comparison["Changed"] == "Yes").sum()
st.metric("Objectives with Different Classification", changed_count)

if len(filtered_scores) > 1:
    correlation = float(np.corrcoef(filtered_scores["Weighted.Score"], filtered_pcs["PC1"])[0, 1])
    st.metric("Correlation between Weighted Score and PC1", f"{correlation:.2f}")
else:
    st.metric("Correlation between Weighted Score and PC1", "N/A")

st.markdown("### Equal Weight Results")
st.dataframe(filtered_equal, width="stretch")
