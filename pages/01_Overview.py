from __future__ import annotations

import streamlit as st

from sor_dashboard_core import (
    DATA_PATH,
    NUMERIC_COLUMNS,
    PLOTLY_CONFIG,
    SLIDE_WEIGHTS,
    compute_weighted_scores,
    load_dataset,
    plot_correlation_heatmap,
)
from sor_dashboard_ui import classification_filter


st.title("Overview")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

df = load_dataset()
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)

filtered_scores, _ = classification_filter(slide_scores)

st.markdown("### Summary Statistics")
st.dataframe(df[NUMERIC_COLUMNS].describe().T, width="stretch")

st.markdown("### Correlation Heatmap")
st.plotly_chart(
    plot_correlation_heatmap(df, NUMERIC_COLUMNS),
    config=PLOTLY_CONFIG,
    width="stretch",
)

st.markdown("### Filtered Preview")
st.dataframe(filtered_scores, width="stretch")
