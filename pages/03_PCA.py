from __future__ import annotations

import pandas as pd
import streamlit as st

from sor_dashboard_core import (
    DATA_PATH,
    NUMERIC_COLUMNS,
    PLOTLY_CONFIG,
    SLIDE_WEIGHTS,
    compute_weighted_scores,
    load_dataset,
    plot_biplot,
    plot_scree,
    plot_variable_contributions,
    prepare_pca,
)
from sor_dashboard_ui import classification_filter


st.title("PCA")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

df = load_dataset()
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)
pcs_df, pca, _ = prepare_pca(df, NUMERIC_COLUMNS)

filtered_scores, _ = classification_filter(slide_scores)
filtered_indices = filtered_scores.index

st.subheader("Principal Component Analysis")
st.markdown(
    """
    PCA is applied on Z-score normalized data to reveal dominant performance patterns. The scree plot highlights the
    contribution of each component, while the biplot shows how objectives and scoring dimensions align along PC1 and PC2.
    """
)

st.plotly_chart(
    plot_scree(pca),
    config=PLOTLY_CONFIG,
    width="stretch",
)

st.markdown("### Biplot (PC1 vs PC2)")
filtered_pcs = pcs_df.loc[filtered_indices]
if filtered_pcs.empty:
    st.info("No objectives match the current selection for the biplot.")
else:
    st.plotly_chart(
        plot_biplot(filtered_pcs, pca, df.loc[filtered_indices], NUMERIC_COLUMNS),
        config=PLOTLY_CONFIG,
        width="stretch",
    )

loadings = pd.DataFrame(
    pca.components_.T,
    index=NUMERIC_COLUMNS,
    columns=[f"PC{i + 1}" for i in range(len(pca.components_))],
)
st.markdown("### Component Loadings")
st.dataframe(loadings, width="stretch")

st.markdown("### Variable Contributions Bar Chart")
st.plotly_chart(
    plot_variable_contributions(pca, NUMERIC_COLUMNS),
    config=PLOTLY_CONFIG,
    use_container_width=True,
)
