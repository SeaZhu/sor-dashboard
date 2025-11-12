"""
Line-by-line commenting convention: each block explains its purpose in English and, when helpful, includes
example data structures so that the PCA page flow can be understood by reading downward.
"""

# Import __future__ annotations so forward references can be used in type hints.
from __future__ import annotations

# Import pandas to manipulate DataFrames and render tables.
import pandas as pd
# Import Streamlit to render the page and provide interactivity.
import streamlit as st

# Import constants and helpers required by the PCA page from the core module.
from sor_dashboard_core import (
    # DATA_PATH: path to the Excel workbook, for example data/SOR_Review_Offices_Hai.xlsx.
    DATA_PATH,
    # NUMERIC_COLUMNS: list of numeric scoring columns used for PCA such as ["Human.Resources", ...].
    NUMERIC_COLUMNS,
    # PLOTLY_CONFIG: shared Plotly configuration like {"displaylogo": False, "responsive": True}.
    PLOTLY_CONFIG,
    # SLIDE_WEIGHTS: default department weights, e.g., {"Region.Offices": 0.40, ...}.
    SLIDE_WEIGHTS,
    # compute_weighted_scores: function that calculates weighted scores and classifications.
    compute_weighted_scores,
    # load_dataset: function that reads the Excel dataset and performs basic cleanup.
    load_dataset,
    # plot_biplot: builds the biplot showing principal components and variable vectors.
    plot_biplot,
    # plot_scree: renders the explained variance bar chart.
    plot_scree,
    # plot_variable_contributions: draws the variable contribution bar chart.
    plot_variable_contributions,
    # prepare_pca: standardizes data and performs PCA, returning component scores and the model.
    prepare_pca,
)
# Import classification_filter to provide sidebar category filtering.
from sor_dashboard_ui import classification_filter


# Render the page title "PCA".
st.title("PCA")

# Abort early with an error message when the Excel workbook cannot be found.
if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}.")
    st.stop()

# Load the dataset; sample row contains columns such as SOR_ID, Goal, Obj, Human.Resources, etc.
df = load_dataset()
# Compute weighted scores and classifications using SLIDE_WEIGHTS (e.g., Weighted.Score=6.8, Classification="To-Monitor").
slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)
# Run PCA over the numeric columns producing pcs_df (component scores), pca (fitted model), and the scaled array.
pcs_df, pca, _ = prepare_pca(df, NUMERIC_COLUMNS)

# Apply the sidebar filter so only objectives in the selected classifications remain in filtered_scores.
filtered_scores, _ = classification_filter(slide_scores)
# Capture the filtered row indices to synchronize component scores with the same objectives.
filtered_indices = filtered_scores.index

# Introduce the PCA results with descriptive Markdown text.
st.subheader("Principal Component Analysis")
st.markdown(
    """
    PCA is applied on Z-score normalized data to reveal dominant performance patterns. The scree plot highlights the
    contribution of each component, while the biplot shows how objectives and scoring dimensions align along PC1 and PC2.
    """
)

# Render the scree chart; its bars show each component's explained_variance_ratio_.
st.plotly_chart(
    plot_scree(pca),
    config=PLOTLY_CONFIG,
    width="stretch",
)

# Provide a heading for the biplot shown below.
st.markdown("### Biplot (PC1 vs PC2)")
# Keep only the rows for the selected objectives in the component score matrix.
filtered_pcs = pcs_df.loc[filtered_indices]
# Inform the user when no objectives remain after filtering.
if filtered_pcs.empty:
    st.info("No objectives match the current selection for the biplot.")
# Otherwise render the biplot to show component coordinates and variable vectors.
else:
    st.plotly_chart(
        plot_biplot(filtered_pcs, pca, df.loc[filtered_indices], NUMERIC_COLUMNS),
        config=PLOTLY_CONFIG,
        width="stretch",
    )

# Build the component loading matrix where rows correspond to numeric columns and columns correspond to PC indices.
loadings = pd.DataFrame(
    pca.components_.T,
    index=NUMERIC_COLUMNS,
    columns=[f"PC{i + 1}" for i in range(len(pca.components_))],
)
# Display the loadings table so the influence of each variable per component is visible.
st.markdown("### Component Loadings")
st.dataframe(loadings, width="stretch")

# Render a heading for the variable contribution plot.
st.markdown("### Variable Contributions Bar Chart")
# Use plot_variable_contributions to visualize how strongly each variable contributes to the main components.
st.plotly_chart(
    plot_variable_contributions(pca, NUMERIC_COLUMNS),
    config=PLOTLY_CONFIG,
    use_container_width=True,
)
