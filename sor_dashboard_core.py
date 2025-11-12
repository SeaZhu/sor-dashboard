"""
Core module annotations: the constants and functions below support data loading, preprocessing, and Plotly
visualizations used on the PCA page.
"""

# Allow future-style annotations such as list[str] even on older Python versions.
from __future__ import annotations

# Parse Excel XML using the standard library so the workbook can be read without third-party engines.
import xml.etree.ElementTree as ET
# Handle the XLSX zip container structure.
import zipfile
# Cache dataset reads via an LRU decorator so repeated calls are inexpensive.
from functools import lru_cache
# Represent filesystem locations in a cross-platform way.
from pathlib import Path
# Provide type hints for iterable parameters.
from typing import Iterable

# Numerical routines used for PCA math and classification logic.
import numpy as np
# Pandas supplies tabular structures and I/O helpers.
import pandas as pd
# Plotly Express builds quick exploratory charts.
import plotly.express as px
# Plotly Graph Objects provides fine-grained plot assembly (e.g., for the biplot).
import plotly.graph_objects as go
# Use scikit-learn's PCA implementation.
from sklearn.decomposition import PCA
# Standardize columns so each has zero mean and unit variance.
from sklearn.preprocessing import StandardScaler


# Point to the source Excel workbook, typically data/SOR_Review_Offices_Hai.xlsx.
DATA_PATH = Path("data/SOR_Review_Offices_Hai.xlsx")

# Slide weighting used for aggregation: keys are dimension names, values are their relative weights (e.g., Region.Offices=0.40).
SLIDE_WEIGHTS: dict[str, float] = {
    "Human.Resources": 0.10,
    "IT.Services": 0.10,
    "Communications": 0.10,
    "PEO": 0.10,
    "Region.Offices": 0.40,
    "Training.Center": 0.10,
    "Customer.Service": 0.10,
}

# Numeric columns for PCA follow the weighting keys, e.g., ["Human.Resources", "IT.Services", ...].
NUMERIC_COLUMNS: list[str] = list(SLIDE_WEIGHTS.keys())

# Equal weighting used on other pages, stored as a constant for consistency.
EQUAL_WEIGHTS: dict[str, float] = {column: 1.0 for column in NUMERIC_COLUMNS}

# Shared Plotly configuration turns off the logo and enables responsive resizing.
PLOTLY_CONFIG: dict[str, object] = {
    "displaylogo": False,
    "responsive": True,
}


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for si in root.findall("a:si", namespace):
        fragments = [
            node.text or ""
            for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
        ]
        strings.append("".join(fragments))
    return strings


def _cell_index(cell_ref: str) -> int:
    column = 0
    for char in cell_ref:
        if char.isalpha():
            column = column * 26 + (ord(char.upper()) - ord("A") + 1)
        else:
            break
    return column - 1


def _read_xlsx_without_engine(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        worksheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[list[object]] = []
        max_cols = 0
        for row in worksheet.findall(".//a:row", namespace):
            values: dict[int, object] = {}
            for cell in row.findall("a:c", namespace):
                reference = cell.attrib.get("r", "")
                idx = _cell_index(reference)
                value_node = cell.find("a:v", namespace)
                cell_type = cell.attrib.get("t")
                if value_node is None:
                    value = None
                elif cell_type == "s":
                    lookup = int(value_node.text)
                    value = shared_strings[lookup] if lookup < len(shared_strings) else None
                else:
                    value = float(value_node.text)
                values[idx] = value
            if values:
                max_cols = max(max_cols, max(values.keys()) + 1)
                ordered = [values.get(i) for i in range(max_cols)]
                rows.append(ordered)

    if not rows:
        raise ValueError("The workbook appears to be empty.")

    header = [str(item) for item in rows[0]]
    data_rows = [row[: len(header)] for row in rows[1:]]
    return pd.DataFrame(data_rows, columns=header)


# Cache load_dataset results to avoid repeatedly reading from disk.
@lru_cache(maxsize=1)
def load_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """
    Read and sanitize the Excel dataset.

    Example argument:
        path="data/SOR_Review_Offices_Hai.xlsx"
    Example return row:
        {"SOR_ID": "1.1", "Goal": "Goal 1", "Human.Resources": 6.5, ...}
    """

    # Convert the input string to a Path for existence checks.
    dataset_path = Path(path)
    # Raise a FileNotFoundError so the caller can warn the user if the file is missing.
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    try:
        # Prefer pandas.read_excel, which uses any installed Excel engine.
        df = pd.read_excel(dataset_path)
    except (ImportError, ValueError):
        # Fall back to the handcrafted parser when engines are unavailable.
        df = _read_xlsx_without_engine(dataset_path)
    # copy() ensures cached data is not mutated by downstream callers.
    df = df.copy()
    # Normalize SOR_ID to strings (e.g., "1.2") for consistent indexing and display.
    df["SOR_ID"] = df["SOR_ID"].astype(str)
    # Cast numeric columns to floats; invalid entries become NaN for safe math operations.
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Return the cleaned dataset.
    return df


def compute_weighted_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """
    Calculate weighted scores per row and assign qualitative categories.

    Example argument:
        weights={"Human.Resources": 0.1, ...}
    Example return row:
        Weighted.Score=6.82, Classification="To-Monitor"
    """

    # Work only with the weighted columns so ordering matches the provided weights.
    numeric_columns = list(weights.keys())
    # Create a Series cast to float to allow normalization.
    weight_series = pd.Series(weights, dtype="float64")
    # Normalize the weights so they always sum to 1 even if the input does not.
    weight_series = weight_series / weight_series.sum()
    # Multiply and sum per row to produce the weighted score, e.g., scores[0] = 6.82.
    scores = (df[numeric_columns] * weight_series).sum(axis=1)
    # Use pd.cut to map score ranges to descriptive classification labels.
    classification = pd.cut(
        scores,
        bins=[-np.inf, 4, 8, np.inf],
        labels=["To-Improve", "To-Monitor", "Noteworthy"],
        right=False,
    )
    # Copy the input DataFrame to keep the original untouched.
    result = df.copy()
    # Store the weighted score so the UI can display it directly.
    result["Weighted.Score"] = scores
    # Store the classification string, converting from Categorical to plain text for robustness.
    result["Classification"] = classification.astype(str)
    # Return the augmented DataFrame.
    return result


def prepare_pca(df: pd.DataFrame, numeric_columns: Iterable[str]) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Standardize the given numeric columns and perform PCA.

    Example return:
        pcs_df head -> PC1=-1.23, PC2=0.45; pca.components_.shape == (7, 7); scaled is the numpy array.
    """

    # Fit a StandardScaler so each dimension has mean 0 and standard deviation 1.
    scaler = StandardScaler()
    # Fit-transform the numeric columns, producing a 2D array ready for PCA.
    scaled = scaler.fit_transform(df[list(numeric_columns)])
    # Create a PCA object that retains all available components.
    pca = PCA()
    # Fit PCA and compute each row's score along every component.
    components = pca.fit_transform(scaled)
    # Build column names such as ["PC1", "PC2", ...].
    pc_columns = [f"PC{i + 1}" for i in range(components.shape[1])]
    # Wrap the scores in a DataFrame using the original index for easy joins.
    pcs_df = pd.DataFrame(components, columns=pc_columns, index=df.index)
    # Return the component scores, the fitted PCA object, and the standardized matrix.
    return pcs_df, pca, scaled


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: Iterable[str]) -> go.Figure:
    corr = df[list(numeric_columns)].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_weighted_scores(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df,
        x="SOR_ID",
        y="Weighted.Score",
        color="Classification",
        color_discrete_map={
            "Noteworthy": "#2b9348",
            "To-Monitor": "#f9c74f",
            "To-Improve": "#f94144",
        },
        hover_data=["Goal", "Obj"],
    )
    fig.update_layout(yaxis_title="Weighted Score (0-10)", xaxis_title="Strategic Objective")
    return fig


def plot_scree(pca: PCA) -> go.Figure:
    """
    Render a scree plot from the PCA model's explained_variance_ratio_.

    Example:
        explained_variance_ratio_ = [0.42, 0.25, ...]
    """

    # Access the fraction of variance explained by each component.
    explained_variance = pca.explained_variance_ratio_
    # Build the bar chart where x is the component index and y is the variance ratio.
    fig = px.bar(
        x=list(range(1, len(explained_variance) + 1)),
        y=explained_variance,
        labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
    )
    # Customize bar colors to match the dashboard palette.
    fig.update_traces(marker_color="#577590")
    # Tighten margins so the figure fits neatly inside Streamlit containers.
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    # Return the Plotly figure for rendering.
    return fig


def plot_variable_contributions(
    pca: PCA,
    numeric_columns: Iterable[str],
    components: int = 2,
) -> go.Figure:
    """
    Display how strongly each variable contributes to the leading principal components using a grouped bar chart.

    Example:
        numeric_columns=["Human.Resources", ...], components=2 -> tidy DataFrame with Dimension, Component, Contribution
    """

    # Ensure numeric_columns is materialized as a list for index operations.
    numeric_columns = list(numeric_columns)
    # Limit the plotted component count (default: first two components).
    component_count = min(components, pca.components_.shape[0])
    # Prepare labels like ["PC1", "PC2"].
    pc_labels = [f"PC{i + 1}" for i in range(component_count)]
    # Compute contributions via squared loadings (coefficient^2 shows variable influence).
    contributions = (pca.components_[:component_count] ** 2)
    # Normalize each component's contributions so they sum to 1.
    contributions = contributions / contributions.sum(axis=1, keepdims=True)
    # Assemble a wide DataFrame where rows are variables and columns are components.
    contribution_df = pd.DataFrame(contributions.T, index=numeric_columns, columns=pc_labels)
    # Reset the index to produce a Dimension column for the melt step.
    contribution_df = contribution_df.reset_index(names="Dimension")
    # Melt to a tidy format with Dimension, Component, Contribution columns.
    tidy = contribution_df.melt(
        id_vars="Dimension",
        var_name="Component",
        value_name="Contribution",
    )
    # Plot a grouped bar chart showing each variable's influence per component.
    fig = px.bar(
        tidy,
        x="Dimension",
        y="Contribution",
        color="Component",
        barmode="group",
        labels={"Dimension": "Department", "Contribution": "Relative Influence"},
    )
    # Adjust margins so the chart fits inside Streamlit containers.
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    # Return the ready-to-render figure.
    return fig


def plot_biplot(
    pcs_df: pd.DataFrame,
    pca: PCA,
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
) -> go.Figure:
    """
    Build a biplot that positions objectives on the PC1/PC2 plane and overlays variable loadings.

    Example inputs:
        pcs_df.iloc[0]["PC1"] = -1.23, df.iloc[0]["SOR_ID"] = "1.1"
    """

    # Start with an empty figure to which scatter points and arrows are added incrementally.
    fig = go.Figure()
    # Plot objective scores with marker labels showing the SOR_ID.
    fig.add_trace(
        go.Scatter(
            x=pcs_df["PC1"],
            y=pcs_df["PC2"],
            mode="markers+text",
            text=df["SOR_ID"],
            textposition="top center",
            marker=dict(size=10, color="#277da1", opacity=0.8),
            name="Objectives",
        )
    )

    # Compute loading vectors for the first two components scaled by sqrt(explained variance).
    loadings = pca.components_.T[:, :2] * np.sqrt(pca.explained_variance_[:2])
    # Derive a scaling factor from the scatter extent so arrow lengths remain readable.
    scale = 1.1 * np.max(np.abs(pcs_df[["PC1", "PC2"]]).values)
    # Ensure numeric_columns is a list so we can index into it directly.
    numeric_columns = list(numeric_columns)
    for idx, column in enumerate(numeric_columns):
        # Extract the end point of each variable arrow.
        x, y = loadings[idx]
        # Draw a line segment from the origin to the scaled loading vector.
        fig.add_trace(
            go.Scatter(
                x=[0, x * scale],
                y=[0, y * scale],
                mode="lines",
                line=dict(color="#f3722c", width=2),
                showlegend=False,
                hoverinfo="none",
            )
        )
        # Annotate the arrow tip with the variable name.
        fig.add_annotation(
            x=x * scale,
            y=y * scale,
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="middle",
            text=column,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#f3722c",
        )

    # Configure axis titles, zero lines, and hide the legend for a cleaner layout.
    fig.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#999999"),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#999999"),
    )
    # Return the constructed biplot.
    return fig


def weight_summary_table(base: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    merged = base[["SOR_ID", "Classification"]].merge(
        comparison[["SOR_ID", "Classification"]],
        on="SOR_ID",
        suffixes=("_Slide", "_Equal"),
    )
    merged["Changed"] = np.where(
        merged["Classification_Slide"] != merged["Classification_Equal"],
        "Yes",
        "No",
    )
    return merged
