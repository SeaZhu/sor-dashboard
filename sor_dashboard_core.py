from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/SOR_Review_Offices_Hai.xlsx")

SLIDE_WEIGHTS: dict[str, float] = {
    "Human.Resources": 0.10,
    "IT.Services": 0.10,
    "Communications": 0.10,
    "PEO": 0.10,
    "Region.Offices": 0.40,
    "Training.Center": 0.10,
    "Customer.Service": 0.10,
}

NUMERIC_COLUMNS: list[str] = list(SLIDE_WEIGHTS.keys())

EQUAL_WEIGHTS: dict[str, float] = {column: 1.0 for column in NUMERIC_COLUMNS}

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


@lru_cache(maxsize=1)
def load_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    try:
        df = pd.read_excel(dataset_path)
    except (ImportError, ValueError):
        df = _read_xlsx_without_engine(dataset_path)
    df = df.copy()
    df["SOR_ID"] = df["SOR_ID"].astype(str)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return df


def compute_weighted_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    numeric_columns = list(weights.keys())
    weight_series = pd.Series(weights, dtype="float64")
    weight_series = weight_series / weight_series.sum()
    scores = (df[numeric_columns] * weight_series).sum(axis=1)
    classification = pd.cut(
        scores,
        bins=[-np.inf, 4, 8, np.inf],
        labels=["To-Improve", "To-Monitor", "Noteworthy"],
        right=False,
    )
    result = df.copy()
    result["Weighted.Score"] = scores
    result["Classification"] = classification.astype(str)
    return result


def prepare_pca(df: pd.DataFrame, numeric_columns: Iterable[str]) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[list(numeric_columns)])
    pca = PCA()
    components = pca.fit_transform(scaled)
    pc_columns = [f"PC{i + 1}" for i in range(components.shape[1])]
    pcs_df = pd.DataFrame(components, columns=pc_columns, index=df.index)
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
    explained_variance = pca.explained_variance_ratio_
    fig = px.bar(
        x=list(range(1, len(explained_variance) + 1)),
        y=explained_variance,
        labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
    )
    fig.update_traces(marker_color="#577590")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_variable_contributions(
    pca: PCA,
    numeric_columns: Iterable[str],
    components: int = 2,
) -> go.Figure:
    numeric_columns = list(numeric_columns)
    component_count = min(components, pca.components_.shape[0])
    pc_labels = [f"PC{i + 1}" for i in range(component_count)]
    contributions = (pca.components_[:component_count] ** 2)
    contributions = contributions / contributions.sum(axis=1, keepdims=True)
    contribution_df = pd.DataFrame(contributions.T, index=numeric_columns, columns=pc_labels)
    contribution_df = contribution_df.reset_index(names="Dimension")
    tidy = contribution_df.melt(
        id_vars="Dimension",
        var_name="Component",
        value_name="Contribution",
    )
    fig = px.bar(
        tidy,
        x="Dimension",
        y="Contribution",
        color="Component",
        barmode="group",
        labels={"Dimension": "Department", "Contribution": "Relative Influence"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_biplot(
    pcs_df: pd.DataFrame,
    pca: PCA,
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
) -> go.Figure:
    fig = go.Figure()
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

    loadings = pca.components_.T[:, :2] * np.sqrt(pca.explained_variance_[:2])
    scale = 1.1 * np.max(np.abs(pcs_df[["PC1", "PC2"]]).values)
    numeric_columns = list(numeric_columns)
    for idx, column in enumerate(numeric_columns):
        x, y = loadings[idx]
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

    fig.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#999999"),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#999999"),
    )
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
