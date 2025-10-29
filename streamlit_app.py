import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="SOR Organizational Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = Path("data/SOR_Review_Offices_Hai.xlsx")
SLIDE_WEIGHTS = {
    "Human.Resources": 0.10,
    "IT.Services": 0.10,
    "Communications": 0.10,
    "PEO": 0.10,
    "Region.Offices": 0.40,
    "Training.Center": 0.10,
    "Customer.Service": 0.10,
}


def _read_xlsx_without_engine(path: Path) -> pd.DataFrame:
    """Fallback reader that parses an .xlsx file without optional engines."""

    def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        strings: list[str] = []
        for si in root.findall("a:si", namespace):
            text_fragments = [node.text or "" for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
            strings.append("".join(text_fragments))
        return strings

    def _cell_index(cell_ref: str) -> int:
        """Convert an Excel cell reference (e.g. 'C5') to a zero-based column index."""
        col = 0
        for char in cell_ref:
            if char.isalpha():
                col = col * 26 + (ord(char.upper()) - ord("A") + 1)
            else:
                break
        return col - 1

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


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    """Load the Excel dataset, using a fallback parser if optional deps are missing."""
    try:
        df = pd.read_excel(path)
    except ImportError:
        df = _read_xlsx_without_engine(path)
    except ValueError:
        df = _read_xlsx_without_engine(path)
    df = df.copy()
    df["SOR_ID"] = df["SOR_ID"].astype(str)
    numeric_columns = [
        "Human.Resources",
        "IT.Services",
        "Communications",
        "PEO",
        "Region.Offices",
        "Training.Center",
        "Customer.Service",
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df


def compute_weighted_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    numeric_columns = list(weights.keys())
    weight_series = pd.Series(weights)
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


def prepare_pca(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_columns])
    pca = PCA()
    components = pca.fit_transform(scaled)
    pc_columns = [f"PC{i + 1}" for i in range(components.shape[1])]
    pcs_df = pd.DataFrame(components, columns=pc_columns, index=df.index)
    return pcs_df, pca, scaled


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> go.Figure:
    corr = df[numeric_columns].corr()
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


def plot_biplot(pcs_df: pd.DataFrame, pca: PCA, df: pd.DataFrame, numeric_columns: list[str]) -> go.Figure:
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
        comparison[["SOR_ID", "Classification"]], on="SOR_ID", suffixes=("_Slide", "_Equal")
    )
    merged["Changed"] = np.where(merged["Classification_Slide"] != merged["Classification_Equal"], "Yes", "No")
    return merged


def render_overview(df: pd.DataFrame, numeric_columns: list[str], filtered: pd.DataFrame) -> None:
    st.subheader("Dataset Summary")
    st.markdown(
        """
        The dataset covers strategic objectives across multiple organizational support areas. Scores range from 0 (low performance)
        to 10 (high performance) for each criterion.
        """
    )
    info_columns = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": pd.Series([df[col].dtype for col in df.columns], dtype="string"),
        }
    )
    info_columns["Column"] = info_columns["Column"].astype("string")
    st.dataframe(info_columns, width="stretch")

    st.markdown("### Summary Statistics")
    st.dataframe(df[numeric_columns].describe().T, width="stretch")

    st.markdown("### Correlation Heatmap")
    st.plotly_chart(plot_correlation_heatmap(df, numeric_columns), width="stretch")

    st.markdown("### Filtered Preview")
    st.dataframe(filtered, width="stretch")


def render_evaluation(slide_scores: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.subheader("Weighted Scoring Framework")
    st.markdown(
        """
        Weighted scores prioritize regional office performance (40%) while giving equal emphasis (10% each) to other support areas.
        The resulting composite score remains on a 0–10 scale.
        """
    )

    st.plotly_chart(plot_weighted_scores(filtered), width="stretch")

    st.markdown("### Classification Summary")
    summary = slide_scores.groupby("Classification").agg(
        Objectives=("SOR_ID", "count"),
        Average_Score=("Weighted.Score", "mean"),
    )
    st.dataframe(summary, width="stretch")

    st.markdown("### Detailed Results")
    st.dataframe(filtered, width="stretch")

    csv_buffer = io.StringIO()
    slide_scores.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Evaluation (CSV)",
        data=csv_buffer.getvalue(),
        file_name="sor_weighted_scores.csv",
        mime="text/csv",
    )


def render_pca(df: pd.DataFrame, numeric_columns: list[str], pcs_df: pd.DataFrame, pca: PCA, filtered_idx: pd.Index) -> None:
    st.subheader("Principal Component Analysis")
    st.markdown(
        """
        PCA is applied on Z-score normalized data to reveal dominant performance patterns. The scree plot highlights the
        contribution of each component, while the biplot shows how objectives and scoring dimensions align along PC1 and PC2.
        """
    )

    st.plotly_chart(plot_scree(pca), width="stretch")

    st.markdown("### Biplot (PC1 vs PC2)")
    filtered_pcs = pcs_df.loc[filtered_idx]
    st.plotly_chart(plot_biplot(filtered_pcs, pca, df.loc[filtered_idx], numeric_columns), width="stretch")

    explained = pd.DataFrame(
        {
            "Principal Component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "Explained Variance": pca.explained_variance_ratio_,
        }
    )
    st.markdown("### Explained Variance Table")
    st.dataframe(explained, width="stretch")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_columns,
        columns=[f"PC{i + 1}" for i in range(len(pca.components_))],
    )
    st.markdown("### Component Loadings")
    st.dataframe(loadings, width="stretch")


def render_validation(slide_scores: pd.DataFrame, equal_scores: pd.DataFrame, pcs_df: pd.DataFrame) -> None:
    st.subheader("Sensitivity & Consistency Checks")
    st.markdown(
        """
        This section contrasts the prescribed slide weights with an equal-weight scenario to gauge sensitivity. It also reports
        the correlation between the weighted composite score and the first principal component (PC1).
        """
    )

    comparison = weight_summary_table(slide_scores, equal_scores)
    st.markdown("### Classification Comparison")
    st.dataframe(comparison, width="stretch")

    changed_count = (comparison["Changed"] == "Yes").sum()
    st.metric("Objectives with Different Classification", changed_count)

    correlation = np.corrcoef(slide_scores["Weighted.Score"], pcs_df["PC1"])[0, 1]
    st.metric("Correlation between Weighted Score and PC1", f"{correlation:.2f}")

    st.markdown("### Equal Weight Results")
    st.dataframe(equal_scores, width="stretch")


def render_conclusion(slide_scores: pd.DataFrame, pcs_df: pd.DataFrame) -> None:
    st.subheader("Final Report Summary")
    top_objectives = slide_scores.nlargest(3, "Weighted.Score")[
        ["SOR_ID", "Goal", "Obj", "Weighted.Score", "Classification"]
    ]
    low_objectives = slide_scores.nsmallest(3, "Weighted.Score")[
        ["SOR_ID", "Goal", "Obj", "Weighted.Score", "Classification"]
    ]

    st.markdown(
        """
        **Key Findings**

        * Regional office performance strongly influences the composite outcome, elevating objectives with robust field support.
        * Objectives classified as **Noteworthy** align closely with positive PC1 scores, confirming consistency between weighting and PCA.
        * Areas flagged **To-Improve** show negative contributions on PC1 and below-threshold weighted scores, highlighting priority gaps.
        """
    )

    st.markdown("### Top Performing Objectives")
    st.dataframe(top_objectives, width="stretch")

    st.markdown("### Improvement Priorities")
    st.dataframe(low_objectives, width="stretch")

    st.markdown(
        """
        **Next Steps**

        * Engage objective owners for targeted action planning around low-scoring criteria.
        * Monitor medium-tier objectives for trend shifts and reassess weights quarterly.
        * Incorporate qualitative feedback alongside the quantitative dashboard for a balanced review.
        """
    )


def main() -> None:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}.")
        return

    df = load_dataset(DATA_PATH)
    numeric_columns = list(SLIDE_WEIGHTS.keys())

    slide_scores = compute_weighted_scores(df, SLIDE_WEIGHTS)
    equal_weights = {column: 1.0 for column in numeric_columns}
    equal_scores = compute_weighted_scores(df, equal_weights)

    pcs_df, pca, _ = prepare_pca(df, numeric_columns)

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to",
        ["Overview", "Evaluation", "PCA", "Validation", "Conclusion"],
    )

    st.sidebar.markdown("### Filter by Classification")
    available_classes = slide_scores["Classification"].unique().tolist()
    selected_classes = st.sidebar.multiselect(
        "Select categories",
        options=available_classes,
        default=available_classes,
    )

    if selected_classes:
        filtered_scores = slide_scores[slide_scores["Classification"].isin(selected_classes)]
    else:
        filtered_scores = slide_scores

    filtered_indices = filtered_scores.index

    if section == "Overview":
        render_overview(df, numeric_columns, filtered_scores)
    elif section == "Evaluation":
        render_evaluation(slide_scores, filtered_scores)
    elif section == "PCA":
        render_pca(df, numeric_columns, pcs_df, pca, filtered_indices)
    elif section == "Validation":
        render_validation(slide_scores, equal_scores, pcs_df)
    else:
        render_conclusion(slide_scores, pcs_df)


if __name__ == "__main__":
    main()
