import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def render(st):
    # Definir paleta de colores
    PROMPT_COLOR_MAP = {
        "On Tragedy": "#0072B2",
        "Egyptian Social Structure": "#E69F00",
        "The Third Wave": "#009E73",
        "Excerpt from The Jungle": "#D55E00",
    }
    DIST_COLOR_CONTENT = "#56B4E9"
    DIST_COLOR_WORDING = "#CC79A7"
    LINE_COLOR = "#F0E442"

    st.title("Resultados de modelos")

    SCRIPT_DIR = Path(__file__).resolve().parent
    csv_path = SCRIPT_DIR / "analisis_resumenes.csv"

    df = pd.read_csv(csv_path)

    df["len_mid"] = pd.to_numeric(df["len_mid"], errors="coerce")

    st.header("Análisis de Métricas de Calidad")

    st.subheader("Relación entre la longitud del resumen y las calificaciones promedio")

    metricas = ["content", "wording"]
    selected_metrics = st.multiselect(
        "Selecciona las métricas a mostrar:",
        metricas,
        default=metricas,
        key="metrics_1",
    )

    min_len, max_len = st.slider(
        "Selecciona el rango de longitud (número de palabras)",
        int(df["len_mid"].min()),
        int(df["len_mid"].max()),
        (int(df["len_mid"].min()), int(df["len_mid"].max())),
        key="slider_1",
    )

    filtered_df = df[(df["len_mid"] >= min_len) & (df["len_mid"] <= max_len)]

    # Crear color map para las métricas seleccionadas
    color_map = {"content": DIST_COLOR_CONTENT, "wording": DIST_COLOR_WORDING}

    fig = px.line(
        filtered_df,
        x="len_mid",
        y=selected_metrics,
        markers=True,
        labels={
            "len_mid": "Número de palabras en el resumen",
            "value": "Promedio de calificación",
            "variable": "Métrica",
        },
        title="Relación entre la longitud del resumen y las calificaciones promedio",
        color_discrete_map=color_map,
    )

    fig.update_layout(legend_title_text="Métrica", template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    st.header("Otras métricas de análisis")

    fig_all = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Content",
            "Wording",
            "Syntactic Complexity",
            "Semantic Complexity",
        ),
        vertical_spacing=0.20,
        horizontal_spacing=0.1,
    )

    # Content
    fig_all.add_trace(
        go.Scatter(
            x=filtered_df["len_mid"],
            y=filtered_df["content"],
            mode="lines+markers",
            name="Content",
            line=dict(color=DIST_COLOR_CONTENT),
            marker=dict(color=DIST_COLOR_CONTENT),
        ),
        row=1,
        col=1,
    )

    # Wording
    fig_all.add_trace(
        go.Scatter(
            x=filtered_df["len_mid"],
            y=filtered_df["wording"],
            mode="lines+markers",
            name="Wording",
            line=dict(color=DIST_COLOR_WORDING),
            marker=dict(color=DIST_COLOR_WORDING),
        ),
        row=1,
        col=2,
    )

    # Syntactic Complexity
    fig_all.add_trace(
        go.Scatter(
            x=filtered_df["len_mid"],
            y=filtered_df["syntactic_complexity"],
            mode="lines+markers",
            name="Syntactic",
            line=dict(color="#009E73"),
            marker=dict(color="#009E73"),
        ),
        row=2,
        col=1,
    )

    # Semantic Complexity
    fig_all.add_trace(
        go.Scatter(
            x=filtered_df["len_mid"],
            y=filtered_df["semantic_complexity"],
            mode="lines+markers",
            name="Semantic",
            line=dict(color="#D55E00"),
            marker=dict(color="#D55E00"),
        ),
        row=2,
        col=2,
    )

    fig_all.update_xaxes(title_text="Número de palabras", row=2, col=1)
    fig_all.update_xaxes(title_text="Número de palabras", row=2, col=2)
    fig_all.update_yaxes(title_text="Calificación", row=1, col=1)
    fig_all.update_yaxes(title_text="Calificación", row=1, col=2)
    fig_all.update_yaxes(title_text="Complejidad", row=2, col=1)
    fig_all.update_yaxes(title_text="Complejidad", row=2, col=2)

    fig_all.update_layout(height=700, showlegend=False, template="plotly_white")

    st.plotly_chart(fig_all, use_container_width=True)

    st.header("Matriz de Correlación")

    corr_matrix = filtered_df[
        ["content", "wording", "syntactic_complexity", "semantic_complexity", "len_mid"]
    ].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[[0, "#0072B2"], [0.5, "#FFFFFF"], [1, "#D55E00"]],
        labels=dict(color="Correlación"),
        title="Correlación entre Variables",
    )

    fig_corr.update_layout(template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)
