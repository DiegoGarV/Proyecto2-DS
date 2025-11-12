import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

# Ruta del dataset
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_merged.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Calculamos longitud del texto si no existe
    if "n_words" not in df.columns:
        df["n_words"] = df["text"].astype(str).str.split().apply(len)
    if "n_chars" not in df.columns:
        df["n_chars"] = df["text"].astype(str).str.len()
    return df

def render(st):
    st.title("Auditoría de Sesgos y Correlaciones")
    st.caption("Explora relaciones entre longitud, métricas lingüísticas y calificaciones (*content* y *wording*).")

    # Cargar datos
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return

    # Sidebar filtros
    prompts = ["(Todos)"] + sorted(df["prompt_title"].dropna().unique().tolist())
    prompt_sel = st.sidebar.selectbox("Filtrar por prompt:", prompts, index=0)

    if prompt_sel != "(Todos)":
        df = df[df["prompt_title"] == prompt_sel]

    st.divider()
    st.subheader("Tendencia de las calificaciones según la longitud del resumen")

    # Longitud vs Content/Wording
    fig_len = px.scatter(
        df, x="n_words", y="content",
        color="prompt_title",
        opacity=0.6,
        labels={"n_words": "Longitud (número de palabras)", "content": "Calificación de Content"},
        title="Relación entre longitud y calificación de Content"
    )

    # línea de tendencia simple
    if len(df) > 2:
        coef = np.polyfit(df["n_words"], df["content"], 1)
        x_line = np.linspace(df["n_words"].min(), df["n_words"].max(), 100)
        y_line = coef[0]*x_line + coef[1]
        fig_len.add_trace(
            go.Scatter(
                x=x_line, y=y_line, mode="lines", name="Tendencia lineal",
                line=dict(color="black", dash="dash")
            )
        )

    st.plotly_chart(fig_len, use_container_width=True)

    fig_len2 = px.scatter(
        df, x="n_words", y="wording",
        color="prompt_title",
        opacity=0.6,
        labels={"n_words": "Longitud (número de palabras)", "wording": "Calificación de Wording"},
        title="Relación entre longitud y calificación de Wording"
    )

    # Agregar tendencia manual
    if len(df) > 2:
        coef2 = np.polyfit(df["n_words"], df["wording"], 1)
        x_line2 = np.linspace(df["n_words"].min(), df["n_words"].max(), 100)
        y_line2 = coef2[0]*x_line2 + coef2[1]
        fig_len2.add_trace(
            go.Scatter(
                x=x_line2, y=y_line2, mode="lines", name="Tendencia lineal",
                line=dict(color="black", dash="dash")
            )
        )

    st.plotly_chart(fig_len2, use_container_width=True)

    st.divider()
    st.subheader("Correlaciones entre métricas y calificaciones")

    numeric_cols = ["n_words", "n_chars", "content", "wording"]
    corr = df[numeric_cols].corr().round(3)
    st.dataframe(corr, use_container_width=True)
    st.caption("Correlaciones de Pearson entre métricas simples y las calificaciones.")

    st.divider()
    st.subheader("Distribución de longitudes por prompt")

    fig_box = px.box(
        df, x="prompt_title", y="n_words", color="prompt_title",
        title="Distribución de longitud de resúmenes por prompt",
        labels={"n_words": "Número de palabras", "prompt_title": "Prompt"}
    )
    st.plotly_chart(fig_box, use_container_width=True)
