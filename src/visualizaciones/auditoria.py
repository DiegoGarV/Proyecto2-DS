import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

# Ruta del dataset
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_merged.csv"

PROMPT_COLOR_MAP = {
    "On Tragedy": "#0072B2",
    "Egyptian Social Structure": "#E69F00",
    "The Third Wave": "#009E73",
    "Excerpt from The Jungle": "#D55E00",
}
DIST_COLOR_CONTENT = "#56B4E9"
DIST_COLOR_WORDING = "#CC79A7"
LINE_COLOR = "#F0E442"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Detectar columna de texto disponible
    text_col = None
    for c in ["clean_text", "text", "summary", "student_summary"]:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        # Evitar crash si no hay texto; crea columnas de longitud vacías
        df["n_words"] = 0
        df["n_chars"] = 0
    else:
        s = df[text_col].fillna("").astype(str)
        if "n_words" not in df.columns:
            df["n_words"] = s.str.split().str.len()
        if "n_chars" not in df.columns:
            df["n_chars"] = s.str.len()

    return df


def render(st):
    st.title("Auditoría de Sesgos y Correlaciones")
    st.caption(
        "Explora relaciones entre longitud, métricas lingüísticas y calificaciones (*content* y *wording*)."
    )

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
        df,
        x="n_words",
        y="content",
        color="prompt_title",                       # <-- columna
        color_discrete_map=PROMPT_COLOR_MAP,       # <-- mapa de colores
        category_orders={"prompt_title": list(PROMPT_COLOR_MAP.keys())},
        opacity=0.6,
        labels={"n_words": "Longitud (número de palabras)", "content": "Calificación de Content"},
        title="Relación entre longitud y calificación de Content",
    )
    # línea de tendencia
    if len(df) > 2 and df["n_words"].nunique() > 1:
        coef = np.polyfit(df["n_words"], df["content"], 1)
        x_line = np.linspace(df["n_words"].min(), df["n_words"].max(), 100)
        y_line = coef[0] * x_line + coef[1]
        fig_len.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode="lines", name="Tendencia lineal",
                line=dict(color=LINE_COLOR, dash="dash"),
            )
        )
    st.plotly_chart(fig_len, use_container_width=True)


    fig_len2 = px.scatter(
        df,
        x="n_words",
        y="wording",
        color="prompt_title",
        color_discrete_map=PROMPT_COLOR_MAP,
        category_orders={"prompt_title": list(PROMPT_COLOR_MAP.keys())},
        opacity=0.6,
        labels={"n_words": "Longitud (número de palabras)", "wording": "Calificación de Wording"},
        title="Relación entre longitud y calificación de Wording",
    )
    if len(df) > 2 and df["n_words"].nunique() > 1:
        coef2 = np.polyfit(df["n_words"], df["wording"], 1)
        x_line2 = np.linspace(df["n_words"].min(), df["n_words"].max(), 100)
        y_line2 = coef2[0] * x_line2 + coef2[1]
        fig_len2.add_trace(
            go.Scatter(
                x=x_line2, y=y_line2,
                mode="lines", name="Tendencia lineal",
                line=dict(color=LINE_COLOR, dash="dash"),
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
        df,
        x="prompt_title",
        y="n_words",
        color="prompt_title",  # <-- clave: columna que colorea
        color_discrete_map=PROMPT_COLOR_MAP,  # tu diccionario de colores por prompt
        category_orders={"prompt_title": list(PROMPT_COLOR_MAP.keys())},  # orden/consistencia
        title="Distribución de longitud de resúmenes por prompt",
        labels={"n_words": "Número de palabras", "prompt_title": "Prompt"},
    )
    st.plotly_chart(fig_box, use_container_width=True)

