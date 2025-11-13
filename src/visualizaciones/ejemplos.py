import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

PROMPT_COLOR_MAP = {
    "On Tragedy": "#0072B2",
    "Egyptian Social Structure": "#E69F00",
    "The Third Wave": "#009E73",
    "Excerpt from The Jungle": "#D55E00",
}
DIST_COLOR_CONTENT = "#56B4E9"
DIST_COLOR_WORDING = "#CC79A7"
LINE_COLOR = "#F0E442"

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_merged.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "n_words" not in df.columns:
        df["n_words"] = df["text"].astype(str).str.split().apply(len)
    df["promedio"] = df[["content", "wording"]].mean(axis=1)
    return df


def render(st):
    st.title("Galería de Ejemplos")
    st.caption(
        "Explora ejemplos representativos de resúmenes según sus calificaciones."
    )

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return

    # Filtros
    prompts = ["(Todos)"] + sorted(df["prompt_title"].dropna().unique().tolist())
    prompt_sel = st.sidebar.selectbox("Filtrar por prompt:", prompts, index=0)

    if prompt_sel != "(Todos)":
        df = df[df["prompt_title"] == prompt_sel]

    st.sidebar.markdown("### Rango de calificaciones")
    c_min, c_max = st.sidebar.slider(
        "Content",
        float(df["content"].min()),
        float(df["content"].max()),
        (float(df["content"].quantile(0.25)), float(df["content"].quantile(0.75))),
    )
    w_min, w_max = st.sidebar.slider(
        "Wording",
        float(df["wording"].min()),
        float(df["wording"].max()),
        (float(df["wording"].quantile(0.25)), float(df["wording"].quantile(0.75))),
    )

    filtered = df[
        (df["content"].between(c_min, c_max)) & (df["wording"].between(w_min, w_max))
    ]

    st.divider()
    st.subheader("Tabla de ejemplos filtrados")
    st.dataframe(
        filtered[["prompt_title", "content", "wording", "n_words"]]
        .sort_values("content", ascending=False)
        .head(50),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Ejemplo representativo")

    if len(filtered) > 0:
        # ejemplo más cercano al promedio
        target_c = filtered["content"].median()
        target_w = filtered["wording"].median()
        filtered["dist"] = np.abs(filtered["content"] - target_c) + np.abs(
            filtered["wording"] - target_w
        )
        example = filtered.sort_values("dist").iloc[0]

        st.metric("Prompt", example["prompt_title"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Content", f"{example['content']:.2f}")
        col2.metric("Wording", f"{example['wording']:.2f}")
        col3.metric("Longitud (palabras)", f"{example['n_words']}")

        with st.expander("Ver resumen completo"):
            st.write(example["text"])
    else:
        st.warning("No hay ejemplos en el rango seleccionado.")

    st.divider()

    st.subheader("Distribución de calificaciones (vista rápida)")

    df_melt = df.melt(
        value_vars=["content", "wording"], var_name="metric", value_name="score"
    )

    fig = px.histogram(
        df_melt,
        x="score",
        color="metric",
        barmode="overlay",
        opacity=0.6,
        color_discrete_map={
            "content": DIST_COLOR_CONTENT,
            "wording": DIST_COLOR_WORDING,
        },
        labels={"score": "Calificación", "metric": "Métrica"},
        title="Distribución de Content y Wording",
    )

    fig.update_layout(showlegend=True, bargap=0.1)

    st.plotly_chart(fig, use_container_width=True)
