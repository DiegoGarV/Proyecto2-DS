import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_merged.csv"
PROMPT_COL = "prompt_title"
METRICS = ["content", "wording"]

PROMPT_COLOR_MAP = {
    "On Tragedy": "#0072B2",
    "Egyptian Social Structure": "#E69F00",
    "The Third Wave": "#009E73",
    "Excerpt from The Jungle": "#D55E00",
}
DIST_COLOR_CONTENT = "#56B4E9"
DIST_COLOR_WORDING = "#CC79A7"
LINE_COLOR = "#F0E442"


@st.cache_data(show_spinner=False)
def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Limpieza minima/estandares
    if PROMPT_COL not in df.columns:
        raise ValueError(f"Falta la columna '{PROMPT_COL}' en el dataset.")
    for m in METRICS:
        if m not in df.columns:
            raise ValueError(f"Falta la columna de metrica '{m}'.")
    return df


def _summary_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    def iqr(x):
        return (
            np.subtract(*np.percentile(x.dropna(), [75, 25]))
            if len(x.dropna())
            else np.nan
        )

    g = df.groupby(PROMPT_COL)[metric]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "media": g.mean(),
            "sigma": g.std(),
            "mediana": g.median(),
            "iqr": g.apply(iqr),
        }
    ).reset_index()
    return out


def render(st):
    st.title("Dificultad por prompt")
    st.caption(
        "Comparacion de **content** y **wording** por prompt: promedios, dispersion y distribucion."
    )

    df = _load_data(DATA_PATH)

    # Widgets
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        metric = st.selectbox("Metrica", METRICS, index=0)
    with c2:
        order_by = st.selectbox("Ordenar por", ["media", "mediana"], index=0)
    with c3:
        text_filter = st.text_input("Filtrar prompts (contiene)", value="").strip()

    # Resumen por prompt
    table = _summary_table(df, metric)

    if text_filter:
        table = table[table[PROMPT_COL].str.contains(text_filter, case=False, na=False)]

    # Orden
    asc = False  # de mayor a menor por defecto
    table_sorted = table.sort_values(order_by, ascending=asc)

    st.subheader("Promedio por prompt (±1σ)")
    fig_bar = px.bar(
        table_sorted,
        x=PROMPT_COL,
        y="media",
        error_y="sigma",
        hover_data=["n", "mediana", "iqr"],
        labels={"media": f"{metric} (media)", PROMPT_COL: "Prompt"},
        color=PROMPT_COL,
        color_discrete_map=PROMPT_COLOR_MAP,
    )
    fig_bar.update_layout(
        xaxis={
            "categoryorder": "array",
            "categoryarray": table_sorted[PROMPT_COL].tolist(),
        }
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Boxplots por prompt")
    # Para boxplots, filtramos por prompts presentes en la tabla ordenada (por si hubo filtro)
    prompts_keep = table_sorted[PROMPT_COL].unique().tolist()
    df_box = df[df[PROMPT_COL].isin(prompts_keep)]
    fig_box = px.box(
        df_box,
        x=PROMPT_COL,
        y=metric,
        points="outliers",
        labels={PROMPT_COL: "Prompt", metric: metric},
        color=PROMPT_COL,
        color_discrete_map=PROMPT_COLOR_MAP,
    )
    fig_box.update_layout(
        xaxis={"categoryorder": "array", "categoryarray": prompts_keep}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Tabla resumen"):
        st.dataframe(table_sorted, use_container_width=True)
        st.download_button(
            "Descargar CSV",
            data=table_sorted.to_csv(index=False).encode("utf-8"),
            file_name=f"resumen_por_prompt_{metric}.csv",
            mime="text/csv",
        )
