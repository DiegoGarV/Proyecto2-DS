from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Importa colores
from .inicio import (
    PROMPT_COLOR_MAP,
    DIST_COLOR_CONTENT,
    DIST_COLOR_WORDING,
)

APP_DIR = Path(__file__).resolve().parents[1]
ROOT    = APP_DIR.parent
DATA_CSV = ROOT / "data" / "Data_merged.csv"
DATA_PARQUET = ROOT / "data" / "notebook_data" / "merged_with_features.parquet"
NOTEBOOK_DATA_DIR = ROOT / "data" / "notebook_data"
NOTEBOOK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Carga DataFrame con features de longitud
@st.cache_data
def load_df_with_features():
    """
    Carga parquet con features si existe; si no, lee el CSV base y
    calcula word_count/char_count/sentence_count/wps.
    Luego guarda el parquet para reuso en data/notebook_data/.
    """
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    else:
        df = pd.read_csv(DATA_CSV)

        # Garantizar columnas mínimas
        req = ["content", "wording", "prompt_title"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(
                f"Faltan columnas requeridas {miss} en {DATA_CSV.name}."
            )

        # Texto base para calcular métricas de longitud
        text_col_candidates = ["clean_text", "text", "summary", "student_summary"]
        text_col = next((c for c in text_col_candidates if c in df.columns), None)

        # Calcula features de longitud
        if "word_count" not in df.columns or "char_count" not in df.columns \
           or "sentence_count" not in df.columns or "wps" not in df.columns:

            if text_col is None:
                tmp = pd.Series([""] * len(df))
            else:
                tmp = df[text_col].fillna("").astype(str)

            df = df.copy()
            df["char_count"]     = tmp.str.len()
            df["word_count"]     = tmp.str.split().str.len()
            df["sentence_count"] = tmp.str.count(r"[\.!\?]") + 1
            df.loc[tmp.str.strip().eq(""), "sentence_count"] = 0
            df["wps"] = df["word_count"] / df["sentence_count"].replace(0, np.nan)

        # Guarda parquet para futuros arranques
        try:
            df.to_parquet(DATA_PARQUET, index=False)
        except Exception:
            pass

    # Limpieza mínima
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["content", "wording"])
    return df

# Normaliza content/wording por prompt (z-score)
def normalize_by_prompt(dff, cols=("content", "wording")):
    g = dff.groupby("prompt_title")
    out = dff.copy()
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, np.nan)
        out[f"{c}_z"] = (dff[c] - mu) / sd
    return out

# Scatter word_count vs metric con línea de tendencia
def scatter_wc_vs_metric(dff, metric_col, color_map):
    common = dict(
        data_frame=dff,
        x="word_count",
        y=metric_col,
        color="prompt_title",
        opacity=0.55,
        hover_data=["prompt_title"],
        color_discrete_map=color_map,
        category_orders={"prompt_title": list(color_map.keys())},
    )
    try:
        fig = px.scatter(trendline="lowess", **common)
    except Exception:
        try:
            fig = px.scatter(trendline="ols", **common)
        except Exception:
            fig = px.scatter(**common)
            if len(dff) >= 2 and dff["word_count"].nunique() > 1:
                coef = np.polyfit(dff["word_count"], dff[metric_col], deg=1)
                xs = np.linspace(dff["word_count"].min(), dff["word_count"].max(), 100)
                ys = coef[0] * xs + coef[1]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Tendencia (OLS)"))

    fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    fig.update_layout(
        xaxis_title="word_count",
        yaxis_title=metric_col,
        legend_title_text="Prompt",
    )
    return fig

def render(st):
    st.title("Longitud y estructura")
    st.caption("¿Cuánto influye la longitud (word_count) y la estructura en las calificaciones?")

    # 1) Carga
    try:
        df = load_df_with_features()
    except Exception as e:
        st.error(str(e))
        return

    # 2) Filtros y opciones
    prompts = ["(Todos)"] + sorted(df["prompt_title"].dropna().unique().tolist())
    sel_prompt = st.sidebar.selectbox("Filtrar por prompt:", prompts, index=0)

    metric_choice = st.sidebar.radio(
        "Métrica a analizar:", ["content", "wording"], horizontal=True, index=0
    )

    norm_by_prompt = st.sidebar.toggle(
        "Normalizar scores por prompt (z-score)", value=False,
        help="Controla por 'dificultad' del prompt estandarizando content/wording por prompt."
    )

    # Slider de rango de palabras
    p1, p99 = np.nanpercentile(df["word_count"], [1, 99])
    wc_min, wc_max = st.sidebar.slider(
        "Rango de word_count:", int(p1), int(p99), (int(p1), int(p99)), step=1
    )

    # 3) Subset según prompt y rango
    if sel_prompt != "(Todos)":
        dff = df[df["prompt_title"] == sel_prompt].copy()
    else:
        dff = df.copy()

    dff = dff[(dff["word_count"] >= wc_min) & (dff["word_count"] <= wc_max)].copy()

    # Colores fijos por prompts presentes
    present_prompts = dff["prompt_title"].dropna().unique().tolist()
    color_map_present = {p: PROMPT_COLOR_MAP.get(p, "#000000") for p in present_prompts}

    # 4) Normalización
    plot_metric = metric_choice
    if norm_by_prompt:
        dff = normalize_by_prompt(dff, cols=("content", "wording"))
        plot_metric = f"{metric_choice}_z"

    # 5) KPIs
    st.subheader("Resumen")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("n° respuestas", f"{len(dff):,}")
    with c2:
        st.metric("correlación (word_count, " + metric_choice + ")",
                  f"{pd.Series(dff['word_count']).corr(pd.Series(dff[plot_metric])):.3f}")
    with c3:
        st.metric("mediana de " + metric_choice, f"{dff[plot_metric].median():.3f}")

    st.divider()

    # 6) Scatter con tendencia
    st.subheader("Relación longitud con puntaje")
    st.caption("Scatter de word_count vs {} con línea de tendencia.".format(metric_choice))
    fig_sc = scatter_wc_vs_metric(dff, plot_metric, color_map_present)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # 7) Heatmap 2D (bines)
    st.subheader("Mapa de densidad (bines)")
    st.caption("Densidad de observaciones en la cuadrícula word_count × {}.".format(metric_choice))
    # nbins adaptativos
    nbx = max(10, min(40, int(np.sqrt(len(dff)) / 2)))
    nby = nbx
    fig_heat = px.density_heatmap(
        dff,
        x="word_count", y=plot_metric,
        nbinsx=nbx, nbinsy=nby,
        color_continuous_scale="Blues",
        histfunc="count",
    )
    fig_heat.update_layout(
        xaxis_title="word_count (palabras)",
        yaxis_title=plot_metric,
        coloraxis_colorbar_title="número de obsercaiones"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # 8) Barras por quintil de longitud
    st.subheader("Promedio por quintil de longitud")
    st.caption("Se agrupa **word_count** en 5 quintiles y se promedia {} en cada uno.".format(metric_choice))
    try:
        dff = dff.copy()
        dff["wc_quintil"] = pd.qcut(dff["word_count"], 5, labels=["Q1","Q2","Q3","Q4","Q5"])
        bars = (dff.groupby("wc_quintil")[plot_metric]
                    .mean()
                    .reindex(["Q1","Q2","Q3","Q4","Q5"])
                    .reset_index())
        # Usa colores de distribución para diferenciar de los scatters
        bar_color = DIST_COLOR_CONTENT if metric_choice == "content" else DIST_COLOR_WORDING
        fig_bar = px.bar(bars, x="wc_quintil", y=plot_metric, text=plot_metric)
        fig_bar.update_traces(marker_color=bar_color, texttemplate="%{text:.3f}", textposition="outside")
        fig_bar.update_layout(
            xaxis_title="Quintil de word_count",
            yaxis_title=("{} (z-score)".format(metric_choice) if norm_by_prompt else metric_choice),
            uniformtext_minsize=8, uniformtext_mode="hide",
            yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor="#777"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.info("No fue posible calcular quintiles de word_count en este filtro.")