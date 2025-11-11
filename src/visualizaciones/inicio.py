from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROMPT_COLOR_MAP = {
    "On Tragedy":                "#0072B2",
    "Egyptian Social Structure": "#E69F00",
    "The Third Wave":            "#009E73",
    "Excerpt from The Jungle":   "#D55E00",
}
DIST_COLOR_CONTENT = "#56B4E9"
DIST_COLOR_WORDING = "#CC79A7"
LINE_COLOR = "#F0E442"

APP_DIR = Path(__file__).resolve().parents[1]
ROOT    = APP_DIR.parent
DATA_CSV = ROOT / "data" / "Data_merged.csv"
DATA_PARQUET = ROOT / "data" / "notebook_data" / "merged_with_features.parquet"

@st.cache_data
def load_df():
    df = pd.read_csv(DATA_CSV)

    # Normaliza nombres esperados
    required = ["content", "wording", "prompt_title"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas {missing}. Asegúrate de que Data_merged.csv "
            "contenga 'content', 'wording' y 'prompt_title'."
        )
    # Limpieza mínima
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["content", "wording"])
    return df

# Correlación Pearson con manejo de NaN
def _pearson(x, y):
    if len(x) < 2:
        return np.nan
    return pd.Series(x).corr(pd.Series(y))

# Hace scatter con línea de tendencia (LOWESS o OLS)
def _scatter_with_trend(df, color_by, color_map):
    common_kwargs = dict(
        data_frame=df,
        x="content",
        y="wording",
        color=color_by,
        opacity=0.55,
        hover_data=["prompt_title"],
        color_discrete_map=color_map,
        category_orders={color_by: list(color_map.keys())}
    )

    try:
        fig = px.scatter(trendline="lowess", **common_kwargs)
    except Exception:
        try:
            fig = px.scatter(trendline="ols", **common_kwargs)
        except Exception:
            fig = px.scatter(**common_kwargs)
            if len(df) >= 2:
                coef = np.polyfit(df["content"], df["wording"], deg=1)
                xline = np.linspace(df["content"].min(), df["content"].max(), 100)
                yline = coef[0] * xline + coef[1]
                fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines", name="Tendencia (OLS)"))

    fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    fig.update_layout(legend_title_text="Prompt")
    return fig

# Histograma en densidad con curva KDE
def _hist_density(series, title, bar_color, line_color):
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    fig = px.histogram(
        x=series.dropna(),
        nbins=40,
        histnorm="probability density",
        opacity=0.85
    )
    fig.update_traces(marker=dict(color=bar_color, line=dict(width=0)))

    fig.update_layout(
        title=title,
        xaxis_title=title,
        yaxis_title="Densidad",
        showlegend=True,
        legend_title_text=""
    )

    try:
        x_min, x_max = float(series.min()), float(series.max())
        if x_min < 0 < x_max:
            fig.add_vline(x=0, line_dash="dash", line_color="#000000", opacity=0.4)
    except Exception:
        pass

    try:
        from scipy.stats import gaussian_kde
        s = series.dropna()
        if len(s) > 3 and (s.max() > s.min()):
            xs = np.linspace(s.min(), s.max(), 200)
            kde = gaussian_kde(s)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=kde(xs),
                    mode="lines",
                    name="Densidad (KDE)",
                    line=dict(width=3, color=line_color)
                )
            )
    except Exception:
        pass

    return fig

def render(st):
    st.title("Panorama general")
    st.caption("Vista global de *content* y *wording* con filtros por prompt.")

    # --- Carga de datos ---
    try:
        df = load_df()
    except Exception as e:
        st.error(str(e))
        return

    # --- Sidebar: filtro de prompt ---
    prompts = ["(Todos)"] + sorted(df["prompt_title"].dropna().unique().tolist())
    choice = st.sidebar.selectbox("Filtrar por prompt:", prompts, index=0)

    if choice != "(Todos)":
        dff = df[df["prompt_title"] == choice].copy()
    else:
        dff = df.copy()

    # Colores estables por prompt
    present_prompts = dff["prompt_title"].dropna().unique().tolist()
    color_map_present = {p: PROMPT_COLOR_MAP.get(p, "#000000") for p in present_prompts}

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("n° respuestas", f"{len(dff):,}")
    with col2:
        st.metric("content (mediana)", f"{dff['content'].median():.3f}")
    with col3:
        st.metric("wording (mediana)", f"{dff['wording'].median():.3f}")
    with col4:
        corr = _pearson(dff["content"], dff["wording"])
        st.metric("correlación content–wording", f"{corr:.3f}" if pd.notna(corr) else "—")

    st.divider()

    # --- Scatter content vs wording ---
    st.subheader("Relación entre rúbricas")
    st.caption("Nube de puntos con línea de tendencia.")
    fig_scatter = _scatter_with_trend(
        dff.assign(color_prompt=dff["prompt_title"]),
        color_by="color_prompt",
        color_map=color_map_present
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # --- Histogramas / densidades lado a lado ---
    st.subheader("Distribuciones")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Distribución de **content**")
        st.plotly_chart(
            _hist_density(dff["content"], "content",
                        bar_color=DIST_COLOR_CONTENT, line_color=LINE_COLOR),
            use_container_width=True
        )
    with c2:
        st.caption("Distribución de **wording**")
        st.plotly_chart(
            _hist_density(dff["wording"], "wording",
                        bar_color=DIST_COLOR_WORDING, line_color=LINE_COLOR),
            use_container_width=True
        )
