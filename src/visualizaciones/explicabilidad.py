import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_merged.csv"
PROMPT_COL = "prompt_title"
TEXT_COL = "text"
TARGETS = ["content", "wording"]

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
    # Validaciones minimas
    if PROMPT_COL not in df.columns:
        raise ValueError(f"Falta la columna '{PROMPT_COL}'.")
    if TEXT_COL not in df.columns:
        # Fallback heuristico si no hay 'text': busca una columna tipo texto larga
        text_like = [c for c in df.columns if df[c].dtype == object]
        if not text_like:
            raise ValueError("No encontre columna de texto. Ajusta TEXT_COL.")
        df = df.rename(columns={text_like[0]: TEXT_COL})
    for t in TARGETS:
        if t not in df.columns:
            raise ValueError(f"Falta la columna '{t}'.")
    df = df.dropna(subset=[TEXT_COL]).copy()
    return df


def _get_prompts(df: pd.DataFrame):
    opts = ["(Global)"] + sorted(df[PROMPT_COL].dropna().unique().tolist())
    return opts


# ---------- Modelado ligero y cache ----------
def _make_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
    )


def _fit_models(df: pd.DataFrame):
    # Crea un vectorizador y dos modelos Ridge (uno para cada target)
    vec = _make_vectorizer()
    X = vec.fit_transform(df[TEXT_COL])

    models = {}
    for tgt in TARGETS:
        y = df[tgt].astype(float)
        m = Ridge(alpha=1.0, random_state=42)
        m.fit(X, y)
        yhat = m.predict(X)
        models[tgt] = {"model": m, "r2_in_sample": float(r2_score(y, yhat))}
    return vec, models


@st.cache_resource(show_spinner=False)
def _get_cached_models(df_key: str, data_for_models: pd.DataFrame):
    # df_key debe cambiar cuando se filtra por prompt
    vec, models = _fit_models(data_for_models)
    return vec, models


# ---------- Utilidades de inspeccion ----------
def _top_coefficients(vectorizer, model, k=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_.ravel()
    order_pos = np.argsort(coefs)[::-1][:k]
    order_neg = np.argsort(coefs)[:k]
    top_pos = pd.DataFrame(
        {"token": feature_names[order_pos], "weight": coefs[order_pos]}
    )
    top_neg = pd.DataFrame(
        {"token": feature_names[order_neg], "weight": coefs[order_neg]}
    )
    return top_pos, top_neg


def _explain_text(vectorizer, model, text: str, k=20):
    # tokens en el texto intersectados con features del vectorizador y ordenados por |peso|
    if not text:
        return pd.DataFrame(columns=["token", "weight", "present"])
    feats = set(vectorizer.get_feature_names_out())
    # tokenizacion simple alineada con TfidfVectorizer(lowercase/strip)
    words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
    # tambien generamos bigrams simples
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    tokens_text = Counter([w for w in words + bigrams if w in feats])

    # pesos del modelo
    coefs = dict(zip(vectorizer.get_feature_names_out(), model.coef_.ravel()))
    rows = [
        {"token": t, "weight": coefs.get(t, 0.0), "freq": c}
        for t, c in tokens_text.items()
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values(by=["weight", "freq"], ascending=[False, False])
        .head(k)
    )
    return df


# ---------- Vista principal ----------
def render(st):
    st.title("Explicabilidad")

    # Carga de datos
    if not DATA_PATH.exists():
        st.warning(f"No se encontro {DATA_PATH}. Sube el CSV para continuar.")
        up = st.file_uploader("Sube Data_merged.csv", type=["csv"])
        if up is None:
            st.stop()
        df = pd.read_csv(up)
    else:
        df = _load_data(DATA_PATH)

    # Filtro por prompt (para entrenar el modelo con subset)
    prompts = _get_prompts(df)
    sel = st.selectbox("Contexto del modelo", prompts, index=0)
    if sel != "(Global)":
        df_sub = df[df[PROMPT_COL] == sel].copy()
        df_key = f"prompt::{sel}::{len(df_sub)}"
        st.info(f"Modelo entrenado solo con el prompt **{sel}** (n={len(df_sub)}).")
        if len(df_sub) < 50:
            st.warning(
                "Datos escasos para este prompt; las señales pueden ser inestables."
            )
    else:
        df_sub = df
        df_key = f"global::{len(df_sub)}"

    # Entrenar/recuperar modelos cacheados
    with st.spinner("Entrenando/recuperando modelos…"):
        vectorizer, models = _get_cached_models(df_key, df_sub[[TEXT_COL] + TARGETS])

    # Panel: Top tokens por metrica (±)
    st.subheader("Tokens mas influyentes por metrica")
    c1, c2 = st.columns(2)
    with c1:
        target_a = st.selectbox("Metrica A", TARGETS, index=0, key="tgtA")
        topA_pos, topA_neg = _top_coefficients(
            vectorizer, models[target_a]["model"], k=20
        )

        st.markdown(f"**Top +{target_a}**")
        st.dataframe(topA_pos, use_container_width=True, hide_index=True)
        # Barras para los top positivos
        figA_pos = px.bar(
            topA_pos,
            x="token",
            y="weight",
            title=f"Top +{target_a}",
            labels={"token": "Token / n-grama", "weight": "Peso"},
            color_discrete_sequence=[
                DIST_COLOR_CONTENT if target_a == "content" else DIST_COLOR_WORDING
            ],
        )
        figA_pos.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(
            figA_pos,
            use_container_width=True,
            key=f"figA_pos_{target_a}",
        )

        st.markdown(f"**Top –{target_a}**")
        st.dataframe(topA_neg, use_container_width=True, hide_index=True)
        # Barras para los top negativos
        figA_neg = px.bar(
            topA_neg,
            x="token",
            y="weight",
            title=f"Top –{target_a}",
            labels={"token": "Token / n-grama", "weight": "Peso"},
            color_discrete_sequence=[
                DIST_COLOR_CONTENT if target_a == "content" else DIST_COLOR_WORDING
            ],
        )
        figA_neg.update_layout(xaxis={"categoryorder": "total ascending"})
        st.plotly_chart(
            figA_neg,
            use_container_width=True,
            key=f"figA_neg_{target_a}",
        )

        st.caption(f"R² (in-sample) {target_a}: {models[target_a]['r2_in_sample']:.3f}")

    with c2:
        target_b = st.selectbox("Metrica B", TARGETS, index=1, key="tgtB")
        topB_pos, topB_neg = _top_coefficients(
            vectorizer, models[target_b]["model"], k=20
        )

        st.markdown(f"**Top +{target_b}**")
        st.dataframe(topB_pos, use_container_width=True, hide_index=True)
        # Barras para los top positivos
        figB_pos = px.bar(
            topB_pos,
            x="token",
            y="weight",
            title=f"Top +{target_b}",
            labels={"token": "Token / n-grama", "weight": "Peso"},
            color_discrete_sequence=[
                DIST_COLOR_CONTENT if target_b == "content" else DIST_COLOR_WORDING
            ],
        )
        figB_pos.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(
            figB_pos,
            use_container_width=True,
            key=f"figB_pos_{target_b}",
        )

        st.markdown(f"**Top –{target_b}**")
        st.dataframe(topB_neg, use_container_width=True, hide_index=True)
        # Barras para los top negativos
        figB_neg = px.bar(
            topB_neg,
            x="token",
            y="weight",
            title=f"Top –{target_b}",
            labels={"token": "Token / n-grama", "weight": "Peso"},
            color_discrete_sequence=[
                DIST_COLOR_CONTENT if target_b == "content" else DIST_COLOR_WORDING
            ],
        )
        figB_neg.update_layout(xaxis={"categoryorder": "total ascending"})
        st.plotly_chart(
            figB_neg,
            use_container_width=True,
            key=f"figB_neg_{target_b}",
        )

        st.caption(f"R² (in-sample) {target_b}: {models[target_b]['r2_in_sample']:.3f}")

    st.divider()

    # Buscador de token
    st.subheader("Buscador de token")
    token_query = st.text_input(
        "Busca un token/ngrama exacto (minusculas, sin tildes):", ""
    )
    if token_query:
        feats = set(vectorizer.get_feature_names_out())
        if token_query in feats:
            wa = models[TARGETS[0]]["model"].coef_.ravel()[
                list(vectorizer.get_feature_names_out()).index(token_query)
            ]
            wb = models[TARGETS[1]]["model"].coef_.ravel()[
                list(vectorizer.get_feature_names_out()).index(token_query)
            ]
            st.success(
                f"Peso en {TARGETS[0]}: {wa:.4f} | Peso en {TARGETS[1]}: {wb:.4f}"
            )
        else:
            st.warning("Ese token no esta en el vocabulario del vectorizador actual.")

    st.divider()
