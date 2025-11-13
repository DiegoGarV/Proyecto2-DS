import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import plotly.express as px
import joblib
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


# ---------- Vista principal ----------
def render(st):
    st.title("Explicabilidad")
    st.caption(
        "Inspeccion de tokens mas influyentes (TF-IDF + Ridge). Puedes filtrar por prompt o usar el modelo global."
    )

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

    # Cargar modelo/vectorizador ya entrenados y construir estructura 'models'
    with st.spinner("Cargando modelo entrenado…"):
        model = joblib.load("src/modelo_resumen.pkl")
        vectorizer = joblib.load("src/vectorizer_resumen.pkl")

        X_sub = vectorizer.transform(df_sub[TEXT_COL])
        y_sub = df_sub[TARGETS].values

        try:
            y_pred = model.predict(X_sub)
        except Exception:
            y_pred = None

        coefs = getattr(model, "coef_", None)
        models = {}

        if coefs is not None:
            coefs = np.array(coefs)

            if coefs.ndim == 2 and coefs.shape[0] >= len(TARGETS):
                for i, tgt in enumerate(TARGETS):

                    class ModelWrapper:
                        def __init__(self, coef_row):
                            self.coef_ = np.array(coef_row)

                    mwrap = ModelWrapper(coefs[i])
                    r2_val = (
                        float(r2_score(y_sub[:, i], y_pred[:, i]))
                        if (y_pred is not None and np.ndim(y_pred) == 2)
                        else np.nan
                    )
                    models[tgt] = {"model": mwrap, "r2_in_sample": r2_val}

            elif coefs.ndim == 1:

                class ModelWrapper:
                    def __init__(self, coef_vec):
                        self.coef_ = np.array(coef_vec)

                for tgt in TARGETS:
                    models[tgt] = {"model": ModelWrapper(coefs), "r2_in_sample": np.nan}
        else:
            st.error(
                "El modelo cargado no tiene atributo 'coef_'. No puedo calcular importancias de tokens."
            )
            return

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
