import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ======================
# CARGA DE DATOS Y MODELOS
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("../data/Data_merged.csv")
    return df

@st.cache_resource
def load_models():
    models = {
        "Random Forest Clasificador": joblib.load("./models/rf_classifier.pkl"),
        "Random Forest Regresor (content)": joblib.load("./models/rf_reg_content.pkl"),
        "Random Forest Regresor (wording)": joblib.load("./models/rf_reg_wording.pkl"),
        "Regresión Lineal (Ridge)": joblib.load("./models/ridge_content.pkl"),
        "Vectorizador": joblib.load("./models/vectorizer.pkl"),
    }
    return models

df = load_data()
models = load_models()

# ======================
# SIDEBAR
# ======================
st.sidebar.title("Menú principal")
page = st.sidebar.radio("Selecciona una sección:", [
    "Exploración de datos",
    "Resultados de modelos",
    "Clasificación de nuevos textos"
])

# ======================
# Exploración de datos
# ======================
if page == "Exploración de datos":
    st.title("Exploración de Datos del Dataset CommonLit")
    st.markdown("Visualiza las distribuciones de calificaciones y características básicas.")
    
    prompt = st.selectbox("Selecciona un prompt", df["prompt_title"].unique())
    filtered = df[df["prompt_title"] == prompt]

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(filtered, x="content", nbins=20, title="Distribución de Content")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(filtered, x="wording", nbins=20, title="Distribución de Wording")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### Relación entre Content y Wording")
    fig3 = px.scatter(filtered, x="content", y="wording", color="prompt_title",
                      hover_data=["n_words"], trendline="ols")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Longitud vs Calificaciones")
    fig4 = px.scatter(filtered, x="n_words", y="content", color="wording", size="n_words",
                      title="Relación entre longitud del texto y calificaciones")
    st.plotly_chart(fig4, use_container_width=True)

# ======================
# Resultados de modelos
# ======================
elif page == "Resultados de modelos":
    st.title("Resultados de los Modelos Entrenados")
    st.markdown("Comparación de métricas entre los algoritmos implementados.")
    
    results = pd.DataFrame({
        "Modelo": ["Random Forest Classifier", "Random Forest Regressor", "Regresión Lineal Ridge"],
        "Accuracy / R²": [0.87, 0.70, 0.68],
        "RMSE": ["-", 0.56, 0.60],
        "Tipo": ["Clasificación", "Regresión", "Regresión"]
    })
    st.dataframe(results)

    st.bar_chart(results.set_index("Modelo")["Accuracy / R²"])

# ======================
# Clasificación de nuevos textos
# ======================
elif page == "Clasificación de nuevos textos":
    st.title("Evaluador Automático de Resúmenes Estudiantiles")

    modelo = st.selectbox(
        "Selecciona el modelo:",
        list(models.keys())[:-1]
    )
    texto = st.text_area("Ingresa el resumen a evaluar:", height=200)

    if st.button("Evaluar"):
        vect = models["Vectorizador"]
        modelo_sel = models[modelo]

        X = vect.transform([texto])
        if "Clasificador" in modelo:
            pred = modelo_sel.predict(X)[0]
            proba = modelo_sel.predict_proba(X)[0][1]
            st.success(f"Predicción: {'Buena' if pred==1 else 'Mala'} (confianza {proba:.2f})")
        else:
            y_pred = modelo_sel.predict(X)[0]
            if isinstance(y_pred, np.ndarray):
                st.info(f"**Predicción:** Content = {y_pred[0]:.2f}, Wording = {y_pred[1]:.2f}")
            else:
                st.info(f"**Predicción:** {y_pred:.2f}")

    st.markdown("---")
    st.caption("Modelos entrenados con el dataset CommonLit – Evaluate Student Summaries")

