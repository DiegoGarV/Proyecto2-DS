
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk

nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize

model = joblib.load("modelo_resumen.pkl")
vectorizer = joblib.load("vectorizer_resumen.pkl")

def calcular_caracteristicas(texto):
    """Calcula métricas básicas de un texto."""
    # Tokenización básica
    palabras = word_tokenize(texto)
    oraciones = sent_tokenize(texto)

    num_palabras = len(palabras)
    num_caracteres = len(texto)
    promedio_longitud_palabra = np.mean([len(p) for p in palabras]) if palabras else 0
    promedio_longitud_oracion = np.mean([len(o.split()) for o in oraciones]) if oraciones else 0
    riqueza_lexica = len(set(palabras)) / num_palabras if num_palabras > 0 else 0

    return {
        "Número de caracteres": num_caracteres,
        "Número de palabras": num_palabras,
        "Promedio de longitud de palabra": round(promedio_longitud_palabra, 2),
        "Promedio de longitud de oración": round(promedio_longitud_oracion, 2),
        "Riqueza léxica (diversidad de palabras)": round(riqueza_lexica, 2)
    }
def render(st):
    st.title("Predicción interactiva")
    st.title("Clasificador de Resúmenes")
    st.write("Esta herramienta predice si un resumen es **bueno o no**, basado en tu modelo entrenado.")

    # Entrada de texto
    user_input = st.text_area("✍️ Escribe el resumen aquí:", height=200)

    if st.button("Predecir"):
        if user_input.strip() == "":
            st.warning("Por favor escribe un resumen para analizarlo.")
        else:
            # Vectorizar texto del usuario
            X_user = vectorizer.transform([user_input])
            pred = model.predict(X_user)[0]
            proba = model.predict_proba(X_user)[0]

            # Mostrar resultado
            st.subheader("Resultado de la predicción:")
            if pred == 1:  
                st.success(f"✅ Es un **buen resumen** (probabilidad {proba[1]*100:.2f}%)")
            else:
                st.error(f"No parece un buen resumen (probabilidad {proba[0]*100:.2f}%)")

            # Mostrar probabilidades
            st.write("Distribución de probabilidad:", {
                "No bueno": round(proba[0]*100, 2),
                "Bueno": round(proba[1]*100, 2)
            })

            caracteristicas = calcular_caracteristicas(user_input)
            st.table(pd.DataFrame(caracteristicas.items(), columns=["Característica", "Valor"]))

