import streamlit as st
import pandas as pd
import plotly.express as px

def render(st):
    st.title("📊 Resultados de Modelos y Complejidades")
    st.write("Visualización de cómo las características del resumen (longitud y complejidad) influyen en las calificaciones promedio.")

    # === Cargar datos ===
    df = pd.read_csv("visualizaciones/analisis_resumenes.csv")

    # Asegurar tipos numéricos
    numeric_cols = [
        "len_mid",
        "syntactic_complexity",
        "semantic_complexity",
        "content_by_len",
        "wording_by_len"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # === Seleccionar característica a analizar ===
    caracteristicas = {
        "Longitud del resumen": "len_mid",
        "Complejidad sintáctica": "syntactic_complexity",
        "Complejidad semántica": "semantic_complexity"
    }

    selected_feature_label = st.selectbox(
        "Selecciona la característica a analizar:",
        list(caracteristicas.keys())
    )
    selected_feature = caracteristicas[selected_feature_label]

    # === Seleccionar métricas a mostrar ===
    metricas = []
    if "content_by_len" in df.columns:
        metricas.append("content_by_len")
    if "wording_by_len" in df.columns:
        metricas.append("wording_by_len")

    selected_metrics = st.multiselect(
        "Selecciona las métricas a mostrar:",
        metricas,
        default=metricas
    )

    # === Slider para filtrar rango de la característica ===
    min_val, max_val = float(df[selected_feature].min()), float(df[selected_feature].max())
    rango = st.slider(
        f"Selecciona el rango para {selected_feature_label.lower()}",
        float(min_val),
        float(max_val),
        (float(min_val), float(max_val))
    )

    # Filtrar por rango
    filtered_df = df[
        (df[selected_feature] >= rango[0]) & (df[selected_feature] <= rango[1])
    ]

    # === Graficar ===
    fig = px.line(
        filtered_df,
        x=selected_feature,
        y=selected_metrics,
        markers=True,
        labels={
            selected_feature: selected_feature_label,
            "value": "Promedio de calificación",
            "variable": "Métrica"
        },
        title=f"Relación entre {selected_feature_label.lower()} y las calificaciones promedio"
    )

    fig.update_layout(
        legend_title_text="Métrica",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

   