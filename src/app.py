# BLOQUE — src/app.py
import sys
from pathlib import Path
import importlib
import streamlit as st

# --- Rutas robustas (independientes del working dir) ---
APP_DIR = Path(__file__).resolve().parent  # .../src
ROOT = APP_DIR.parent  # proyecto
VIS_DIR = APP_DIR / "visualizaciones"

# Asegurar que src esté en sys.path para poder importar visualizaciones.*
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# --- Config básica de la app ---
st.set_page_config(
    page_title="Proyecto2-DS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Definir páginas (Título visible -> módulo a importar) ---
PAGES = {
    "Inicio": "visualizaciones.inicio",
    "Prompts": "visualizaciones.prompts",
    "Longitud y estructura": "visualizaciones.long_estruct",
    "Explicabilidad": "visualizaciones.explicabilidad",
    "Comparador de modelos": "visualizaciones.modelos",
    "Predicción interactiva": "visualizaciones.prediccion",
    "Auditoría": "visualizaciones.auditoria",
    "Ejemplos": "visualizaciones.ejemplos",
}

# --- Sidebar: selector de página ---
st.sidebar.title("Navegación")
page_title = st.sidebar.selectbox("Ir a:", list(PAGES.keys()), index=0)

# --- Cargar y ejecutar la página seleccionada ---
module_name = PAGES[page_title]
module = importlib.import_module(module_name)

# Cada página expone una función render(st)
if hasattr(module, "render"):
    module.render(st)
else:
    st.error(f"La página '{module_name}' no define la función render(st).")

# --- Pie simple ---
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto2-DS • Streamlit multipágina")
