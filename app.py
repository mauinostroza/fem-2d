"""Router de la interfaz Streamlit: elige entre el modo lineal (placa con
agujero, SolidsPy) y el modo no lineal (sección de hormigón armado,
`fem2d_nl`). Cada modo vive en `ui/`; este archivo no contiene lógica de
dominio propia.
"""

import streamlit as st

from ui import linear_plate, rc_section

st.set_page_config(page_title="FEM 2D", layout="wide")

MODES = {
    "Placa con agujero (lineal)": linear_plate,
    "Sección de hormigón armado (no lineal)": rc_section,
}

mode_label = st.sidebar.radio("Modo de análisis", list(MODES.keys()))
st.sidebar.divider()

MODES[mode_label].render()
