"""Interfaz Streamlit: análisis FEM de una placa con agujero (SolidsPy).

Orquesta los módulos de fem2d; no contiene lógica de dominio propia.
"""

import io

import streamlit as st

from fem2d.boundary_conditions import (
    Edge,
    EdgeCondition,
    EdgeLoad,
    Support,
    build_cons_array,
    build_loads_array,
)
from fem2d.exceptions import BoundaryConditionError, GeometryError, SolverError
from fem2d.geometry import PlateGeometryParams, build_plate_mesh
from fem2d.materials import MATERIAL_PRESETS, MaterialProps
from fem2d.solver import run_analysis
from fem2d.visualization import plot_deformed, plot_mesh, plot_von_mises, summary_dataframe

st.set_page_config(page_title="FEM 2D — Placa con agujero", layout="wide")

SUPPORT_LABELS = {
    "Libre": Support.FREE,
    "Empotrado (fijo x,y)": Support.FIXED,
    "Fijo en X (desliza Y)": Support.ROLLER_X,
    "Fijo en Y (desliza X)": Support.ROLLER_Y,
}
DIRECTION_LABELS = {
    "Normal saliente (tracción positiva)": "normal",
    "Horizontal (x)": "x",
    "Vertical (y)": "y",
}
EDGE_ORDER = [Edge.LEFT, Edge.RIGHT, Edge.TOP, Edge.BOTTOM]

st.title("Análisis FEM de placa con agujero")
st.caption(
    "Unidades del proyecto: longitud en mm, esfuerzo y módulo de Young en MPa, "
    "fuerza en N (espesor unitario, formulación de tensión plana)."
)

st.sidebar.subheader("Material")
# Fuera del form: seleccionar el preset aquí sí dispara un rerun inmediato,
# así los campos de E/ν abajo reflejan la elección desde el primer envío.
preset_name = st.sidebar.selectbox("Preset de material", list(MATERIAL_PRESETS.keys()))

with st.sidebar.form("parametros"):
    st.subheader("Geometría")
    col1, col2 = st.columns(2)
    length = col1.number_input("Largo L (mm)", min_value=0.1, value=100.0, step=10.0)
    height = col2.number_input("Alto H (mm)", min_value=0.1, value=50.0, step=10.0)
    hole_radius = st.number_input(
        "Radio del agujero R (mm)", min_value=0.0, value=10.0, step=1.0,
        help="0 = placa sin agujero.",
    )
    mesh_size = st.number_input("Tamaño de malla (mm)", min_value=0.01, value=4.0, step=0.5)
    with st.expander("Avanzado: refinamiento cerca del agujero"):
        hole_mesh_size = st.number_input(
            "Tamaño de malla en el agujero (mm)",
            min_value=0.0,
            value=max(mesh_size / 3, 0.1),
            step=0.1,
            help="0 = usar automáticamente un tercio del tamaño de malla general.",
        )

    st.subheader("Material")
    st.caption(f"Preset seleccionado: {preset_name}")
    if MATERIAL_PRESETS[preset_name] is None:
        col3, col4 = st.columns(2)
        young_modulus = col3.number_input("Módulo de Young E (MPa)", min_value=1.0, value=200_000.0)
        poisson_ratio = col4.number_input(
            "Coeficiente de Poisson ν", min_value=0.0, max_value=0.499, value=0.3
        )
    else:
        preset = MATERIAL_PRESETS[preset_name]
        young_modulus, poisson_ratio = preset.young_modulus, preset.poisson_ratio
        st.caption(f"E = {young_modulus:,.0f} MPa, ν = {poisson_ratio:.2f}")

    st.subheader("Condiciones de contorno y cargas")
    st.caption("Configure cada borde de la placa de forma independiente.")
    edge_conditions: dict[Edge, EdgeCondition] = {}
    for edge in EDGE_ORDER:
        st.markdown(f"**Borde {edge.value}**")
        support_label = st.selectbox(
            "Apoyo", list(SUPPORT_LABELS.keys()), key=f"support_{edge.value}"
        )
        apply_load = st.checkbox(
            "Aplicar carga distribuida", key=f"load_check_{edge.value}"
        )
        # Nota: dentro de st.form los widgets no disparan un rerun hasta el
        # envío, así que estos campos se muestran siempre (no solo cuando
        # el checkbox está marcado); el checkbox decide, al enviar, si la
        # carga se aplica o se ignora.
        c3, c4 = st.columns([1, 1])
        magnitude = c3.number_input(
            "Esfuerzo (MPa)", value=10.0, key=f"load_mag_{edge.value}"
        )
        direction_label = c4.selectbox(
            "Dirección", list(DIRECTION_LABELS.keys()), key=f"load_dir_{edge.value}"
        )
        load = (
            EdgeLoad(magnitude=magnitude, direction=DIRECTION_LABELS[direction_label])
            if apply_load
            else None
        )
        edge_conditions[edge] = EdgeCondition(
            support=SUPPORT_LABELS[support_label], load=load
        )

    st.subheader("Resistencia (opcional)")
    use_admissible = st.checkbox("Definir esfuerzo admisible")
    admissible_stress_input = st.number_input(
        "Esfuerzo admisible (MPa)", min_value=0.1, value=250.0
    )
    admissible_stress = admissible_stress_input if use_admissible else None

    submitted = st.form_submit_button("Ejecutar análisis", type="primary")

if submitted:
    try:
        geometry_params = PlateGeometryParams(
            length=length,
            height=height,
            hole_radius=hole_radius,
            mesh_size=mesh_size,
            hole_mesh_size=hole_mesh_size or None,
        )
        mesh = build_plate_mesh(geometry_params)

        material = MaterialProps(young_modulus=young_modulus, poisson_ratio=poisson_ratio)
        cons = build_cons_array(mesh.points, length, height, edge_conditions)
        loads = build_loads_array(mesh.points, length, height, edge_conditions)

        result = run_analysis(mesh, material, cons, loads)
        st.session_state["last_result"] = (mesh, result, hole_radius, admissible_stress)

    except (GeometryError, BoundaryConditionError, SolverError) as exc:
        st.error(str(exc))
        st.session_state.pop("last_result", None)

if "last_result" in st.session_state:
    mesh, result, hole_radius, admissible_stress = st.session_state["last_result"]

    tab_stress, tab_mesh, tab_deformed = st.tabs(["Esfuerzos", "Malla", "Deformada"])

    with tab_stress:
        c1, c2, c3 = st.columns(3)
        c1.metric("Esfuerzo máx. von Mises", f"{result.max_von_mises:.2f} MPa")
        max_xy = result.points[result.max_von_mises_node]
        c2.metric("Ubicación del máximo", f"x={max_xy[0]:.1f}, y={max_xy[1]:.1f} mm")
        if admissible_stress:
            utilization = result.max_von_mises / admissible_stress
            c3.metric("Factor de utilización", f"{utilization:.2f}")
            if utilization > 1:
                c3.error("Excede el esfuerzo admisible")
            else:
                c3.success("Dentro del esfuerzo admisible")

        fig = plot_von_mises(mesh, result, hole_radius=hole_radius, admissible_stress=None)
        st.pyplot(fig)

        if admissible_stress:
            st.markdown("**Factor de utilización (σ_vm / σ_admisible)**")
            fig_util = plot_von_mises(
                mesh, result, hole_radius=hole_radius, admissible_stress=admissible_stress
            )
            st.pyplot(fig_util)

        df = summary_dataframe(result)
        with st.expander("Tabla de resultados por nodo"):
            st.dataframe(df, use_container_width=True)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button(
            "Descargar resultados (CSV)",
            csv_buffer.getvalue(),
            file_name="resultados_fem.csv",
            mime="text/csv",
        )
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight")
        col_dl2.download_button(
            "Descargar gráfico de esfuerzos (PNG)",
            png_buffer.getvalue(),
            file_name="esfuerzos_von_mises.png",
            mime="image/png",
        )

    with tab_mesh:
        st.write(f"Nodos: {mesh.points.shape[0]} — Elementos: {mesh.triangles.shape[0]}")
        st.pyplot(plot_mesh(mesh))

    with tab_deformed:
        max_disp = max(abs(result.displacement).max(), 1e-12)
        suggested_scale = 0.1 * min(length, height) / max_disp
        scale_factor = st.slider(
            "Factor de escala de la deformada",
            min_value=0.0,
            max_value=float(suggested_scale * 10),
            value=float(suggested_scale),
        )
        st.pyplot(plot_deformed(mesh, result, scale_factor))
else:
    st.info("Configure los parámetros en el panel izquierdo y presione **Ejecutar análisis**.")
