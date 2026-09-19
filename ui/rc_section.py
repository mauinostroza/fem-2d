"""Modo no lineal: sección de hormigón armado (momento-curvatura vía CDP +
malla estructurada + solver Newton-Raphson propio, `fem2d_nl`).

Ver los avisos de honestidad técnica en el README antes de usar resultados
de este modo para producción: pico de tracción uniaxial ~30-40% sobre ft,
valores de tabla de adherencia no verificados contra el MC2010 original,
`eps_c1`/ablandamiento en compresión aproximados, `y_na` fijo en
`moment_curvature` (N no es exactamente cero tras la fisuración), y la
comparación cuantitativa perfecto-vs-bond_slip descartada en S5 por una
sensibilidad no diagnosticada al número de pasos de carga.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from fem2d_nl.exceptions import Fem2DError
from fem2d_nl.materials.bond_mc2010 import BondLaw, BondParameters
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.materials.concrete_curves import mc2010_eci, mc2010_fcm, mc2010_fctm, mc2010_fracture_energy
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel
from fem2d_nl.mesh.section import RebarLayer, SectionGeometry
from fem2d_nl.mesh.structured import build_section_mesh, nominal_char_length
from fem2d_nl.postprocess import moment_curvature, transformed_elastic_centroid
from fem2d_nl.solver.newton import SolverOptions
from fem2d_nl.visualization_nl import plot_damage_contour, plot_moment_curvature, plot_section_mesh

SESSION_KEY = "rc:last_result"

_DEFAULT_LAYERS = pd.DataFrame(
    [{"depth_y (mm)": 40.0, "n_barras": 3, "diámetro (mm)": 16.0}]
)


def render() -> None:
    st.title("Momento-curvatura de sección de hormigón armado")
    st.caption(
        "Modo experimental (no lineal): hormigón con daño-plasticidad (CDP), "
        "acero multilineal, adherencia opcional. Ver los avisos de honestidad "
        "técnica en el README antes de usar en producción."
    )

    with st.sidebar.form("parametros_rc"):
        st.subheader("Geometría de la franja")
        col1, col2 = st.columns(2)
        width = col1.number_input("Ancho b (mm)", min_value=1.0, value=200.0, step=10.0)
        height = col2.number_input("Altura h (mm)", min_value=1.0, value=400.0, step=10.0)
        span = st.number_input(
            "Longitud de la franja analizada (mm)", min_value=1.0, value=200.0, step=10.0,
            help="Tramo corto usado solo para imponer flexión (ver README); no es la luz de la viga.",
        )
        col3, col4 = st.columns(2)
        nx = col3.number_input("Elementos en x", min_value=1, value=3, step=1)
        ny = col4.number_input("Elementos en y", min_value=1, value=6, step=1)
        st.caption(
            "Malla más fina = más preciso pero más lento (~1-2 s por paso de carga "
            "para una malla chica; puede tardar minutos con mallas grandes)."
        )

        st.subheader("Hormigón")
        fck = st.number_input("f'ck (MPa)", min_value=1.0, value=25.0, step=1.0)
        fcm, fctm = mc2010_fcm(fck), mc2010_fctm(fck)
        st.caption(
            f"fcm={fcm:.1f} MPa, fctm={fctm:.2f} MPa, Eci={mc2010_eci(fcm):,.0f} MPa, "
            f"GF={mc2010_fracture_energy(fcm):.4f} N/mm (fib Model Code 2010)."
        )
        with st.expander("Avanzado: overrides de hormigón (valores no verificados por defecto)"):
            override_ft = st.checkbox("Sobreescribir ft")
            ft_input = st.number_input("ft (MPa)", min_value=0.01, value=float(fctm))
            override_gf = st.checkbox("Sobreescribir GF")
            gf_input = st.number_input("GF (N/mm)", min_value=1e-4, value=float(mc2010_fracture_energy(fcm)), format="%.4f")
            override_eps_c1 = st.checkbox("Sobreescribir eps_c1")
            eps_c1_input = st.number_input("eps_c1", min_value=1e-4, value=0.0022, format="%.4f")

        st.subheader("Acero de refuerzo")
        col5, col6 = st.columns(2)
        fy = col5.number_input("fy (MPa)", min_value=1.0, value=420.0, step=10.0)
        fu = col6.number_input("fu (MPa)", min_value=1.0, value=500.0, step=10.0)
        col7, col8 = st.columns(2)
        eps_u = col7.number_input("Deformación última εu", min_value=0.001, value=0.08, format="%.3f")
        es_steel = col8.number_input("Módulo de Young Es (MPa)", min_value=1.0, value=200_000.0)

        st.subheader("Armadura (capas)")
        st.caption("Cada fila colapsa n barras del mismo diámetro en una línea de elementos.")
        layers_df = st.data_editor(_DEFAULT_LAYERS, num_rows="dynamic", width="stretch")

        st.subheader("Adherencia")
        bond_mode_label = st.radio("Modo", ["Adherencia perfecta", "Adherencia (fib MC2010)"])
        bond_condition = st.selectbox("Condición de adherencia (si aplica)", ["good", "other"])

        st.subheader("Análisis")
        eps_cu = st.number_input(
            "Deformación máxima objetivo en la fibra extrema (εcu)",
            min_value=1e-5, value=0.0035, format="%.4f",
            help="MC2010 usa 0.0035 como deformación última de compresión.",
        )
        n_steps = st.number_input("Pasos de carga", min_value=2, value=15, step=1)
        with st.expander("Avanzado: solver no lineal"):
            max_iterations = st.number_input("Iteraciones máx. por paso", min_value=5, value=30, step=5)
            max_cutbacks = st.number_input("Recortes de paso máx.", min_value=1, value=6, step=1)

        submitted = st.form_submit_button("Ejecutar análisis", type="primary")

    if submitted:
        try:
            geom = SectionGeometry(width=width, height=height, span=span, nx=int(nx), ny=int(ny))

            steel = MultilinearSteel(
                points=np.array([[fy / es_steel, fy], [eps_u, fu]]), young_modulus=es_steel
            )
            layers = [
                RebarLayer(
                    depth_y=float(row["depth_y (mm)"]),
                    n_bars=int(row["n_barras"]),
                    diameter=float(row["diámetro (mm)"]),
                    steel=steel,
                )
                for _, row in layers_df.dropna().iterrows()
            ]

            cdp_params = CDPParameters(
                young_modulus=mc2010_eci(fcm),
                poisson_ratio=0.2,
                fck=fck,
                ft=ft_input if override_ft else None,
                gf=gf_input if override_gf else None,
                eps_c1=eps_c1_input if override_eps_c1 else None,
            )
            l_ch = nominal_char_length(geom)
            cdp = ConcreteCDP(cdp_params, char_length=l_ch)

            bond_mode = "perfect" if bond_mode_label.startswith("Adherencia perfecta") else "bond_slip"
            bond_law = (
                BondLaw(BondParameters(fcm=cdp.fcm, bar_diameter=layers[0].diameter, bond_condition=bond_condition))
                if bond_mode == "bond_slip" and layers
                else None
            )

            model = build_section_mesh(geom, layers, cdp, bond_mode=bond_mode, bond_params=bond_law)
            y_na = transformed_elastic_centroid(geom, layers, cdp.e0) if layers else geom.height / 2.0
            kappa_max = eps_cu / max(geom.height - y_na, 1e-6)

            progress_bar = st.sidebar.progress(0.0, text="Ejecutando análisis…")

            def _progress(step_idx, iters, res_norm, lam):
                progress_bar.progress(min(lam, 1.0), text=f"Paso {step_idx + 1} — λ={lam:.2f}")

            options = SolverOptions(
                n_steps=int(n_steps), max_iterations=int(max_iterations), max_cutbacks=int(max_cutbacks)
            )
            result = moment_curvature(
                model, geom.span, y_na, kappa_max, n_steps=int(n_steps),
                options=options, progress_cb=_progress,
            )
            progress_bar.empty()

            if not result.converged:
                st.warning(
                    f"El análisis no convergió del todo: {result.message} "
                    "Se muestran los pasos que sí convergieron."
                )

            st.session_state[SESSION_KEY] = (model, layers, result)

        except Fem2DError as exc:
            st.error(str(exc))
            st.session_state.pop(SESSION_KEY, None)

    if SESSION_KEY in st.session_state:
        model, layers, result = st.session_state[SESSION_KEY]

        if len(result.kappa) == 0:
            st.error("Ningún paso convergió; no hay resultados para mostrar.")
            return

        tab_mc, tab_mesh, tab_damage = st.tabs(["Momento-curvatura", "Malla", "Daño"])

        with tab_mc:
            c1, c2, c3 = st.columns(3)
            c1.metric("Momento máximo", f"{result.moment.max() / 1e6:.2f} kN·m")
            c2.metric("Curvatura final", f"{result.kappa[-1]:.3e} 1/mm")
            n_final = result.axial[-1]
            c3.metric("Fuerza axial final (N)", f"{n_final / 1e3:.2f} kN")
            if abs(n_final) > 0.1 * abs(result.moment[-1] / max(model.nodes[:, 1].max(), 1.0)):
                st.caption(
                    "⚠ La fuerza axial no es despreciable frente a M/altura: recordar que "
                    "`y_na` queda fijo (ver aviso de honestidad en el README) — no es "
                    "flexión pura exacta en curvaturas grandes."
                )

            st.pyplot(plot_moment_curvature(result.kappa, result.moment))

            df = pd.DataFrame({"kappa": result.kappa, "M": result.moment, "N": result.axial})
            with st.expander("Tabla de la curva momento-curvatura"):
                st.dataframe(df, width="stretch")
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Descargar curva M-κ (CSV)", csv_buffer.getvalue(),
                file_name="momento_curvatura.csv", mime="text/csv",
            )

        with tab_mesh:
            st.write(f"Nodos: {model.n_nodes} — Elementos de hormigón: {model.groups[0].n_elements}")
            st.pyplot(plot_section_mesh(model, layers))

        with tab_damage:
            if result.final_states is not None:
                kind_label = st.radio("Tipo de daño", ["A tracción", "A compresión"], horizontal=True)
                kind = "tensile" if kind_label == "A tracción" else "compressive"
                st.pyplot(plot_damage_contour(model, result.final_states, kind=kind))
            else:
                st.info("No hay estado disponible (ningún paso convergió).")
    else:
        st.info("Configure los parámetros en el panel izquierdo y presione **Ejecutar análisis**.")
