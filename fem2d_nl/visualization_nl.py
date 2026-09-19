"""Gráficos matplotlib para el modo no lineal (sección de hormigón
armado): curva momento-curvatura, preview de la malla de sección con las
capas de armadura, y contorno de daño. Mismo patrón que `fem2d/visualization.py`
(backend Agg, sin GUI, seguro para un servidor Streamlit)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def plot_moment_curvature(kappa: np.ndarray, moment: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kappa, moment / 1e6, marker="o", markersize=3, color="tab:blue")
    ax.set_xlabel("Curvatura κ (1/mm)")
    ax.set_ylabel("Momento M (kN·m)")
    ax.set_title("Momento-curvatura")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _quad_polygons(model) -> tuple[np.ndarray, np.ndarray]:
    """Vértices (ne,4,2) y conectividad del primer grupo `quad4` del
    modelo (por construcción de `build_section_mesh`, siempre el
    hormigón)."""
    concrete = model.groups[0]
    conn = concrete.connectivity
    verts = model.nodes[conn]  # (ne,4,2)
    return verts, conn


def plot_section_mesh(model, rebar_layers=None) -> plt.Figure:
    """Preview de la franja de sección: contorno de elementos de
    hormigón y, si se dan, las capas de armadura resaltadas."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    verts, _ = _quad_polygons(model)
    coll = PolyCollection(verts, facecolor="lightgray", edgecolor="tab:blue", linewidth=0.5)
    ax.add_collection(coll)

    for group in model.groups[1:]:
        if group.kind != "truss2":
            continue
        for row in group.connectivity:
            xy = model.nodes[row]
            ax.plot(xy[:, 0], xy[:, 1], color="tab:red", linewidth=2.0, solid_capstyle="round")

    if rebar_layers:
        for layer in rebar_layers:
            ax.annotate(
                f"{layer.n_bars}⌀{layer.diameter:g}",
                xy=(model.nodes[:, 0].max(), layer.depth_y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color="tab:red",
            )

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Franja de sección (elevación)")
    fig.tight_layout()
    return fig


def plot_damage_contour(model, final_states: list, kind: str = "tensile") -> plt.Figure:
    """Contorno de daño del hormigón en el último paso convergido.

    `final_states` es `MomentCurvatureResult.final_states` (estado por
    grupo, alineado con `model.groups` — el grupo 0 es siempre el
    hormigón, por construcción de `build_section_mesh`). El material de
    ese grupo debe exponer `damage_at(kappa_t, kappa_c)` (`ConcreteCDP`).
    `kind`: `"tensile"` o `"compressive"`.
    """
    concrete_group = model.groups[0]
    concrete_state = final_states[0]
    material = concrete_group.material

    d_t, d_c = material.damage_at(concrete_state["kappa_t"], concrete_state["kappa_c"])
    damage_per_gp = d_t if kind == "tensile" else d_c
    n_gauss = concrete_group.n_gauss_per_element
    damage_per_element = damage_per_gp.reshape(concrete_group.n_elements, n_gauss).mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    verts, _ = _quad_polygons(model)
    coll = PolyCollection(verts, array=damage_per_element, cmap="Reds", edgecolor="face")
    coll.set_clim(0.0, 1.0)
    ax.add_collection(coll)
    fig.colorbar(coll, ax=ax, label=f"Daño {'a tracción' if kind == 'tensile' else 'a compresión'} (0-1)")

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Contorno de daño (último paso convergido)")
    fig.tight_layout()
    return fig
