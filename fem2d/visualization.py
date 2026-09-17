"""Gráficos matplotlib: malla, deformada y contornos de esfuerzo."""

import matplotlib

matplotlib.use("Agg")  # backend sin GUI: seguro para un servidor Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.tri import Triangulation

from fem2d.geometry import PlateMesh
from fem2d.solver import AnalysisResult


def _hole_outline(mesh: PlateMesh, hole_radius: float) -> np.ndarray | None:
    """Puntos del contorno del agujero, ordenados angularmente, para
    dibujarlo como una línea limpia (a partir de las celdas "line" de la
    malla, si existen)."""
    if hole_radius <= 0 or mesh.boundary_lines is None:
        return None
    # el agujero está centrado en la placa: el centro exacto es el centro
    # de la caja envolvente de la malla (el agujero no toca los bordes)
    center = (mesh.points.min(axis=0) + mesh.points.max(axis=0)) / 2
    dist = np.linalg.norm(mesh.points - center, axis=1)
    on_hole = np.abs(dist - hole_radius) < 1e-3 * max(hole_radius, 1.0)
    if not np.any(on_hole):
        return None
    pts = mesh.points[on_hole]
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return pts[order]


def plot_mesh(mesh: PlateMesh) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    triang = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    ax.triplot(triang, color="tab:blue", linewidth=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Malla ({mesh.points.shape[0]} nodos, {mesh.triangles.shape[0]} elementos)")
    fig.tight_layout()
    return fig


def plot_deformed(
    mesh: PlateMesh, result: AnalysisResult, scale_factor: float
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    triang_original = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    ax.triplot(triang_original, color="lightgray", linewidth=0.4, label="Original")

    deformed_points = mesh.points + scale_factor * result.displacement
    triang_deformed = Triangulation(
        deformed_points[:, 0], deformed_points[:, 1], mesh.triangles
    )
    ax.triplot(triang_deformed, color="tab:red", linewidth=0.5, label="Deformada")

    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Deformada (factor de escala x{scale_factor:g})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_von_mises(
    mesh: PlateMesh,
    result: AnalysisResult,
    hole_radius: float = 0.0,
    admissible_stress: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    triang = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)

    field = result.von_mises
    label = "Von Mises (MPa)"
    title = "Esfuerzo de von Mises"
    if admissible_stress:
        field = result.von_mises / admissible_stress
        label = "Factor de utilización (σ_vm / σ_admisible)"
        title = "Factor de utilización"

    contour = ax.tripcolor(triang, field, shading="gouraud", cmap="jet")
    fig.colorbar(contour, ax=ax, label=label)

    outline = _hole_outline(mesh, hole_radius)
    if outline is not None:
        outline_closed = np.vstack([outline, outline[0]])
        ax.plot(outline_closed[:, 0], outline_closed[:, 1], color="black", linewidth=1.2)
    elif hole_radius > 0:
        center = (mesh.points.min(axis=0) + mesh.points.max(axis=0)) / 2
        ax.add_patch(Circle(center, hole_radius, fill=False, edgecolor="black", linewidth=1.2))

    max_point = result.points[result.max_von_mises_node]
    ax.plot(*max_point, marker="x", color="white", markeredgewidth=2, markersize=10)
    ax.annotate(
        f"máx: {result.max_von_mises:.1f} MPa",
        xy=max_point,
        xytext=(10, 10),
        textcoords="offset points",
        color="black",
        backgroundcolor="white",
        fontsize=9,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def summary_dataframe(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(result.points.shape[0]),
            "x": result.points[:, 0],
            "y": result.points[:, 1],
            "ux": result.displacement[:, 0],
            "uy": result.displacement[:, 1],
            "sigma_xx": result.stress[:, 0],
            "sigma_yy": result.stress[:, 1],
            "tau_xy": result.stress[:, 2],
            "von_mises": result.von_mises,
        }
    )
