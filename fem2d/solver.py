"""Integración con el solver de SolidsPy: ensamblado de arrays, resolución
y cálculo de esfuerzos de von Mises.
"""

from dataclasses import dataclass

import numpy as np
from solidspy.solids_GUI import solids_auto

from fem2d.boundary_conditions import check_rigid_body_constraints
from fem2d.exceptions import SolverError
from fem2d.geometry import PlateMesh
from fem2d.materials import MaterialProps

TRIANGLE_ELEMENT_TYPE = 3  # triángulo lineal de 3 nodos (CST) en SolidsPy


@dataclass(frozen=True)
class AnalysisResult:
    points: np.ndarray  # (nnodes, 2)
    triangles: np.ndarray  # (nele, 3)
    displacement: np.ndarray  # (nnodes, 2)
    stress: np.ndarray  # (nnodes, 3) [sigma_xx, sigma_yy, tau_xy]
    von_mises: np.ndarray  # (nnodes,)
    max_von_mises: float
    max_von_mises_node: int


def build_solidspy_data(
    mesh: PlateMesh, material: MaterialProps, cons: np.ndarray, loads: np.ndarray
) -> dict:
    """Arma el dict `data` con el contrato exacto que espera
    `solidspy.solids_GUI.solids_auto`.

    Contrato verificado contra el código fuente de SolidsPy 1.1.0.post1:
    `nodes` son [id, x, y] con id 0-based y consecutivo, coincidente con la
    posición de fila (el ensamblador indexa por posición, no busca por id).
    """
    nnodes = mesh.points.shape[0]
    ids = np.arange(nnodes, dtype=float)
    nodes = np.column_stack([ids, mesh.points])  # (n, 3) [id, x, y]

    n_tri = mesh.triangles.shape[0]
    elements = np.column_stack(
        [
            np.arange(n_tri),
            np.full(n_tri, TRIANGLE_ELEMENT_TYPE),
            np.zeros(n_tri, dtype=int),
            mesh.triangles,
        ]
    ).astype(int)  # (n, 6) [id, tipo=3, idx_material=0, n1, n2, n3]

    mats = np.array([[material.young_modulus, material.poisson_ratio]])

    return {
        "nodes": nodes,
        "cons": cons.astype(int),
        "elements": elements,
        "mats": mats,
        "loads": loads,
    }


def von_mises_stress(sxx: np.ndarray, syy: np.ndarray, txy: np.ndarray) -> np.ndarray:
    """Esfuerzo equivalente de von Mises en tensión plana. SolidsPy no lo
    calcula: solo devuelve [sigma_xx, sigma_yy, tau_xy] por nodo."""
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * txy**2)


def _cst_strain_matrix(coord: np.ndarray) -> tuple[np.ndarray, float]:
    """Matriz B (deformación-desplazamiento) de un triángulo de deformación
    constante (CST), y el área del elemento.

    Fórmula analítica estándar; no depende de `solidspy.postprocesor`.
    """
    x1, y1 = coord[0]
    x2, y2 = coord[1]
    x3, y3 = coord[2]
    area2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    b_mat = np.array(
        [
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3],
        ]
    ) / area2
    return b_mat, abs(area2) / 2.0


def _plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    factor = young / (1.0 - poisson**2)
    return factor * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, (1.0 - poisson) / 2.0],
        ]
    )


def compute_nodal_stress(
    nodes: np.ndarray, elements: np.ndarray, mats: np.ndarray, displacement: np.ndarray
) -> np.ndarray:
    """Esfuerzo promedio en cada nodo, calculado a partir de los
    desplazamientos con la formulación estándar de triángulo de
    deformación constante (CST).

    No usa `solidspy.postprocesor.strain_nodes`: esa función tiene un bug
    confirmado en la versión publicada de SolidsPy (1.1.0.post1) para
    elementos tipo 3 (triángulo lineal). Su `str_el3` reserva espacio para
    3 puntos de evaluación por elemento pero `gauss_tri(order=1)` solo
    entrega 1 punto, así que únicamente el primer nodo local de cada
    elemento recibe el esfuerzo calculado; los otros dos quedan en cero.
    El resultado es un campo de esfuerzos con dos tercios de sus
    contribuciones faltantes. Aquí se recalcula el esfuerzo (constante por
    elemento en un CST) y se promedia entre los elementos que comparten
    cada nodo, replicando la técnica de recuperación nodal que SolidsPy
    pretende implementar, pero sin el bug.
    """
    nnodes = nodes.shape[0]
    stress_sum = np.zeros((nnodes, 3))
    node_count = np.zeros(nnodes)

    for ele in elements:
        _, _, mat_idx, n1, n2, n3 = ele
        local_nodes = (n1, n2, n3)
        coord = nodes[np.array(local_nodes), 1:]
        b_mat, _ = _cst_strain_matrix(coord)

        u_local = np.empty(6)
        u_local[0::2] = displacement[np.array(local_nodes), 0]
        u_local[1::2] = displacement[np.array(local_nodes), 1]

        strain = b_mat @ u_local
        young, poisson = mats[mat_idx]
        d_mat = _plane_stress_matrix(young, poisson)
        stress = d_mat @ strain

        for node in local_nodes:
            stress_sum[node] += stress
            node_count[node] += 1

    return stress_sum / node_count[:, None]


def run_analysis(
    mesh: PlateMesh, material: MaterialProps, cons: np.ndarray, loads: np.ndarray
) -> AnalysisResult:
    """Ejecuta el análisis FEM completo y devuelve desplazamientos y
    esfuerzos (incluido von Mises) en cada nodo."""
    check_rigid_body_constraints(mesh.points, cons)

    data = build_solidspy_data(mesh, material, cons, loads)
    try:
        disp = solids_auto(data, plot_contours=False, compute_strains=False)
    except Exception as exc:
        raise SolverError(f"El solver de SolidsPy falló: {exc}") from exc

    stress = compute_nodal_stress(data["nodes"], data["elements"], data["mats"], disp)

    if not np.all(np.isfinite(disp)) or not np.all(np.isfinite(stress)):
        raise SolverError(
            "La solución contiene valores no finitos (NaN/Inf); es probable que "
            "el sistema esté indeterminado (rigidez casi singular) o mal "
            "condicionado. Revise las condiciones de apoyo y de carga."
        )

    sxx, syy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    von_mises = von_mises_stress(sxx, syy, txy)
    imax = int(np.argmax(von_mises))

    return AnalysisResult(
        points=mesh.points,
        triangles=mesh.triangles,
        displacement=disp,
        stress=stress,
        von_mises=von_mises,
        max_von_mises=float(von_mises[imax]),
        max_von_mises_node=imax,
    )
