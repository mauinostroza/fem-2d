"""Condiciones de contorno y cargas definidas por borde de la placa.

La placa es un rectángulo [0, L] x [0, H]. El usuario configura, para cada
uno de los 4 bordes, un tipo de apoyo y, opcionalmente, una carga
distribuida (en MPa, equivalente a N/mm de borde con espesor unitario).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from fem2d.exceptions import BoundaryConditionError


class Edge(str, Enum):
    LEFT = "izquierdo"
    RIGHT = "derecho"
    TOP = "superior"
    BOTTOM = "inferior"


class Support(str, Enum):
    FREE = "libre"
    FIXED = "empotrado"
    ROLLER_X = "restringido_x"  # bc_x=-1, bc_y=0: impide desplazamiento horizontal
    ROLLER_Y = "restringido_y"  # bc_x=0, bc_y=-1: impide desplazamiento vertical


_EDGE_NORMALS = {
    Edge.LEFT: (-1.0, 0.0),
    Edge.RIGHT: (1.0, 0.0),
    Edge.TOP: (0.0, 1.0),
    Edge.BOTTOM: (0.0, -1.0),
}

_DIRECTION_VECTORS = {
    "x": (1.0, 0.0),
    "y": (0.0, 1.0),
}


@dataclass(frozen=True)
class EdgeLoad:
    magnitude: float  # MPa; positivo según la convención de `direction`
    direction: Literal["x", "y", "normal"] = "normal"


@dataclass(frozen=True)
class EdgeCondition:
    support: Support = Support.FREE
    load: EdgeLoad | None = None


EdgeConditions = dict[Edge, EdgeCondition]


def default_edge_tolerance(length: float, height: float) -> float:
    """Tolerancia razonable para identificar nodos de borde por coordenada."""
    return 1e-6 * max(length, height)


def find_edge_nodes(
    points: np.ndarray, edge: Edge, length: float, height: float, tol: float
) -> np.ndarray:
    """Devuelve los índices de los nodos que caen sobre el borde dado."""
    if edge == Edge.LEFT:
        mask = np.abs(points[:, 0] - 0.0) <= tol
    elif edge == Edge.RIGHT:
        mask = np.abs(points[:, 0] - length) <= tol
    elif edge == Edge.BOTTOM:
        mask = np.abs(points[:, 1] - 0.0) <= tol
    elif edge == Edge.TOP:
        mask = np.abs(points[:, 1] - height) <= tol
    else:  # pragma: no cover - Edge es un Enum cerrado
        raise ValueError(f"Borde desconocido: {edge}")
    return np.nonzero(mask)[0]


def tributary_lengths(sorted_coord: np.ndarray) -> np.ndarray:
    """Longitud tributaria de cada nodo a lo largo de un borde (regla del
    punto medio). La suma da exactamente la longitud total del borde, por
    lo que la resultante de la carga repartida es físicamente correcta.
    """
    n = sorted_coord.shape[0]
    if n == 1:
        return np.zeros(1)
    trib = np.empty(n)
    trib[0] = (sorted_coord[1] - sorted_coord[0]) / 2
    trib[-1] = (sorted_coord[-1] - sorted_coord[-2]) / 2
    if n > 2:
        trib[1:-1] = (sorted_coord[2:] - sorted_coord[:-2]) / 2
    return trib


def _edge_running_coord(points: np.ndarray, edge: Edge) -> np.ndarray:
    """Coordenada que varía a lo largo del borde (para ordenar y calcular
    longitudes tributarias)."""
    return points[:, 1] if edge in (Edge.LEFT, Edge.RIGHT) else points[:, 0]


def build_cons_array(
    points: np.ndarray,
    length: float,
    height: float,
    conditions: EdgeConditions,
    tol: float | None = None,
) -> np.ndarray:
    """Construye el array `cons` (nnodes, 2) esperado por SolidsPy.

    Convención: -1 = grado de libertad restringido, 0 = libre. Si dos
    bordes comparten un nodo (una esquina) con distintas restricciones,
    prevalece la más restrictiva: nunca se libera un grado de libertad ya
    fijado por otro borde.
    """
    tol = tol if tol is not None else default_edge_tolerance(length, height)
    cons = np.zeros((points.shape[0], 2), dtype=int)

    for edge, condition in conditions.items():
        if condition.support == Support.FREE:
            continue
        idx = find_edge_nodes(points, edge, length, height, tol)
        if condition.support == Support.FIXED:
            cons[idx, 0] = -1
            cons[idx, 1] = -1
        elif condition.support == Support.ROLLER_X:
            cons[idx, 0] = -1
        elif condition.support == Support.ROLLER_Y:
            cons[idx, 1] = -1

    return cons


def build_loads_array(
    points: np.ndarray,
    length: float,
    height: float,
    conditions: EdgeConditions,
    tol: float | None = None,
) -> np.ndarray:
    """Construye el array `loads` (nloads, 3) = [id_nodo, fx, fy] esperado
    por SolidsPy, repartiendo cada carga de borde entre sus nodos por
    longitud tributaria.
    """
    tol = tol if tol is not None else default_edge_tolerance(length, height)
    nodal_forces: dict[int, list[float]] = {}

    for edge, condition in conditions.items():
        if condition.load is None or condition.load.magnitude == 0.0:
            continue
        idx = find_edge_nodes(points, edge, length, height, tol)
        if idx.size == 0:
            continue

        if condition.load.direction == "normal":
            dir_x, dir_y = _EDGE_NORMALS[edge]
        else:
            dir_x, dir_y = _DIRECTION_VECTORS[condition.load.direction]

        running = _edge_running_coord(points, edge)[idx]
        order = np.argsort(running)
        idx_sorted = idx[order]
        trib = tributary_lengths(running[order])

        for node_id, t in zip(idx_sorted, trib):
            force = condition.load.magnitude * t
            fx, fy = force * dir_x, force * dir_y
            if node_id in nodal_forces:
                nodal_forces[node_id][0] += fx
                nodal_forces[node_id][1] += fy
            else:
                nodal_forces[node_id] = [fx, fy]

    if not nodal_forces:
        return np.zeros((0, 3))

    loads = np.array(
        [[node_id, fx, fy] for node_id, (fx, fy) in sorted(nodal_forces.items())]
    )
    return loads


def check_rigid_body_constraints(points: np.ndarray, cons: np.ndarray) -> None:
    """Verifica que las restricciones impidan los 3 modos de cuerpo rígido
    en 2D (traslación en x, traslación en y, rotación).

    Es un criterio general basado en el rango de la matriz de modos de
    cuerpo rígido restringida a los grados de libertad fijados por el
    usuario, en vez de reglas ad hoc por borde: detecta casos sutiles,
    como un borde vertical "restringido en y" (que no impide la rotación
    de la placa alrededor de ese borde).
    """
    restrained = np.argwhere(cons == -1)  # (k, 2): [idx_nodo, dof]
    if restrained.shape[0] < 3:
        raise BoundaryConditionError(
            "Se requieren al menos 3 grados de libertad restringidos para evitar "
            "el movimiento de cuerpo rígido de la placa. Configure apoyos en al "
            "menos un borde."
        )

    modes = np.empty((restrained.shape[0], 3))
    for k, (node_idx, dof) in enumerate(restrained):
        x, y = points[node_idx]
        modes[k] = (1.0, 0.0, -y) if dof == 0 else (0.0, 1.0, x)

    if np.linalg.matrix_rank(modes) < 3:
        raise BoundaryConditionError(
            "Las condiciones de apoyo actuales no restringen completamente el "
            "movimiento de cuerpo rígido de la placa (falta impedir una "
            "traslación o la rotación). Ejemplos válidos: un borde 'empotrado'; "
            "o un borde vertical 'restringido en x' junto con un borde "
            "horizontal 'restringido en y'."
        )
