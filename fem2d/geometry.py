"""Generación de geometría y malla para una placa rectangular con agujero.

Usa pygmsh (motor OCC de gmsh) para construir la geometría y triangularla.
"""

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pygmsh

from fem2d.exceptions import GeometryError


@contextmanager
def _gmsh_thread_safe():
    """gmsh.initialize() instala un manejador de SIGINT, lo que falla con
    "signal only works in main thread" si no se llama desde el hilo
    principal (p. ej. el hilo de ejecución de un script de Streamlit).
    Mientras se genera la malla, se neutraliza `signal.signal` si no
    estamos en el hilo principal.
    """
    if threading.current_thread() is threading.main_thread():
        yield
        return
    original_signal = signal.signal
    signal.signal = lambda *args, **kwargs: None
    try:
        yield
    finally:
        signal.signal = original_signal


@dataclass(frozen=True)
class PlateGeometryParams:
    length: float  # L, mm (eje x)
    height: float  # H, mm (eje y)
    hole_radius: float  # R, mm; 0 = sin agujero
    mesh_size: float  # tamaño de elemento objetivo, mm
    hole_mesh_size: float | None = None  # refinamiento en el agujero; default = mesh_size/3


@dataclass(frozen=True)
class PlateMesh:
    points: np.ndarray  # (nnodes, 2)
    triangles: np.ndarray  # (nele, 3), índices 0-based en `points`
    boundary_lines: np.ndarray | None  # (nlines, 2), solo para dibujar contornos


def validate_geometry(params: PlateGeometryParams) -> None:
    if params.length <= 0 or params.height <= 0:
        raise GeometryError("El largo (L) y el alto (H) de la placa deben ser mayores que cero.")
    if params.hole_radius < 0:
        raise GeometryError("El radio del agujero no puede ser negativo.")
    if params.mesh_size <= 0:
        raise GeometryError("El tamaño de malla debe ser mayor que cero.")

    min_side = min(params.length, params.height)

    if params.hole_radius > 0.45 * min_side:
        raise GeometryError(
            "El radio del agujero es demasiado grande respecto a la placa "
            "(debe ser menor al 45% del lado más corto). Reduzca R o aumente L/H."
        )
    if params.mesh_size > min_side / 4:
        raise GeometryError(
            "El tamaño de malla es demasiado grande respecto a la placa "
            "(quedarían muy pocos elementos). Reduzca el tamaño de malla."
        )
    if params.mesh_size < min_side / 500:
        raise GeometryError(
            "El tamaño de malla es demasiado fino respecto a la placa "
            "(la malla resultante sería excesivamente grande). Aumente el tamaño de malla."
        )
    if params.hole_mesh_size is not None:
        if params.hole_mesh_size <= 0:
            raise GeometryError("El tamaño de malla del agujero debe ser mayor que cero.")
        if params.hole_radius > 0 and params.hole_mesh_size > params.hole_radius / 2:
            raise GeometryError(
                "El tamaño de malla del agujero es demasiado grande para discretizar "
                "bien el círculo. Redúzcalo a menos de la mitad del radio del agujero."
            )


def _drop_unused_nodes(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elimina nodos que ningún triángulo referencia.

    Es obligatorio tras filtrar solo las celdas "triangle": las operaciones
    booleanas de OCC pueden dejar vértices o aristas auxiliares en
    mesh.points que, si entran al ensamblador, generan filas/columnas nulas
    en la matriz de rigidez y por lo tanto un sistema singular, incluso con
    condiciones de contorno correctas.
    """
    used = np.unique(triangles)
    remap = -np.ones(points.shape[0], dtype=int)
    remap[used] = np.arange(used.size)
    return points[used], remap[triangles]


def _make_mesh_size_callback(length, height, hole_radius, mesh_size, hole_mesh_size):
    """Tamaño de elemento variable: fino junto al agujero, grueso lejos de él.

    pygmsh descarta el `mesh_size` pasado a add_rectangle/add_disk en cuanto
    se aplica una operación booleana OCC (boolean_difference), así que el
    refinamiento cerca del agujero solo se puede lograr con un callback de
    tamaño de malla (mecanismo recomendado por pygmsh para tamaños
    variables en el espacio).
    """
    cx, cy = length / 2, height / 2
    transition = max(3.0 * hole_radius, mesh_size)

    def callback(dim, tag, x, y, z, lc):
        if hole_radius <= 0:
            return mesh_size
        dist_to_hole = max(np.hypot(x - cx, y - cy) - hole_radius, 0.0)
        t = min(dist_to_hole / transition, 1.0)
        return hole_mesh_size + t * (mesh_size - hole_mesh_size)

    return callback


def build_plate_mesh(params: PlateGeometryParams) -> PlateMesh:
    validate_geometry(params)

    hole_mesh_size = params.hole_mesh_size or params.mesh_size / 3

    with _gmsh_thread_safe(), pygmsh.occ.Geometry() as geom:
        rect = geom.add_rectangle([0.0, 0.0, 0.0], params.length, params.height)
        if params.hole_radius > 0:
            hole = geom.add_disk(
                [params.length / 2, params.height / 2, 0.0], params.hole_radius
            )
            geom.boolean_difference(rect, hole)

        geom.characteristic_length_min = (
            hole_mesh_size if params.hole_radius > 0 else params.mesh_size
        )
        geom.characteristic_length_max = params.mesh_size
        geom.set_mesh_size_callback(
            _make_mesh_size_callback(
                params.length, params.height, params.hole_radius, params.mesh_size, hole_mesh_size
            )
        )
        try:
            raw_mesh = geom.generate_mesh(dim=2)
        except Exception as exc:  # errores de gmsh/OCC en geometrías degeneradas
            raise GeometryError(f"No se pudo generar la malla: {exc}") from exc

    if "triangle" not in raw_mesh.cells_dict or raw_mesh.cells_dict["triangle"].shape[0] == 0:
        raise GeometryError(
            "La malla generada no contiene elementos triangulares. "
            "Revise los parámetros de geometría (L, H, R, tamaño de malla)."
        )

    points = raw_mesh.points[:, :2]
    triangles = raw_mesh.cells_dict["triangle"]
    points, triangles = _drop_unused_nodes(points, triangles)
    boundary_lines = raw_mesh.cells_dict.get("line")

    return PlateMesh(points=points, triangles=triangles, boundary_lines=boundary_lines)
