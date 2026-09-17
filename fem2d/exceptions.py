"""Excepciones de dominio de fem2d.

Se usan para mostrar mensajes accionables en la interfaz en vez de
tracebacks crudos.
"""


class Fem2DError(Exception):
    """Base para todas las excepciones de dominio de fem2d."""


class GeometryError(Fem2DError):
    """Parámetros de geometría o malla inválidos."""


class BoundaryConditionError(Fem2DError):
    """Condiciones de contorno o cargas inválidas (p. ej. cuerpo rígido)."""


class SolverError(Fem2DError):
    """El solver de SolidsPy falló o devolvió una solución no válida."""
