"""Excepciones de dominio de fem2d_nl.

Extienden la jerarquía de `fem2d.exceptions` para que la UI las maneje
de forma uniforme entre el modo lineal y el modo no lineal.
"""

from fem2d.exceptions import BoundaryConditionError, Fem2DError, GeometryError

__all__ = [
    "Fem2DError",
    "GeometryError",
    "BoundaryConditionError",
    "MaterialModelError",
    "ConvergenceError",
]


class MaterialModelError(Fem2DError):
    """Parámetros de material inválidos o inconsistentes (p. ej. malla
    demasiado gruesa para la energía de fractura dada: snap-back local)."""


class ConvergenceError(Fem2DError):
    """El solver no lineal no pudo converger en ningún paso útil."""
