"""Esquemas de control del paso de carga.

Ambos exponen la misma interfaz mínima que usa `solver.newton`:

    control.apply(u, lam) -> u          # impone fijos/prescritos en u para el factor lam
    control.free_dofs(ndof) -> np.ndarray
    control.external_force(ndof, lam) -> np.ndarray   # fuerza externa nodal en ese lam

El control de desplazamiento es el default del proyecto (ver plan): con
ablandamiento por daño, el control de carga diverge en el pico. Arc-length
queda para una sesión futura (marcado como no implementado todavía).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DisplacementControl:
    fixed_dofs: np.ndarray                  # apoyos, u=0 siempre
    prescribed_dofs: np.ndarray             # dofs con desplazamiento impuesto = lam * valor
    prescribed_values: np.ndarray           # valor a lam=1, mismo shape que prescribed_dofs

    def apply(self, u: np.ndarray, lam: float) -> np.ndarray:
        u = u.copy()
        u[self.fixed_dofs] = 0.0
        u[self.prescribed_dofs] = lam * self.prescribed_values
        return u

    def free_dofs(self, ndof: int) -> np.ndarray:
        mask = np.ones(ndof, dtype=bool)
        mask[self.fixed_dofs] = False
        mask[self.prescribed_dofs] = False
        return np.nonzero(mask)[0]

    def external_force(self, ndof: int, lam: float) -> np.ndarray:
        return np.zeros(ndof)


@dataclass(frozen=True)
class LoadControl:
    fixed_dofs: np.ndarray
    force_pattern: np.ndarray               # (ndof,) fuerza externa a lam=1

    def apply(self, u: np.ndarray, lam: float) -> np.ndarray:
        u = u.copy()
        u[self.fixed_dofs] = 0.0
        return u

    def free_dofs(self, ndof: int) -> np.ndarray:
        mask = np.ones(ndof, dtype=bool)
        mask[self.fixed_dofs] = False
        return np.nonzero(mask)[0]

    def external_force(self, ndof: int, lam: float) -> np.ndarray:
        return lam * self.force_pattern
