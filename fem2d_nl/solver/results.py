"""Estructuras de resultado del solver no lineal."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class StepResult:
    lam: float                      # factor de carga/desplazamiento del paso
    u: np.ndarray                   # (ndof,) desplazamientos al final del paso
    states: list                    # estado por grupo, ya hecho permanente
    iterations: int
    residual_norm: float
    f_int: np.ndarray               # (ndof,) fuerzas internas en el u convergido
    """Ya calculado al converger el paso: evita tener que volver a llamar
    `assemble()` fuera del solver para leer reacciones/fuerzas (una nueva
    llamada, con un estado que ya no es "de prueba", puede en principio
    disparar una perturbación de diferencias finitas distinta y fallar en
    un material con return mapping propio, p. ej. `concrete_cdp`)."""


@dataclass
class NonlinearResult:
    steps: list[StepResult] = field(default_factory=list)
    converged: bool = True          # False si algún paso no convergió y se abortó
    message: str = ""

    @property
    def load_factors(self) -> np.ndarray:
        return np.array([s.lam for s in self.steps])

    def displacement_history(self, dof: int) -> np.ndarray:
        return np.array([s.u[dof] for s in self.steps])
