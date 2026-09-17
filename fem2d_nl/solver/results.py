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
