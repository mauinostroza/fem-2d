"""Material elástico lineal en tensión plana, y el contrato que deben
seguir todos los materiales no lineales de fem2d_nl (acero multilineal,
hormigón CDP, adherencia).

Contrato de material (struct-of-arrays, por lotes de puntos de
integración — nunca un objeto Python por punto, sería demasiado lento):

    state = material.initial_state(n_points)   # dict[str, np.ndarray], primer eje = n_points
    stress, tangent, new_state = material.integrate(strain, state, dt)

    strain:  (n_points, 3) float  [eps_xx, eps_yy, gamma_xy]  (gamma = 2*eps_xy)
    stress:  (n_points, 3) float  [sigma_xx, sigma_yy, tau_xy]
    tangent: (n_points, 3, 3) float  d(stress)/d(strain), evaluado en el propio `strain`
    new_state: dict[str, np.ndarray] con la misma forma que `state`

`integrate` debe ser puro respecto de `state` (no mutarlo in-place): el
solver decide cuándo el estado de un punto se vuelve permanente (solo al
convergir un paso de carga), así que en cada iteración de Newton se le
puede pasar el mismo `state` de partida con distintos `strain` de prueba.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticPlaneStress:
    young_modulus: float
    poisson_ratio: float

    def _d_matrix(self) -> np.ndarray:
        e, nu = self.young_modulus, self.poisson_ratio
        factor = e / (1.0 - nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ]
        )

    def initial_state(self, n_points: int) -> dict:
        return {}

    def integrate(self, strain: np.ndarray, state: dict, dt: float):
        d_mat = self._d_matrix()
        stress = strain @ d_mat.T
        tangent = np.broadcast_to(d_mat, (strain.shape[0], 3, 3)).copy()
        return stress, tangent, state


@dataclass(frozen=True)
class ElasticUniaxial:
    """Material 1D elástico lineal, para `elements.truss2` (mismo contrato
    escalar que `steel_uniaxial.MultilinearSteel`: útil como placeholder
    antes de que el acero entre en fluencia, o en pruebas donde solo se
    quiere aislar la no linealidad de otro componente, p. ej. adherencia)."""

    young_modulus: float

    def initial_state(self, n_points: int) -> dict:
        return {}

    def integrate(self, strain: np.ndarray, state: dict, dt: float):
        stress = self.young_modulus * strain
        tangent = np.full_like(strain, self.young_modulus)
        return stress, tangent, state
