"""Acero de refuerzo: plasticidad 1D con endurecimiento isótropo
multilineal, definido por una tabla (eps, sigma) que da el usuario.

Contrato de material 1D (usado por `elements.truss2` y, con la misma
forma, por la ley de adherencia): `integrate(eps(n,), state, dt) ->
(sigma(n,), tangent(n,), new_state)` — escalares, no tensores.

Por qué endurecimiento isótropo y no cinemático: bajo carga monótona o
proporcional (el caso de uso de esta app) ambos dan la misma curva, y el
retorno (return map) isótropo con endurecimiento multilineal tiene
solución cerrada (recorriendo segmentos de la tabla), sin iteración y
sin posibilidad de fallo de convergencia. Un modelo cinemático
multilineal exigiría una familia de superficies tipo Iwan, mucho más
costoso, y solo se justificaría con inversión de carga cíclica.
"""

from dataclasses import dataclass, field

import numpy as np

from fem2d_nl.exceptions import MaterialModelError


@dataclass(frozen=True)
class MultilinearSteel:
    points: np.ndarray  # (n,2) [eps, sigma], eps > 0 estrictamente creciente
    young_modulus: float | None = None  # None => sigma[0]/eps[0]

    kappa_table: np.ndarray = field(init=False, repr=False)
    sigma_table: np.ndarray = field(init=False, repr=False)
    slopes: np.ndarray = field(init=False, repr=False)
    e_resolved: float = field(init=False, repr=False)

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise MaterialModelError(
                "La curva del acero debe tener al menos 2 puntos (eps, sigma)."
            )
        eps, sigma = points[:, 0], points[:, 1]
        if np.any(np.diff(eps) <= 0):
            raise MaterialModelError(
                "Los puntos (eps, sigma) del acero deben tener deformación estrictamente creciente."
            )
        if eps[0] <= 0 or sigma[0] <= 0:
            raise MaterialModelError(
                "El primer punto de la curva del acero debe tener eps>0 y sigma>0 (punto de fluencia inicial)."
            )
        e_resolved = self.young_modulus if self.young_modulus is not None else sigma[0] / eps[0]
        if e_resolved <= 0:
            raise MaterialModelError("El módulo de Young del acero debe ser positivo.")

        kappa = np.maximum(eps - sigma / e_resolved, 0.0)
        kappa[0] = 0.0  # por construcción, el primer punto es la fluencia inicial (kappa=0)
        if np.any(np.diff(kappa) <= 0):
            raise MaterialModelError(
                "La curva del acero implica endurecimiento negativo excesivo "
                "(la deformación plástica no es creciente); revise E y los puntos dados."
            )
        slopes = np.diff(sigma) / np.diff(kappa)
        # Nota: E + H > 0 está garantizado automáticamente por construcción
        # (kappa = eps - sigma/E acota H por encima de -E sin importar cuán
        # pronunciado sea el ablandamiento en sigma), así que no hace falta
        # validarlo aparte: probar lo contrario es matemáticamente imposible
        # mientras `np.diff(eps) > 0`, ya validado arriba.

        # kappa_table/sigma_table son la tabla de fluencia sigma_y(kappa):
        # en kappa=0, sigma_y=sigma[0]=fy (el punto de fluencia inicial dado
        # por el usuario). `slopes[i]` es la pendiente entre kappa_table[i] y
        # kappa_table[i+1] (longitud n-1, un valor menos que la tabla).
        object.__setattr__(self, "kappa_table", kappa)
        object.__setattr__(self, "sigma_table", sigma)
        object.__setattr__(self, "slopes", slopes)
        object.__setattr__(self, "e_resolved", e_resolved)

    def _segment_index(self, kappa: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.kappa_table, kappa, side="right") - 1
        return np.clip(idx, 0, len(self.kappa_table) - 2)

    def _upper_bound(self, idx: np.ndarray) -> np.ndarray:
        upper = self.kappa_table[idx + 1].copy()
        upper[idx == len(self.kappa_table) - 2] = np.inf  # última rama: extrapola
        return upper

    def _sigma_y(self, kappa: np.ndarray) -> np.ndarray:
        idx = self._segment_index(kappa)
        return self.sigma_table[idx] + self.slopes[idx] * (kappa - self.kappa_table[idx])

    def initial_state(self, n_points: int) -> dict:
        return {"eps_pl": np.zeros(n_points), "kappa": np.zeros(n_points)}

    def integrate(self, eps: np.ndarray, state: dict, dt: float):
        eps_pl = state["eps_pl"]
        kappa0 = state["kappa"]
        n = eps.shape[0]

        sigma_tr = self.e_resolved * (eps - eps_pl)
        sign = np.where(sigma_tr >= 0.0, 1.0, -1.0)
        g = np.abs(sigma_tr) - self._sigma_y(kappa0)  # residuo a lam=0
        active = g > 1e-10 * max(self.e_resolved, 1.0)

        delta = np.zeros(n)
        kappa_work = kappa0.copy()
        has_plastic = active.copy()

        for _ in range(len(self.kappa_table)):
            if not active.any():
                break
            idx = self._segment_index(kappa_work)
            upper = self._upper_bound(idx)
            denom = self.e_resolved + self.slopes[idx]
            delta_local = g / denom

            within = active & (kappa_work + delta_local <= upper)
            crosses = active & ~within

            delta[within] += delta_local[within]
            kappa_work[within] += delta_local[within]
            active[within] = False

            seg_len = np.zeros(n)
            seg_len[crosses] = upper[crosses] - kappa_work[crosses]
            delta[crosses] += seg_len[crosses]
            g[crosses] -= denom[crosses] * seg_len[crosses]
            kappa_work[crosses] += seg_len[crosses]

        kappa_new = kappa0 + delta
        eps_pl_new = eps_pl + sign * delta
        sigma = sigma_tr - sign * self.e_resolved * delta

        idx_final = self._segment_index(kappa_new)
        h_final = self.slopes[idx_final]
        tangent = np.where(
            has_plastic, self.e_resolved * h_final / (self.e_resolved + h_final), self.e_resolved
        )

        new_state = {"eps_pl": eps_pl_new, "kappa": kappa_new}
        return sigma, tangent, new_state
