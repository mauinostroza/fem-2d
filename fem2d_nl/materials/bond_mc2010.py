"""Ley de adherencia acero-hormigón esfuerzo-deslizamiento (τ-s), forma
del fib Model Code 2010 §6.1 (falla por arrancamiento/pull-out).

Aviso de honestidad técnica (ver plan): la FORMA de la curva (4 ramas:
potencial, meseta, descenso lineal, residual) está bien establecida en
la literatura secundaria consultada, pero los valores numéricos por
defecto de `tau_max, s1, s2, s3, tau_f` (Tabla 6.1-1 del MC2010) no
pudieron cotejarse contra el texto original del código durante la
investigación (paywall). Son razonables para una primera implementación
pero deben verificarse antes de usarse en un cálculo de producción; por
eso `BondParameters` permite pasar cualquiera de ellos explícitamente
para no depender de los defaults.

Solo se implementa el modo de falla por arrancamiento (pull-out); el
modo por hendimiento (splitting) requiere una tabla de parámetros que no
se pudo verificar con confianza suficiente, así que se rechaza
explícitamente en vez de inventar números.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fem2d_nl.exceptions import MaterialModelError


@dataclass(frozen=True)
class BondParameters:
    fcm: float  # MPa
    bar_diameter: float  # mm
    bond_condition: Literal["good", "other"] = "good"
    failure_mode: Literal["pullout"] = "pullout"
    clear_rib_spacing: float | None = None  # mm; None => 0.5*bar_diameter
    alpha: float = 0.4
    elastic_slip_fraction: float = 1e-3
    # overrides explícitos (si se conocen de ensayo, o para no depender
    # de los valores por defecto marcados como no verificados arriba)
    tau_max: float | None = None
    s1: float | None = None
    s2: float | None = None
    s3: float | None = None
    tau_f: float | None = None

    def resolve(self) -> tuple[float, float, float, float, float]:
        if self.failure_mode != "pullout":
            raise MaterialModelError(
                "Solo se implementa adherencia por arrancamiento (pullout); el modo "
                "por hendimiento (splitting) no tiene valores por defecto verificados."
            )
        sqrt_fcm = np.sqrt(self.fcm)
        if self.bond_condition == "good":
            tau_max = 2.5 * sqrt_fcm
            s1 = 1.0
            s2 = 2.0
        else:
            tau_max = 1.25 * sqrt_fcm
            s1 = 1.8
            s2 = 3.6
        tau_f = 0.40 * tau_max
        s3 = self.clear_rib_spacing if self.clear_rib_spacing is not None else 0.5 * self.bar_diameter

        return (
            self.tau_max if self.tau_max is not None else tau_max,
            self.s1 if self.s1 is not None else s1,
            self.s2 if self.s2 is not None else s2,
            self.s3 if self.s3 is not None else s3,
            self.tau_f if self.tau_f is not None else tau_f,
        )


class BondLaw:
    """Ley τ-s con rama elástica inicial (evita rigidez infinita en s=0,
    ya que el exponente `alpha<1` da dτ/ds→∞ en el origen) y descarga
    elástico-dañable con memoria del deslizamiento máximo alcanzado
    (suficiente para carga monótona/pushover; no modela fricción cíclica).
    """

    def __init__(self, params: BondParameters):
        self.p = params
        self.tau_max, self.s1, self.s2, self.s3, self.tau_f = params.resolve()
        if not (0.0 < self.s1 < self.s2 < self.s3):
            raise MaterialModelError(
                f"Parámetros de adherencia inconsistentes: se requiere 0 < s1 < s2 < s3 "
                f"(s1={self.s1}, s2={self.s2}, s3={self.s3})."
            )
        self.alpha = params.alpha
        self.s_e = params.elastic_slip_fraction * self.s1
        self.k0 = self._envelope_and_slope(np.array([self.s_e]))[0][0] / self.s_e
        self.k_min = 1e-6 * self.k0

    def _envelope_and_slope(self, s_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s1, s2, s3 = self.s1, self.s2, self.s3
        tau_max, tau_f, alpha = self.tau_max, self.tau_f, self.alpha

        tau = np.empty_like(s_abs)
        slope = np.empty_like(s_abs)

        branch1 = s_abs <= s1
        r1 = np.clip(s_abs[branch1] / s1, 1e-12, None)
        tau[branch1] = tau_max * r1**alpha
        slope[branch1] = tau_max * alpha / s1 * r1 ** (alpha - 1.0)

        branch2 = (s_abs > s1) & (s_abs <= s2)
        tau[branch2] = tau_max
        slope[branch2] = 0.0

        branch3 = (s_abs > s2) & (s_abs <= s3)
        tau[branch3] = tau_max - (tau_max - tau_f) * (s_abs[branch3] - s2) / (s3 - s2)
        slope[branch3] = -(tau_max - tau_f) / (s3 - s2)

        branch4 = s_abs > s3
        tau[branch4] = tau_f
        slope[branch4] = 0.0

        return tau, slope

    def initial_state(self, n_points: int) -> dict:
        return {"s_max": np.zeros(n_points)}

    def integrate(self, slip: np.ndarray, state: dict, dt: float):
        s_abs = np.abs(slip)
        sign = np.where(slip >= 0.0, 1.0, -1.0)

        s_abs_clip = np.where(s_abs < self.s_e, self.s_e, s_abs)
        env_val, env_slope = self._envelope_and_slope(s_abs_clip)
        env_val = np.where(s_abs < self.s_e, self.k0 * s_abs, env_val)
        env_slope = np.where(s_abs < self.s_e, self.k0, env_slope)

        s_max_prev = state["s_max"]
        loading = s_abs >= s_max_prev
        s_max_new = np.maximum(s_max_prev, s_abs)

        s_max_clip = np.where(s_max_new < self.s_e, self.s_e, s_max_new)
        env_at_max, _ = self._envelope_and_slope(s_max_clip)
        env_at_max = np.where(s_max_new < self.s_e, self.k0 * s_max_new, env_at_max)
        k_sec = np.where(s_max_new > 1e-12, env_at_max / np.maximum(s_max_new, 1e-12), self.k0)

        tau_abs = np.where(loading, env_val, k_sec * s_abs)
        raw_slope = np.where(loading, env_slope, k_sec)
        # k_min es un piso de MAGNITUD, no un mínimo algebraico: solo debe
        # regularizar los tramos de rigidez nula (meseta, residual), nunca
        # recortar hacia arriba una pendiente negativa legítima (rama
        # descendente), que sería un error de signo, no una regularización.
        dtau_ds_abs = np.where(np.abs(raw_slope) < self.k_min, self.k_min, raw_slope)

        tau = sign * tau_abs
        new_state = {"s_max": s_max_new}
        return tau, dtau_ds_abs, new_state
