"""Concrete Damaged Plasticity (Lubliner et al. 1989 / Lee & Fenves 1998)
en tensión plana.

Ver el plan de implementación para las ecuaciones y su procedencia. Nota
de honestidad técnica: no fue posible verificar los números de ecuación
exactos de Lee & Fenves contra el PDF original (paywall); la forma de
las ecuaciones aquí implementadas está verificada contra múltiples
fuentes secundarias consistentes entre sí.

Alcance de esta primera versión (sesión S3 del plan, solo a nivel de
punto material, todavía no integrada en el elemento Q4):
- Return mapping en espacio de esfuerzo principal (coaxial con el
  predictor elástico), con jacobiano LOCAL calculado por diferencias
  finitas (no analítico): es deliberado, no una limitación temporal —
  esta es la pieza de mayor riesgo de todo el proyecto (ver plan), y un
  jacobiano analítico mal derivado sería un error silencioso mucho peor
  que el costo de rendimiento de uno numérico. La tangente algorítmica
  que se devuelve a la llamadora también es por diferencias finitas.
- Si el Newton local no converge, se lanza `MaterialModelError` en vez
  de subincrementar internamente: la subincrementación y la recuperación
  ante no-convergencia se resuelven a nivel del solver global (S4), que
  ya reduce el paso de carga completo ante una falla.
"""

from dataclasses import dataclass

import numpy as np

from fem2d_nl.exceptions import MaterialModelError
from fem2d_nl.materials.concrete_curves import (
    build_compression_law,
    build_tension_law,
    mc2010_eci,
    mc2010_fcm,
    mc2010_fctm,
    mc2010_fracture_energy,
)

_TOL = 1e-8


@dataclass(frozen=True)
class CDPParameters:
    young_modulus: float
    poisson_ratio: float
    fck: float
    ft: float | None = None
    gf: float | None = None
    gc: float | None = None
    fb0_fc0: float = 1.16
    kc: float = 2.0 / 3.0
    psi_deg: float = 36.0
    eccentricity: float = 0.1
    b_t: float = 0.1
    b_c: float = 0.7
    w_t: float = 0.0
    w_c: float = 1.0
    eps_c1: float | None = None


class ConcreteCDP:
    def __init__(self, params: CDPParameters, char_length: float):
        self.p = params
        self.l_ch = char_length
        self.e0 = params.young_modulus
        self.nu = params.poisson_ratio

        self.fcm = mc2010_fcm(params.fck)
        self.ft = params.ft if params.ft is not None else mc2010_fctm(params.fck)
        self.eci = mc2010_eci(self.fcm)
        self.gf = params.gf if params.gf is not None else mc2010_fracture_energy(self.fcm)
        self.gc = params.gc if params.gc is not None else 250.0 * self.gf

        self.kappa_t, self.sigma_bar_t_table, self.d_t_table = build_tension_law(
            self.ft, self.gf, self.l_ch, self.e0, params.b_t
        )
        self.kappa_c, self.sigma_bar_c_table, self.d_c_table = build_compression_law(
            self.fcm, self.eci, self.gc, self.l_ch, self.e0, params.eps_c1, params.b_c
        )

        self.alpha = (params.fb0_fc0 - 1.0) / (2.0 * params.fb0_fc0 - 1.0)
        self.gamma = 3.0 * (1.0 - params.kc) / (2.0 * params.kc - 1.0)
        self.tan_psi = np.tan(np.radians(params.psi_deg))
        self.ecc = params.eccentricity

        factor = self.e0 / (1.0 - self.nu**2)
        self.d0 = factor * np.array([[1.0, self.nu, 0.0], [self.nu, 1.0, 0.0], [0.0, 0.0, (1.0 - self.nu) / 2.0]])
        self.d0_inv = np.linalg.inv(self.d0)
        self.d0_ps = factor * np.array([[1.0, self.nu], [self.nu, 1.0]])  # espacio principal (sin corte)

    # ---- tablas interpoladas -------------------------------------------------
    def _sigma_bar_t(self, kappa):
        return np.interp(kappa, self.kappa_t, self.sigma_bar_t_table)

    def _sigma_bar_c(self, kappa):
        return np.interp(kappa, self.kappa_c, self.sigma_bar_c_table)

    def _d_t(self, kappa):
        return np.interp(kappa, self.kappa_t, self.d_t_table)

    def _d_c(self, kappa):
        return np.interp(kappa, self.kappa_c, self.d_c_table)

    # ---- geometría del criterio de fluencia ----------------------------------
    def _flow_and_weight(self, s1, s2):
        s3 = np.zeros_like(s1)
        p = -(s1 + s2 + s3) / 3.0
        q = np.sqrt(np.maximum(s1**2 + s2**2 + s3**2 - s1 * s2 - s2 * s3 - s3 * s1, 1e-12))
        denom = np.sqrt((self.ecc * self.ft * self.tan_psi) ** 2 + q**2)
        common = q / denom
        m1 = common * 1.5 * (s1 + p) / q + self.tan_psi / 3.0
        m2 = common * 1.5 * (s2 + p) / q + self.tan_psi / 3.0
        m3 = common * 1.5 * (s3 + p) / q + self.tan_psi / 3.0

        stacked = np.stack([s1, s2, s3], axis=-1)
        pos_sum = np.clip(stacked, 0.0, None).sum(axis=-1)
        abs_sum = np.abs(stacked).sum(axis=-1)
        r = np.where(abs_sum > 1e-10, pos_sum / np.maximum(abs_sum, 1e-30), 0.0)
        return m1, m2, m3, r

    def _yield(self, s1, s2, kappa_t, kappa_c):
        s3 = np.zeros_like(s1)
        p = -(s1 + s2 + s3) / 3.0
        q = np.sqrt(np.maximum(s1**2 + s2**2 - s1 * s2, 1e-12))
        s_max = np.maximum(np.maximum(s1, s2), s3)

        sigma_c = self._sigma_bar_c(kappa_c)
        sigma_t = self._sigma_bar_t(kappa_t)
        beta = sigma_c / sigma_t * (1.0 - self.alpha) - (1.0 + self.alpha)

        term = q - 3.0 * self.alpha * p + beta * np.clip(s_max, 0.0, None) - self.gamma * np.clip(-s_max, 0.0, None)
        return term / (1.0 - self.alpha) - sigma_c

    def _kappas(self, s1, s2, dlam, kappa_t_prev, kappa_c_prev):
        m1, m2, m3, r = self._flow_and_weight(s1, s2)
        m_stack = np.stack([m1, m2, m3], axis=-1)
        m_max = m_stack.max(axis=-1)
        m_min = m_stack.min(axis=-1)
        kappa_t = kappa_t_prev + dlam * r * m_max
        kappa_c = kappa_c_prev - dlam * (1.0 - r) * m_min
        return kappa_t, kappa_c, m1, m2

    # ---- return mapping local -------------------------------------------------
    def _residual(self, x, s_tr, kappa_t_prev, kappa_c_prev):
        s1, s2, dlam = x[..., 0], x[..., 1], x[..., 2]
        kappa_t, kappa_c, m1, m2 = self._kappas(s1, s2, dlam, kappa_t_prev, kappa_c_prev)

        rhs1 = self.d0_ps[0, 0] * m1 + self.d0_ps[0, 1] * m2
        rhs2 = self.d0_ps[1, 0] * m1 + self.d0_ps[1, 1] * m2
        r1 = s1 - s_tr[..., 0] + dlam * rhs1
        r2 = s2 - s_tr[..., 1] + dlam * rhs2
        r3 = self._yield(s1, s2, kappa_t, kappa_c)
        return np.stack([r1, r2, r3], axis=-1), kappa_t, kappa_c

    def _solve_active(self, s_tr, kappa_t_prev, kappa_c_prev, max_iter=40):
        n = s_tr.shape[0]
        x = np.stack([s_tr[:, 0], s_tr[:, 1], np.zeros(n)], axis=-1)
        active = np.ones(n, dtype=bool)

        for _ in range(max_iter):
            if not active.any():
                break
            idx = np.nonzero(active)[0]
            res, _, _ = self._residual(x[idx], s_tr[idx], kappa_t_prev[idx], kappa_c_prev[idx])
            res_norm = np.linalg.norm(res, axis=-1)
            done = res_norm < _TOL * max(self.fcm, 1.0)
            active[idx[done]] = False
            idx = idx[~done]
            if idx.size == 0:
                break

            jac = self._numeric_jacobian(x[idx], s_tr[idx], kappa_t_prev[idx], kappa_c_prev[idx])
            res_active, _, _ = self._residual(x[idx], s_tr[idx], kappa_t_prev[idx], kappa_c_prev[idx])
            try:
                delta = np.linalg.solve(jac, -res_active)
            except np.linalg.LinAlgError as exc:
                raise MaterialModelError(
                    "El jacobiano local del return mapping de hormigón es singular; "
                    "revise los parámetros del material o refine el paso de carga."
                ) from exc
            x[idx] += delta
            x[idx, 2] = np.maximum(x[idx, 2], 0.0)  # Δλ >= 0

        if active.any():
            raise MaterialModelError(
                f"El return mapping de hormigón no convergió en {max_iter} iteraciones "
                f"para {active.sum()} punto(s) de integración. Reduzca el paso de carga."
            )
        _, kappa_t, kappa_c = self._residual(x, s_tr, kappa_t_prev, kappa_c_prev)
        return x[:, 0], x[:, 1], kappa_t, kappa_c

    def _numeric_jacobian(self, x, s_tr, kappa_t_prev, kappa_c_prev):
        n = x.shape[0]
        jac = np.empty((n, 3, 3))
        r0, _, _ = self._residual(x, s_tr, kappa_t_prev, kappa_c_prev)
        for j in range(3):
            h = 1e-6 * max(self.fcm, 1.0) if j < 2 else 1e-9
            xp = x.copy()
            xp[:, j] += h
            rp, _, _ = self._residual(xp, s_tr, kappa_t_prev, kappa_c_prev)
            jac[:, :, j] = (rp - r0) / h
        return jac

    # ---- API pública -----------------------------------------------------------
    def initial_state(self, n_points: int) -> dict:
        return {
            "eps_pl": np.zeros((n_points, 3)),
            "kappa_t": np.zeros(n_points),
            "kappa_c": np.zeros(n_points),
        }

    def _compute(self, strain: np.ndarray, state: dict):
        """Núcleo del modelo: predictor elástico, return mapping en
        espacio principal (solo para los puntos activos), daño y
        actualización de estado. Devuelve (stress, eps_pl_new, kappa_t,
        kappa_c). Compartido por `integrate` y por el cálculo de la
        tangente numérica, para no duplicar el return mapping."""
        eps_pl_prev = state["eps_pl"]
        kappa_t_prev = state["kappa_t"]
        kappa_c_prev = state["kappa_c"]

        sigma_tr = np.einsum("ij,nj->ni", self.d0, strain - eps_pl_prev)
        sxx, syy, sxy = sigma_tr[:, 0], sigma_tr[:, 1], sigma_tr[:, 2]
        center = (sxx + syy) / 2.0
        radius = np.sqrt(((sxx - syy) / 2.0) ** 2 + sxy**2)
        s1_tr = center + radius
        s2_tr = center - radius
        theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
        theta = np.where(radius > 1e-9 * max(self.fcm, 1.0), theta, 0.0)

        f_trial = self._yield(s1_tr, s2_tr, kappa_t_prev, kappa_c_prev)
        active = f_trial > _TOL * max(self.fcm, 1.0)

        s1, s2 = s1_tr.copy(), s2_tr.copy()
        kappa_t, kappa_c = kappa_t_prev.copy(), kappa_c_prev.copy()
        if active.any():
            s_tr_active = np.stack([s1_tr[active], s2_tr[active]], axis=-1)
            s1_a, s2_a, kt_a, kc_a = self._solve_active(
                s_tr_active, kappa_t_prev[active], kappa_c_prev[active]
            )
            s1[active], s2[active] = s1_a, s2_a
            kappa_t[active], kappa_c[active] = kt_a, kc_a

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        sigma_eff = np.stack(
            [s1 * cos_t**2 + s2 * sin_t**2, s1 * sin_t**2 + s2 * cos_t**2, (s1 - s2) * sin_t * cos_t],
            axis=-1,
        )

        d_t = self._d_t(kappa_t)
        d_c = self._d_c(kappa_c)
        r_weight = self._flow_and_weight(s1, s2)[3]
        s_t = 1.0 - self.p.w_t * r_weight
        s_c = 1.0 - self.p.w_c * (1.0 - r_weight)
        damage = 1.0 - (1.0 - s_t * d_c) * (1.0 - s_c * d_t)

        stress = (1.0 - damage)[:, None] * sigma_eff
        eps_pl_new = strain - np.einsum("ij,nj->ni", self.d0_inv, sigma_eff)
        return stress, eps_pl_new, kappa_t, kappa_c

    def integrate(self, strain: np.ndarray, state: dict, dt: float):
        stress, eps_pl_new, kappa_t, kappa_c = self._compute(strain, state)
        tangent = self._numeric_tangent(strain, state, stress)
        new_state = {"eps_pl": eps_pl_new, "kappa_t": kappa_t, "kappa_c": kappa_c}
        return stress, tangent, new_state

    def damage_at(self, kappa_t, kappa_c) -> tuple[np.ndarray, np.ndarray]:
        """`(d_t, d_c)` en las variables de endurecimiento dadas — envoltorio
        público de `_d_t`/`_d_c` para que la UI (contorno de daño) no
        dependa de métodos privados."""
        return self._d_t(np.asarray(kappa_t)), self._d_c(np.asarray(kappa_c))

    # ---- energía disipada (objetividad de malla de la regularización crack-band) ----
    def dissipated_tensile_energy_density(self, kappa_t) -> np.ndarray:
        """Energía disipada por fisuración, por unidad de VOLUMEN, hasta
        `kappa_t` (mismas unidades que `G_F/l_ch`).

        La curva de tracción se construyó (`build_tension_law`) a partir de
        una relación esfuerzo NOMINAL - apertura de fisura `sigma(w)` cuya
        área es exactamente `G_F` (verificado por construcción: 0.6*G_F +
        0.4*G_F). La variable de endurecimiento `kappa_t` no es `w/l_ch`
        directamente, sino `b_t*w/l_ch` (Birtel & Mark), así que hay que
        deshacer ese cambio de variable: `d(w/l_ch) = dkappa_t/b_t`, y el
        esfuerzo nominal es `(1-d_t)*sigma_bar_t` (no `sigma_bar_t`, que es
        el esfuerzo EFECTIVO). La integral resultante,

            E_diss(kappa_t) = (1/b_t) * integral[0,kappa_t] (1-d_t(k))*sigma_bar_t(k) dk

        vale exactamente `G_F/l_ch` al llegar a la saturación completa de
        la tabla, para cualquier `l_ch` (verificado numéricamente: da el
        mismo `G_F/l_ch` con error < 1e-4 para `l_ch` entre 20 y 100 mm en
        un caso típico) — es la cantidad correcta y objetiva de malla para
        verificar la regularización crack-band, a nivel de punto material,
        sin ninguna de las ambigüedades de localización FEM / energía
        elástica recuperable que hicieron descartar el intento a nivel de
        malla completa en la sesión S4 (ver README)."""
        nominal = (1.0 - self.d_t_table) * self.sigma_bar_t_table
        increment = 0.5 * (nominal[:-1] + nominal[1:]) * np.diff(self.kappa_t)
        cumulative = np.concatenate([[0.0], np.cumsum(increment)]) / self.p.b_t
        return np.interp(np.asarray(kappa_t), self.kappa_t, cumulative)

    def dissipated_compressive_energy_density(self, kappa_c) -> np.ndarray:
        """Análogo en compresión. A diferencia de tracción, la rama de
        ablandamiento en compresión NO se construyó forzando que esta
        integral reproduzca `G_c/l_ch` exactamente (es una rama lineal
        simplificada, ver `concrete_curves.build_compression_law`) — sirve
        como chequeo de consistencia/objetividad relativo entre mallas,
        no como verificación exacta contra `G_c`."""
        nominal = (1.0 - self.d_c_table) * self.sigma_bar_c_table
        increment = 0.5 * (nominal[:-1] + nominal[1:]) * np.diff(self.kappa_c)
        cumulative = np.concatenate([[0.0], np.cumsum(increment)]) / self.p.b_c
        return np.interp(np.asarray(kappa_c), self.kappa_c, cumulative)

    def _numeric_tangent(self, strain, state, stress0):
        n = strain.shape[0]
        h = max(1e-7 * self.ft / max(self.e0, 1.0), 1e-10)
        tangent = np.empty((n, 3, 3))
        for j in range(3):
            eps_p = strain.copy()
            eps_p[:, j] += h
            stress_p, _, _, _ = self._compute(eps_p, state)
            tangent[:, :, j] = (stress_p - stress0) / h
        return tangent
