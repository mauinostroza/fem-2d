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
    def __init__(self, params: CDPParameters, char_length: float, tangent_mode: str = "numeric"):
        """`tangent_mode` (S7): `"numeric"` (default, sin cambios respecto
        de S1-S6) o `"analytic"` — tangente SEMI-analítica opt-in, ver
        `_compute_with_analytic_tangent`. Nunca cambia por defecto el
        comportamiento ya validado."""
        if tangent_mode not in ("numeric", "analytic"):
            raise MaterialModelError(f"tangent_mode debe ser 'numeric' o 'analytic', no {tangent_mode!r}.")
        self.tangent_mode = tangent_mode
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
        return x[:, 0], x[:, 1], kappa_t, kappa_c, x[:, 2]

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
            s1_a, s2_a, kt_a, kc_a, _dlam_a = self._solve_active(
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
        if self.tangent_mode == "analytic":
            return self._integrate_analytic(strain, state)
        stress, eps_pl_new, kappa_t, kappa_c = self._compute(strain, state)
        tangent = self._numeric_tangent(strain, state, stress)
        new_state = {"eps_pl": eps_pl_new, "kappa_t": kappa_t, "kappa_c": kappa_c}
        return stress, tangent, new_state

    # ---- tangente semi-analítica (S7, opt-in) ---------------------------------
    def _integrate_analytic(self, strain: np.ndarray, state: dict):
        """Como `_compute` + `_numeric_tangent`, pero con una tangente
        SEMI-analítica: analítica en la parte de mayor riesgo algebraico
        bajo (rotación/autovalores, regla de la cadena de la rotación y
        el daño), y por diferencias finitas SOLO en funciones locales
        baratas y sin iteración (`_kappas`, `_flow_and_weight`, `_d_t`/
        `_d_c`) — nunca vuelve a resolver el return mapping completo
        (eso es lo caro de `_numeric_tangent`: 3 llamadas extra a
        `_compute`, cada una con su propio Newton local). El jacobiano
        LOCAL 3x3 del sistema de retorno sigue siendo el de
        `_numeric_jacobian` ya validado (S3) — no se rederiva a mano,
        para no duplicar el riesgo algebraico más alto del proyecto."""
        eps_pl_prev = state["eps_pl"]
        kappa_t_prev = state["kappa_t"]
        kappa_c_prev = state["kappa_c"]
        n = strain.shape[0]
        fcm_scale = max(self.fcm, 1.0)

        sigma_tr = np.einsum("ij,nj->ni", self.d0, strain - eps_pl_prev)
        sxx, syy, sxy = sigma_tr[:, 0], sigma_tr[:, 1], sigma_tr[:, 2]
        u = (sxx - syy) / 2.0
        center = (sxx + syy) / 2.0
        radius = np.sqrt(u**2 + sxy**2)
        s1_tr = center + radius
        s2_tr = center - radius
        rot_active = radius > 1e-9 * fcm_scale
        theta = np.where(rot_active, 0.5 * np.arctan2(2.0 * sxy, sxx - syy), 0.0)

        f_trial = self._yield(s1_tr, s2_tr, kappa_t_prev, kappa_c_prev)
        active = f_trial > _TOL * fcm_scale

        s1, s2 = s1_tr.copy(), s2_tr.copy()
        kappa_t, kappa_c = kappa_t_prev.copy(), kappa_c_prev.copy()
        dlam = np.zeros(n)
        if active.any():
            s_tr_active = np.stack([s1_tr[active], s2_tr[active]], axis=-1)
            s1_a, s2_a, kt_a, kc_a, dlam_a = self._solve_active(
                s_tr_active, kappa_t_prev[active], kappa_c_prev[active]
            )
            s1[active], s2[active] = s1_a, s2_a
            kappa_t[active], kappa_c[active] = kt_a, kc_a
            dlam[active] = dlam_a

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

        tangent = np.tile(self.d0[None, :, :], (n, 1, 1))  # elástico por defecto (puntos inactivos)
        # Activos con autovalores bien separados: tangente semi-analítica.
        active_smooth = active & rot_active
        if active_smooth.any():
            idx = np.nonzero(active_smooth)[0]
            tangent[idx] = self._active_tangent(
                s1_tr[idx], s2_tr[idx], sxx[idx], syy[idx], sxy[idx], u[idx], radius[idx],
                rot_active[idx], cos_t[idx], sin_t[idx],
                s1[idx], s2[idx], dlam[idx], kappa_t_prev[idx], kappa_c_prev[idx],
                kappa_t[idx], kappa_c[idx], d_t[idx], d_c[idx], r_weight[idx], sigma_eff[idx],
            )
        # Activos con autovalores (casi) repetidos (p. ej. compresión
        # equibiaxial exacta): la derivada de autovalores `n_i⊗n_i` es
        # singular/no está definida de forma única ahí (verificado en esta
        # sesión: el error de la tangente semi-analítica se dispara justo
        # en este caso, y solo ahí — a partir de ~0.01% fuera del punto
        # exacto ya coincide con la numérica dentro de <1e-4). Es el mismo
        # caso límite "autovalores repetidos" ya señalado como riesgo en
        # el plan original — en vez de forzar una fórmula regularizada sin
        # verificar, se cae a la tangente numérica (ya validada) solo para
        # este subconjunto raro, que es barato de aislar.
        active_singular = active & ~rot_active
        if active_singular.any():
            idx = np.nonzero(active_singular)[0]
            sub_state = {k: v[idx] for k, v in state.items()}
            tangent[idx] = self._numeric_tangent(strain[idx], sub_state, stress[idx])

        new_state = {"eps_pl": eps_pl_new, "kappa_t": kappa_t, "kappa_c": kappa_c}
        return stress, tangent, new_state

    def _active_tangent(
        self, s1_tr, s2_tr, sxx, syy, sxy, u, radius, rot_active, cos_t, sin_t,
        s1, s2, dlam, kt_prev, kc_prev, kt, kc, d_t, d_c, r_weight, sigma_eff,
    ) -> np.ndarray:
        n = s1.shape[0]
        fcm_scale = max(self.fcm, 1.0)
        radius_safe = np.where(rot_active, radius, 1.0)

        # d(s1_tr,s2_tr)/d(strain): fórmula analítica estándar de la derivada
        # de autovalores de un tensor simétrico 2x2 (n_i⊗n_i), aquí en forma
        # de Voigt directa a partir de (u,v,radio) — ya usada (sin derivar)
        # para el propio ángulo de rotación en `_compute`.
        row1 = np.stack([0.5 + 0.5 * u / radius_safe, 0.5 - 0.5 * u / radius_safe, sxy / radius_safe], axis=-1)
        row2 = np.stack([0.5 - 0.5 * u / radius_safe, 0.5 + 0.5 * u / radius_safe, -sxy / radius_safe], axis=-1)
        fallback = np.array([0.5, 0.5, 0.0])
        row1 = np.where(rot_active[:, None], row1, fallback)
        row2 = np.where(rot_active[:, None], row2, fallback)
        d_str_dstrain = np.stack([row1, row2], axis=1) @ self.d0[None, :, :]  # (n,2,3)

        # Jacobiano local ya validado (S3) en el punto convergido -> dx/d(s_tr)
        # por el teorema de la función implícita (solo s1_tr,s2_tr entran
        # explícitamente en r1,r2 con derivada -1; r3 no depende de ellos
        # directamente), y de ahí dx/d(strain) encadenando con el paso anterior.
        x_conv = np.stack([s1, s2, dlam], axis=-1)
        s_tr = np.stack([s1_tr, s2_tr], axis=-1)
        jac = self._numeric_jacobian(x_conv, s_tr, kt_prev, kc_prev)  # (n,3,3)
        rhs = np.zeros((n, 3, 2))
        rhs[:, 0, 0] = 1.0
        rhs[:, 1, 1] = 1.0
        dx_dstr = np.linalg.solve(jac, rhs)  # (n,3,2)
        dx_dstrain = dx_dstr @ d_str_dstrain  # (n,3,3): filas = d(s1,s2,dlam)/d(strain)
        ds1_dstrain = dx_dstrain[:, 0, :]
        ds2_dstrain = dx_dstrain[:, 1, :]
        ddlam_dstrain = dx_dstrain[:, 2, :]

        # d(kappa_t,kappa_c)/d(s1,s2,dlam): diferencias finitas de `_kappas`
        # (función cerrada, sin Newton — barata, no es el jacobiano de retorno)
        h_s, h_l = 1e-6 * fcm_scale, 1e-9
        kt0, kc0, _, _ = self._kappas(s1, s2, dlam, kt_prev, kc_prev)
        kt_1, kc_1, _, _ = self._kappas(s1 + h_s, s2, dlam, kt_prev, kc_prev)
        kt_2, kc_2, _, _ = self._kappas(s1, s2 + h_s, dlam, kt_prev, kc_prev)
        kt_3, kc_3, _, _ = self._kappas(s1, s2, dlam + h_l, kt_prev, kc_prev)
        dkt_ds1, dkc_ds1 = (kt_1 - kt0) / h_s, (kc_1 - kc0) / h_s
        dkt_ds2, dkc_ds2 = (kt_2 - kt0) / h_s, (kc_2 - kc0) / h_s
        dkt_dl, dkc_dl = (kt_3 - kt0) / h_l, (kc_3 - kc0) / h_l
        dkt_dstrain = dkt_ds1[:, None] * ds1_dstrain + dkt_ds2[:, None] * ds2_dstrain + dkt_dl[:, None] * ddlam_dstrain
        dkc_dstrain = dkc_ds1[:, None] * ds1_dstrain + dkc_ds2[:, None] * ds2_dstrain + dkc_dl[:, None] * ddlam_dstrain

        # d(theta)/d(strain): derivada analítica de 0.5*atan2(2*sxy,sxx-syy)
        r2 = radius**2
        r2_safe = np.where(rot_active, r2, 1.0)
        dtheta_dstr = np.stack(
            [np.where(rot_active, -sxy / (4.0 * r2_safe), 0.0),
             np.where(rot_active, sxy / (4.0 * r2_safe), 0.0),
             np.where(rot_active, (sxx - syy) / (4.0 * r2_safe), 0.0)],
            axis=-1,
        )  # (n,3)
        dtheta_dstrain = np.einsum("ni,ij->nj", dtheta_dstr, self.d0)  # (n,3)

        # d(sigma_eff)/d(strain): a través de s1,s2 (ya tenemos sus
        # sensibilidades) Y de theta (rotación fija del predictor, pero
        # también depende de la deformación)
        dsigxx_dtheta = -2.0 * cos_t * sin_t * (s1 - s2)
        dsigyy_dtheta = 2.0 * sin_t * cos_t * (s1 - s2)
        dsigxy_dtheta = (s1 - s2) * (cos_t**2 - sin_t**2)

        dsigeff_dstrain = np.empty((n, 3, 3))
        dsigeff_dstrain[:, 0, :] = (
            cos_t[:, None] ** 2 * ds1_dstrain + sin_t[:, None] ** 2 * ds2_dstrain
            + dsigxx_dtheta[:, None] * dtheta_dstrain
        )
        dsigeff_dstrain[:, 1, :] = (
            sin_t[:, None] ** 2 * ds1_dstrain + cos_t[:, None] ** 2 * ds2_dstrain
            + dsigyy_dtheta[:, None] * dtheta_dstrain
        )
        dsigeff_dstrain[:, 2, :] = (
            (sin_t * cos_t)[:, None] * (ds1_dstrain - ds2_dstrain) + dsigxy_dtheta[:, None] * dtheta_dstrain
        )

        # d(daño)/d(strain): diferencias finitas SOLO en `_d_t`/`_d_c`
        # (interpolación de tabla) y `_flow_and_weight` (cerrada, sin
        # Newton), encadenadas con las sensibilidades ya analíticas de
        # (s1,s2,kappa_t,kappa_c) obtenidas arriba.
        #
        # OJO con el paso de esta FD en particular: `kappa_t`/`kappa_c`
        # son variables de endurecimiento con escala MUY distinta a la de
        # los esfuerzos (p. ej. kappa_t puede ser ~1e-5, con espaciado de
        # tabla del mismo orden) — un paso ligado a `fcm` (como el de
        # `h_s` más abajo, correcto para s1/s2) salta varios segmentos de
        # la tabla interpolada y da una derivada completamente errónea
        # (bug real encontrado y corregido durante la validación de esta
        # sesión, ver `tests/nl/test_concrete_cdp.py`). El paso debe ser
        # una fracción chica del RANGO TOTAL de cada tabla, no de `fcm`.
        h_kt = 1e-6 * max(self.kappa_t[-1], 1e-12)
        h_kc = 1e-6 * max(self.kappa_c[-1], 1e-12)
        ddt_dkt = (self._d_t(kt + h_kt) - d_t) / h_kt
        ddc_dkc = (self._d_c(kc + h_kc) - d_c) / h_kc

        r0 = r_weight
        r_1 = self._flow_and_weight(s1 + h_s, s2)[3]
        r_2 = self._flow_and_weight(s1, s2 + h_s)[3]
        dr_ds1 = (r_1 - r0) / h_s
        dr_ds2 = (r_2 - r0) / h_s
        dr_dstrain = dr_ds1[:, None] * ds1_dstrain + dr_ds2[:, None] * ds2_dstrain

        dst_dstrain = -self.p.w_t * dr_dstrain
        dsc_dstrain = self.p.w_c * dr_dstrain
        ddt_dstrain = ddt_dkt[:, None] * dkt_dstrain
        ddc_dstrain = ddc_dkc[:, None] * dkc_dstrain

        s_t = 1.0 - self.p.w_t * r_weight
        s_c = 1.0 - self.p.w_c * (1.0 - r_weight)
        term1 = 1.0 - s_t * d_c
        term2 = 1.0 - s_c * d_t
        dterm1 = -(dst_dstrain * d_c[:, None] + s_t[:, None] * ddc_dstrain)
        dterm2 = -(dsc_dstrain * d_t[:, None] + s_c[:, None] * ddt_dstrain)
        damage = 1.0 - term1 * term2
        ddamage_dstrain = -(dterm1 * term2[:, None] + term1[:, None] * dterm2)

        # stress_i = (1-daño)*sigma_eff_i  ->  regla del producto
        tangent = (1.0 - damage)[:, None, None] * dsigeff_dstrain - sigma_eff[:, :, None] * ddamage_dstrain[:, None, :]
        return tangent

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
