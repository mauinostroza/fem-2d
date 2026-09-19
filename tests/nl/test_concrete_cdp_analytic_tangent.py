"""Sesión S7.4: tangente semi-analítica de `ConcreteCDP` (`tangent_mode=
"analytic"`, opt-in — el default sigue siendo `"numeric"`, sin cambios de
comportamiento para S1-S6).

Ver el docstring de `ConcreteCDP._integrate_analytic`/`_active_tangent`
para la deducción. Resumen de lo que es analítico vs. diferencias finitas:
analítico en la parte de mayor riesgo algebraico (derivada de autovalores
del predictor elástico, teorema de la función implícita sobre el sistema
local YA validado de `_numeric_jacobian`, rotación de vuelta a Cartesiano);
diferencias finitas SOLO en funciones locales baratas y sin iteración
(`_kappas`, `_flow_and_weight`, `_d_t`/`_d_c`) — nunca se rehace el return
mapping completo (eso es lo caro de la tangente numérica original: 3
llamadas extra a `_compute`).

**Bug real encontrado y corregido durante la validación de esta sesión**:
el paso de diferencias finitas para `d(d_t)/d(kappa_t)` estaba ligado a
`fcm` (razonable para perturbar esfuerzos `s1,s2`, pero `kappa_t`/`kappa_c`
son variables de endurecimiento de escala MUY distinta — p. ej.
`kappa_t~1e-5` con tablas de espaciado similar) y saltaba varios segmentos
de la tabla interpolada, dando una derivada completamente errónea (~60-
114% de error en la tangente resultante). Se corrigió escalando el paso
al rango propio de cada tabla (`kappa_t[-1]`/`kappa_c[-1]`), no a `fcm`.

**Limitación real, no oculta**: en autovalores EXACTAMENTE repetidos
(p. ej. compresión equibiaxial exacta, `s1_tr=s2_tr`), la derivada de
autovalores `n_i⊗n_i` no está definida de forma única (el mismo caso
límite "autovalores repetidos" señalado como riesgo en el plan original).
Se maneja con una caída a la tangente numérica SOLO para ese subconjunto
raro de puntos (barato de aislar, ver `_integrate_analytic`), no con una
fórmula regularizada sin verificar.
"""

import numpy as np
import pytest

from fem2d_nl.exceptions import MaterialModelError
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP

FCK = 25.0


def _pair(char_length=50.0):
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK)
    numeric = ConcreteCDP(params, char_length=char_length, tangent_mode="numeric")
    analytic = ConcreteCDP(params, char_length=char_length, tangent_mode="analytic")
    return numeric, analytic


def test_invalid_tangent_mode_rejected():
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK)
    with pytest.raises(MaterialModelError):
        ConcreteCDP(params, char_length=50.0, tangent_mode="bogus")


def test_default_tangent_mode_is_numeric_unchanged():
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK)
    cdp = ConcreteCDP(params, char_length=50.0)
    assert cdp.tangent_mode == "numeric"


@pytest.mark.parametrize(
    "strain,label",
    [
        ([1e-5, -2e-6, 0.0], "uniaxial tension elastic"),
        ([3e-4, -6e-5, 0.0], "uniaxial tension cracked"),
        ([2e-3, -4e-4, 0.0], "uniaxial tension deep softening"),
        ([-1.5e-3, 3e-4, 0.0], "uniaxial compression"),
        ([-1e-3, -1e-3, 0.0], "biaxial compression (autovalores repetidos)"),
        ([5e-4, -1e-4, 3e-4], "tension + shear"),
        ([-8e-4, -2e-4, 5e-4], "compression + shear"),
        ([0.0, 0.0, 4e-4], "pure shear"),
        ([1e-6, -2e-7, 0.0], "tiny elastic (inactivo)"),
    ],
)
def test_analytic_tangent_matches_numeric_across_states(strain, label):
    numeric, analytic = _pair()
    strain_arr = np.array([strain])
    state = numeric.initial_state(1)

    stress_n, tan_n, new_state_n = numeric.integrate(strain_arr, state, dt=1.0)
    stress_a, tan_a, new_state_a = analytic.integrate(strain_arr, state, dt=1.0)

    assert stress_a == pytest.approx(stress_n, abs=1e-8, rel=1e-8), label
    tan_scale = max(np.max(np.abs(tan_n)), 1.0)
    assert np.max(np.abs(tan_a - tan_n)) / tan_scale < 1e-3, label
    for key in ("eps_pl", "kappa_t", "kappa_c"):
        assert new_state_a[key] == pytest.approx(new_state_n[key], abs=1e-8, rel=1e-6), (label, key)


@pytest.mark.slow
def test_analytic_tangent_matches_numeric_random_sweep():
    """Barrido amplio y aleatorio (semilla fija, reproducible): ningún
    estado debe superar el 1% de error relativo en la tangente, salvo
    los pocos que el propio return mapping rechaza (deformaciones fuera
    de rango, no relacionado con la tangente)."""
    numeric, analytic = _pair()
    rng = np.random.default_rng(42)
    worst = 0.0
    evaluated = 0
    for _ in range(150):
        strain = (rng.random((1, 3)) - 0.5) * np.array([[2.5e-3, 2.5e-3, 2e-3]])
        state = numeric.initial_state(1)
        try:
            _, tan_n, _ = numeric.integrate(strain, state, dt=1.0)
            _, tan_a, _ = analytic.integrate(strain, state, dt=1.0)
        except MaterialModelError:
            continue
        evaluated += 1
        tan_err = np.max(np.abs(tan_n - tan_a)) / max(np.max(np.abs(tan_n)), 1.0)
        worst = max(worst, tan_err)
        assert tan_err < 1e-2, f"strain={strain}, err={tan_err:.3e}"
    assert evaluated > 100  # el barrido debe haber evaluado una cantidad razonable de puntos
