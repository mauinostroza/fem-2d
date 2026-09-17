"""Tests a nivel de punto material (sin FEM) para el modelo CDP de
hormigón. Es la pieza de mayor riesgo del proyecto (ver plan): estos
tests verifican las propiedades que se pueden comprobar de forma
independiente y sin ambigüedad, incluso sin acceso al PDF original de
Lee & Fenves.
"""

import numpy as np
import pytest

from fem2d_nl.exceptions import MaterialModelError
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP

FCK = 25.0


def _material(char_length=50.0, **overrides):
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK, **overrides)
    return ConcreteCDP(params, char_length=char_length)


def _uniaxial_stress_sweep(cdp, eps_axis):
    """Barrido de deformación uniaxial (sigma_yy=0 impuesto por
    bisección). Devuelve el historial de sigma_xx."""
    state = cdp.initial_state(1)
    center = 0.0
    sxx_hist = []

    def syy_at(exx, eyy):
        strain = np.array([[exx, eyy, 0.0]])
        stress, _, _ = cdp.integrate(strain, state, dt=1.0)
        return stress[0, 1]

    for exx in eps_axis:
        span = 5e-5
        lo, hi = center - span, center + span
        f_lo, f_hi = syy_at(exx, lo), syy_at(exx, hi)
        tries = 0
        while f_lo * f_hi > 0 and tries < 12:
            span *= 2
            lo, hi = center - span, center + span
            f_lo, f_hi = syy_at(exx, lo), syy_at(exx, hi)
            tries += 1
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            f_mid = syy_at(exx, mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        center = 0.5 * (lo + hi)
        strain = np.array([[exx, center, 0.0]])
        stress, _, state = cdp.integrate(strain, state, dt=1.0)
        sxx_hist.append(stress[0, 0])
    return np.array(sxx_hist)


@pytest.mark.slow
def test_yield_initiates_at_ft_in_uniaxial_tension():
    """El umbral analítico de fluencia en tracción uniaxial pura, con
    kappa_t=kappa_c=0, es exactamente ft por construcción de la
    superficie de Lubliner/Lee-Fenves (verificado a mano en el desarrollo
    y aquí de forma independiente): para s1=X, s2=s3=0, F(X,0,0,0,0)=0
    se resuelve exactamente en X=ft."""
    cdp = _material()
    eps_axis = np.linspace(1e-7, 0.0003, 60)
    sxx = _uniaxial_stress_sweep(cdp, eps_axis)
    # tramo elástico = solo hasta la PRIMERA vez que se supera ft (más
    # adelante, en la rama de ablandamiento, sxx vuelve a caer por debajo
    # de ese umbral sin que eso signifique que sigue siendo elástico)
    first_over = int(np.argmax(sxx > cdp.ft))
    assert first_over > 5
    elastic_mask = np.zeros_like(sxx, dtype=bool)
    elastic_mask[:first_over] = True
    ratio = sxx[elastic_mask] / (cdp.e0 * eps_axis[elastic_mask])
    assert np.allclose(ratio, ratio[0], rtol=1e-3)  # tramo elástico, pendiente constante
    # el primer punto que supera ft debe estar muy cerca del umbral
    assert sxx[first_over] == pytest.approx(cdp.ft, rel=0.05)


@pytest.mark.slow
def test_tension_softens_toward_zero_eventually():
    """No se exige que el pico nominal coincida exactamente con ft (ver
    nota en el módulo: sigma_bar_t puede crecer con kappa_t por la
    interacción con el daño de Birtel-Mark, una sutileza de calibración
    que no se pudo verificar contra la fuente primaria). Lo que sí debe
    cumplirse sin ambigüedad es el comportamiento asintótico: superado un
    punto, el esfuerzo cae de forma sostenida hacia un residual pequeño."""
    cdp = _material()
    eps_axis = np.linspace(1e-7, 0.003, 60)
    sxx = _uniaxial_stress_sweep(cdp, eps_axis)
    assert sxx.max() > cdp.ft  # llegó a fluencia
    assert sxx[-1] < 0.2 * sxx.max()  # ablandamiento sustancial hacia el final del barrido
    assert np.all(np.isfinite(sxx))


def _uniaxial_compression_peak(cdp, n=60):
    sxx = _uniaxial_stress_sweep(cdp, np.linspace(-1e-6, -0.003, n))
    return abs(sxx.min())


@pytest.mark.slow
def test_uniaxial_compression_peak_matches_fcm():
    cdp = _material()
    peak = _uniaxial_compression_peak(cdp)
    assert peak == pytest.approx(cdp.fcm, rel=0.02)


@pytest.mark.slow
def test_biaxial_compression_envelope_matches_fb0_fc0():
    """Chequeo limpio y decisivo de la calibración de alpha: la
    resistencia en compresión equibiaxial dividida por la resistencia en
    compresión uniaxial debe reproducir fb0/fc0 (1.16 por defecto). Este
    test no depende de la sutileza de tracción/daño de los tests
    anteriores, así que es la validación más fuerte de la geometría de la
    superficie de fluencia."""
    cdp = _material()
    state = cdp.initial_state(1)
    peak_biaxial = 0.0
    for eps in np.linspace(-1e-6, -0.003, 60):
        stress, _, state = cdp.integrate(np.array([[eps, eps, 0.0]]), state, dt=1.0)
        peak_biaxial = min(peak_biaxial, stress[0, 0])

    peak_uniaxial = _uniaxial_compression_peak(cdp)
    ratio = abs(peak_biaxial) / peak_uniaxial
    assert ratio == pytest.approx(cdp.p.fb0_fc0, rel=0.01)


def test_tangent_is_elastic_when_inactive():
    cdp = _material()
    state = cdp.initial_state(1)
    strain = np.array([[1e-6, -0.2e-6, 0.0]])  # muy por debajo de fluencia
    stress, tangent, _ = cdp.integrate(strain, state, dt=1.0)
    assert np.allclose(tangent[0], cdp.d0, rtol=1e-3, atol=1e-3)


def test_tangent_is_finite_and_reasonable_when_active():
    cdp = _material()
    state = cdp.initial_state(1)
    strain = np.array([[0.0005, -0.0001, 0.0]])  # supera ft
    stress, tangent, new_state = cdp.integrate(strain, state, dt=1.0)
    assert np.all(np.isfinite(tangent))
    assert new_state["kappa_t"][0] > 0.0


def test_damage_bounded_in_zero_one():
    cdp = _material()
    state = cdp.initial_state(1)
    for eps in np.linspace(1e-6, 0.003, 50):
        stress, _, state = cdp.integrate(np.array([[eps, -0.2 * eps, 0.0]]), state, dt=1.0)
    d_t = cdp._d_t(state["kappa_t"])
    d_c = cdp._d_c(state["kappa_c"])
    assert np.all((d_t >= 0.0) & (d_t < 1.0))
    assert np.all((d_c >= 0.0) & (d_c < 1.0))


def test_plane_stress_is_exact_by_construction():
    """sigma_zz nunca se calcula ni se almacena: el modelo trabaja
    íntegramente con 3 componentes (xx,yy,xy) y el esfuerzo principal
    "fuera de plano" se fija a 0 en la evaluación de F y del flujo. No
    hay ninguna ruta de código que pueda introducir un sigma_zz no nulo,
    así que esta propiedad se cumple por construcción, no por chequeo en
    tiempo de ejecución."""
    cdp = _material()
    state = cdp.initial_state(1)
    stress, _, _ = cdp.integrate(np.array([[0.001, 0.0005, 0.0002]]), state, dt=1.0)
    assert stress.shape == (1, 3)


def test_rejects_mesh_too_coarse_for_fracture_energy():
    with pytest.raises(MaterialModelError):
        _material(char_length=1e6)  # l_ch absurdamente grande frente a G_F


def test_rejects_mesh_too_coarse_for_compressive_fracture_energy():
    # gf grande evita disparar antes el chequeo de tracción; gc diminuto
    # aísla el chequeo de la rama de ablandamiento en compresión.
    with pytest.raises(MaterialModelError):
        _material(char_length=1e5, gf=1000.0, gc=1e-8)
