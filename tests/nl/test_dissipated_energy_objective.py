"""Cierre del ítem abierto de la sesión S4: un test de objetividad de malla
de la regularización crack-band que SÍ es correcto.

S4 intentó comparar la energía disipada entre mallas a nivel de FEM
completo (área bajo la curva carga-desplazamiento GLOBAL) y lo descartó
porque esa área incluye energía elástica recuperable del resto de la
estructura, no solo la disipada por la fisura. La cantidad correcta es la
densidad de energía disipada POR PUNTO MATERIAL (`ConcreteCDP.
dissipated_tensile_energy_density`, construida a partir de la propia tabla
`sigma_bar_t_table`/`d_t_table` que ya usa el material) — al llegar a
saturación completa de la rama de ablandamiento, esa integral vale
exactamente `G_F/l_ch` para CUALQUIER `l_ch`, sin ambigüedad de
localización ni necesidad de FEM. Ver el docstring del método en
`concrete_cdp.py` para la deducción.
"""

import numpy as np
import pytest

from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP


def _cdp(l_ch, fck=25.0):
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=fck)
    return ConcreteCDP(params, char_length=l_ch)


@pytest.mark.parametrize("l_ch", [10.0, 25.0, 50.0, 100.0, 200.0])
def test_tensile_dissipated_energy_density_times_l_ch_equals_gf(l_ch):
    # tolerancia dominada por la discretización de la tabla interna de
    # `build_tension_law` (n=24 puntos por defecto, no por el propio
    # método de integración): verificado por separado que con una tabla
    # más fina (n=400) el error baja de ~0.4% a ~1e-3%.
    cdp = _cdp(l_ch)
    kappa_ult = cdp.kappa_t[-1]  # saturación completa de la rama de ablandamiento
    e_diss = cdp.dissipated_tensile_energy_density(kappa_ult)
    assert e_diss * l_ch == pytest.approx(cdp.gf, rel=0.01)


def test_tensile_dissipated_energy_density_is_monotonic_in_kappa():
    cdp = _cdp(l_ch=50.0)
    kappas = np.linspace(0.0, cdp.kappa_t[-1], 30)
    e_diss = cdp.dissipated_tensile_energy_density(kappas)
    assert np.all(np.diff(e_diss) >= -1e-12)
    assert e_diss[0] == pytest.approx(0.0, abs=1e-10)


def test_tensile_dissipated_energy_density_independent_of_l_ch_at_matched_fraction():
    """La misma FRACCIÓN de kappa_ult (mismo estado físico de daño relativo)
    debe disipar, por unidad de volumen, una densidad de energía cuyo
    producto por l_ch es el mismo `G_F`, para dos mallas de tamaño
    distinto — esta es la propiedad de objetividad de malla en sí misma,
    la que el intento fallido de S4 no pudo verificar de forma limpia."""
    fracs = [0.25, 0.5, 0.75, 1.0]
    l_ch_values = [15.0, 40.0, 90.0]
    results = {}
    for l_ch in l_ch_values:
        cdp = _cdp(l_ch)
        kappas = np.array(fracs) * cdp.kappa_t[-1]
        results[l_ch] = cdp.dissipated_tensile_energy_density(kappas) * l_ch

    baseline = results[l_ch_values[0]]
    for l_ch in l_ch_values[1:]:
        assert results[l_ch] == pytest.approx(baseline, rel=5e-3)


def test_compressive_dissipated_energy_density_is_monotonic_and_finite():
    """La compresión no tiene la propiedad exacta de `=G_c/l_ch` (ver
    docstring), pero sí debe ser una función monótona y finita de kappa_c —
    chequeo de consistencia mínimo, no de objetividad exacta."""
    cdp = _cdp(l_ch=50.0)
    kappas = np.linspace(0.0, cdp.kappa_c[-1], 30)
    e_diss = cdp.dissipated_compressive_energy_density(kappas)
    assert np.all(np.isfinite(e_diss))
    assert np.all(np.diff(e_diss) >= -1e-10)
