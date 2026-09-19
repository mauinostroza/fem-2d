"""Sesión S5, nivel estructural: `postprocess.moment_curvature` sobre una
franja construida con `mesh.structured.build_section_mesh`.

**Actualización S7 (retractación de un hallazgo de S5)**: S5 había
reportado que el modo `bond_slip` mostraba una sensibilidad sospechosa al
número de pasos de carga (~40-50% de diferencia en curvaturas intermedias
entre 10 y 15-30 pasos), y por eso se descartó la comparación cuantitativa
de `M_u` planeada originalmente. En S7 se investigó ese hallazgo con un
script de diagnóstico limpio (barrido sistemático de `n_steps` =
10/20/40/80 sobre el mismo caso, ver
`tests/nl/test_bond_slip_step_convergence.py`) y **no se reprodujo**: el
momento final coincide dentro de ~0.2% en las cuatro resoluciones. La
inspección del estado de los `bond_link` confirma que el mecanismo de
adherencia funciona correctamente (`s_max` no nulo y físicamente
razonable en los puntos interiores de la barra, exactamente nulo en los
dos extremos por la propia condición de borde ahí). La conclusión es que
el hallazgo de S5 fue casi con certeza un artefacto del script de
depuración ad-hoc usado en ese momento (no reconstruible ni reproducible
con el código de producción), no un defecto real de `moment_curvature`/
`build_section_mesh`/`bond_link`. Se agregó un test de regresión
(`test_bond_slip_step_convergence.py`) para dejar esto asentado. La
comparación cuantitativa perfecto-vs-`bond_slip` de `M_u` sigue sin
implementarse en este archivo (no era el foco de S7), pero ya no hay una
duda de correctitud abierta que lo justifique descartar.
"""

import numpy as np
import pytest

from fem2d_nl.materials.bond_mc2010 import BondLaw, BondParameters
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel
from fem2d_nl.mesh.section import RebarLayer, SectionGeometry
from fem2d_nl.mesh.structured import build_section_mesh, nominal_char_length
from fem2d_nl.postprocess import moment_curvature, transformed_elastic_centroid
from fem2d_nl.solver.newton import SolverOptions

pytestmark = pytest.mark.slow


def _steel():
    return MultilinearSteel(points=np.array([[0.002, 400.0], [0.05, 460.0]]))


def test_elastic_moment_curvature_matches_transformed_beam_theory():
    """Curvatura bien por debajo de la deformación de fisuración (sin
    armadura, para aislar el chequeo del propio FEM de flexión): M debe
    coincidir con la teoría de vigas (`M=E*I*kappa`) dentro de una
    tolerancia que refleja el "shear locking" conocido de elementos Q4
    bilineales bajo flexión pura con pocos elementos a lo largo del tramo
    (verificado por separado: el error baja de ~2.7% a ~0.1% al refinar
    `nx` de 2 a 16 — es un artefacto de discretización bien documentado en
    la literatura de FEM, no un error de signo/BC)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=6, ny=8)
    l_ch = nominal_char_length(geom)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    cdp = ConcreteCDP(params, char_length=l_ch)
    model = build_section_mesh(geom, [], cdp, bond_mode="perfect")

    y_na = geom.height / 2.0
    kappa_max = 2e-7  # deformación de fibra extrema = 4e-5, bien por debajo de ft/E0 ~ 8.5e-5
    res = moment_curvature(
        model, geom.span, y_na, kappa_max, n_steps=4,
        options=SolverOptions(n_steps=4, max_iterations=30),
    )
    assert res.converged
    assert np.allclose(res.axial, 0.0, atol=1e-3)  # sin armadura, y_na exacto: N=0 por simetría

    i_gross = geom.width * geom.height**3 / 12.0
    ei = cdp.e0 * i_gross
    assert res.moment[-1] == pytest.approx(ei * res.kappa[-1], rel=0.02)


def test_perfect_bond_softens_after_cracking():
    """Con armadura, empujado bien más allá de la deformación de
    fisuración: la rigidez secante final (M/kappa) debe ser sustancialmente
    menor que la rigidez elástica de la sección transformada no fisurada —
    evidencia de que el modelo efectivamente fisura y redistribuye
    esfuerzos, no un chequeo cuantitativo contra una fórmula de M_cr
    (demasiado sensible al `y_na` fijo para ser confiable, ver aviso del
    módulo `postprocess`)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=3, ny=4)
    l_ch = nominal_char_length(geom)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    cdp = ConcreteCDP(params, char_length=l_ch)
    layer = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=_steel())
    model = build_section_mesh(geom, [layer], cdp, bond_mode="perfect")
    y_na = transformed_elastic_centroid(geom, [layer], cdp.e0)

    kappa_max = 1.2e-5
    res = moment_curvature(
        model, geom.span, y_na, kappa_max, n_steps=10,
        options=SolverOptions(n_steps=10, max_iterations=30, max_cutbacks=6),
    )
    assert res.converged
    assert np.all(np.diff(res.moment) > 0.0)  # momento monótono creciente

    i_gross = geom.width * geom.height**3 / 12.0
    n_ratio = _steel().e_resolved / cdp.e0
    i_steel = (n_ratio - 1.0) * layer.area * (layer.depth_y - y_na) ** 2
    ei_elastic = cdp.e0 * (i_gross + i_steel)
    secant_final = res.moment[-1] / res.kappa[-1]
    assert secant_final < 0.6 * ei_elastic


def test_progress_cb_and_final_states_are_populated():
    """S6: `moment_curvature` expone `progress_cb` (para la barra de
    progreso de la UI) y `final_states` (para el contorno de daño)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=2, ny=4)
    l_ch = nominal_char_length(geom)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    cdp = ConcreteCDP(params, char_length=l_ch)
    model = build_section_mesh(geom, [], cdp, bond_mode="perfect")

    calls = []
    res = moment_curvature(
        model, geom.span, geom.height / 2.0, kappa_max=2e-7, n_steps=3,
        options=SolverOptions(n_steps=3, max_iterations=30),
        progress_cb=lambda *args: calls.append(args),
    )
    assert res.converged
    assert len(calls) == 3
    assert res.final_states is not None
    assert len(res.final_states) == len(model.groups)
    assert res.final_states[0]["kappa_t"].shape[0] == model.groups[0].n_gauss_total


def test_bond_slip_mode_runs_and_is_monotonic():
    """Chequeo modesto y robusto del camino `bond_slip` a nivel de malla
    completa (nodos de acero duplicados + `bond_link`): converge y da un
    momento monótono creciente. No se compara cuantitativamente contra el
    modo de adherencia perfecta (ver aviso de honestidad del módulo)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=3, ny=4)
    l_ch = nominal_char_length(geom)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    cdp = ConcreteCDP(params, char_length=l_ch)
    layer = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=_steel())
    bond_law = BondLaw(BondParameters(fcm=cdp.fcm, bar_diameter=16.0, bond_condition="good"))
    model = build_section_mesh(geom, [layer], cdp, bond_mode="bond_slip", bond_params=bond_law)
    y_na = transformed_elastic_centroid(geom, [layer], cdp.e0)

    kappa_max = 1.2e-5
    res = moment_curvature(
        model, geom.span, y_na, kappa_max, n_steps=10,
        options=SolverOptions(n_steps=10, max_iterations=30, max_cutbacks=6),
    )
    assert res.converged
    assert np.all(np.diff(res.moment) > 0.0)
    assert res.moment[-1] > 0.0
