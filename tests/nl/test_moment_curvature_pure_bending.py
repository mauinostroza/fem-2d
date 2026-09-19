"""Sesión S7.2: `postprocess.moment_curvature_pure_bending` — `y_na` por
bisección en cada paso de curvatura, para forzar `N≈0` en vez del `y_na`
fijo de `moment_curvature` (ver el aviso de honestidad de `postprocess.py`).

Aviso de honestidad técnica: es notablemente cara — cada paso de
curvatura implica resolver el problema no lineal completo varias veces
(bisección anidada). Medido en esta sesión: ~120s por paso de curvatura
para una sección con armadura de 12 elementos (`nx=3,ny=4`), con
`max_bisection=12`. Por eso los tests aquí usan mallas/pasos mínimos, y
`test_pure_bending_migrates_neutral_axis_with_cracking` no exige `N≈0`
exacto sino una cota floja (la excentricidad `N·altura/M` debe ser
pequeña, no necesariamente nula) — con el presupuesto de bisección usado
en el test, la convergencia de `N` no siempre llega al `tol_axial`
calculado antes de agotar `max_bisection`, un límite de costo/precisión
real, no un error de la búsqueda (que sí migra `y_na` de forma monótona y
físicamente sensata, ver el test)."""

import numpy as np
import pytest

from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel
from fem2d_nl.mesh.section import RebarLayer, SectionGeometry
from fem2d_nl.mesh.structured import build_section_mesh, nominal_char_length
from fem2d_nl.postprocess import moment_curvature_pure_bending, transformed_elastic_centroid
from fem2d_nl.solver.newton import SolverOptions

pytestmark = pytest.mark.slow


def test_pure_bending_matches_fixed_y_na_in_elastic_symmetric_case():
    """Sin armadura (sección simétrica), en régimen elástico: la
    bisección debe encontrar `y_na≈height/2` (el mismo valor que usa
    `moment_curvature` con centroide fijo) y dar N≈0 con precisión, ya
    que ahí la búsqueda converge rápido (pocas iteraciones de
    bisección bastan por la linealidad del problema)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=3, ny=4)
    l_ch = nominal_char_length(geom)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    cdp = ConcreteCDP(params, char_length=l_ch)
    model = build_section_mesh(geom, [], cdp, bond_mode="perfect")

    kappa_max = 2e-7  # elástico, bien por debajo de la fisuración
    res = moment_curvature_pure_bending(
        model, geom.span, kappa_max, n_steps=2, options=SolverOptions(n_steps=1, max_iterations=30)
    )
    assert res.converged
    assert res.y_na_history is not None
    assert np.allclose(res.y_na_history, geom.height / 2.0, atol=0.5)
    assert np.allclose(res.axial, 0.0, atol=1e-3)


def test_pure_bending_migrates_neutral_axis_with_cracking():
    """Con armadura, empujado más allá de la fisuración: `y_na` debe
    migrar de forma MONÓTONA desde el centroide elástico hacia la zona de
    compresión (y creciente, ya que la armadura de este test está cerca
    de la fibra inferior en tracción) a medida que crece la curvatura, y
    la excentricidad resultante (N·altura/M) debe ser pequeña (mucho
    menor que la que se ve con `y_na` fijo, aunque no exactamente cero
    dentro del presupuesto de bisección usado — ver aviso del módulo)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=3, ny=4)
    l_ch = nominal_char_length(geom)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    cdp = ConcreteCDP(params, char_length=l_ch)
    steel = MultilinearSteel(points=np.array([[0.002, 400.0], [0.05, 460.0]]))
    layer = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=steel)
    model = build_section_mesh(geom, [layer], cdp, bond_mode="perfect")
    y_na_elastic = transformed_elastic_centroid(geom, [layer], cdp.e0)

    kappa_max = 4e-6  # bien pasada la fisuración
    res = moment_curvature_pure_bending(
        model, geom.span, kappa_max, n_steps=2, y_na_init=y_na_elastic, max_bisection=12,
        options=SolverOptions(n_steps=1, max_iterations=30, max_cutbacks=6),
    )
    assert res.converged
    assert res.y_na_history is not None
    assert np.all(np.diff(res.y_na_history) > 0.0)  # migra monótonamente
    assert res.y_na_history[0] > y_na_elastic  # ya se corrió respecto del centroide elástico

    eccentricity_fraction = np.abs(res.axial) * geom.height / np.abs(res.moment)
    assert np.all(eccentricity_fraction < 0.05)
