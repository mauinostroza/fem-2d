"""Tests a nivel de punto material (sin FEM) para MultilinearSteel."""

import numpy as np
import pytest

from fem2d_nl.exceptions import MaterialModelError
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel

POINTS = np.array(
    [
        [0.002, 400.0],
        [0.02, 450.0],
        [0.08, 500.0],
    ]
)


def test_monotonic_loading_reproduces_table_points_exactly():
    steel = MultilinearSteel(points=POINTS)
    for eps_i, sigma_i in POINTS:
        state = steel.initial_state(1)
        sigma, tangent, _ = steel.integrate(np.array([eps_i]), state, dt=1.0)
        assert sigma[0] == pytest.approx(sigma_i, rel=1e-8)


def test_elastic_branch_matches_young_modulus():
    steel = MultilinearSteel(points=POINTS)
    state = steel.initial_state(1)
    eps = 0.001  # por debajo de la fluencia inicial (0.002)
    sigma, tangent, _ = steel.integrate(np.array([eps]), state, dt=1.0)
    assert sigma[0] == pytest.approx(200_000.0 * eps, rel=1e-8)
    assert tangent[0] == pytest.approx(200_000.0, rel=1e-8)


def test_unload_is_elastic_then_reyields_symmetrically():
    steel = MultilinearSteel(points=POINTS)
    state = steel.initial_state(1)
    # cargar hasta el segundo punto de la tabla
    sigma1, _, state = steel.integrate(np.array([0.02]), state, dt=1.0)
    assert sigma1[0] == pytest.approx(450.0, rel=1e-8)

    # descarga pequeña: debe ser puramente elástica (pendiente E)
    sigma2, tangent2, state2 = steel.integrate(np.array([0.019]), state, dt=1.0)
    assert sigma2[0] == pytest.approx(450.0 - 200_000.0 * 0.001, rel=1e-6)
    assert tangent2[0] == pytest.approx(200_000.0, rel=1e-6)
    # la descarga elástica no debe alterar la deformación plástica acumulada
    assert state2["kappa"][0] == pytest.approx(state["kappa"][0], rel=1e-8)

    # descarga grande: debe re-plastificar en compresión, simétrico en kappa
    sigma3, _, state3 = steel.integrate(np.array([-0.02]), state, dt=1.0)
    assert sigma3[0] < 0.0
    assert abs(sigma3[0]) >= 400.0  # ya fluyó en compresión
    assert state3["kappa"][0] > state["kappa"][0]


def test_rejects_non_increasing_strain_table():
    bad_points = np.array([[0.01, 400.0], [0.005, 450.0]])
    with pytest.raises(MaterialModelError):
        MultilinearSteel(points=bad_points)
