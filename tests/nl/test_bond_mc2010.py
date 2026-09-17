"""Tests a nivel de punto material (sin FEM) para la ley de adherencia."""

import numpy as np
import pytest

from fem2d_nl.exceptions import MaterialModelError
from fem2d_nl.materials.bond_mc2010 import BondLaw, BondParameters

PARAMS = BondParameters(fcm=30.0, bar_diameter=16.0, bond_condition="good")


def test_four_branches_reproduce_envelope():
    law = BondLaw(PARAMS)
    state = law.initial_state(1)

    # dentro de la rama potencial
    s_mid = law.s1 / 2
    tau, _, state = law.integrate(np.array([s_mid]), state, dt=1.0)
    assert tau[0] == pytest.approx(law.tau_max * (s_mid / law.s1) ** law.alpha, rel=1e-6)

    # meseta: tau=tau_max en cualquier punto de [s1,s2]
    state2 = law.initial_state(1)
    tau_plateau, slope_plateau, _ = law.integrate(np.array([(law.s1 + law.s2) / 2]), state2, dt=1.0)
    assert tau_plateau[0] == pytest.approx(law.tau_max, rel=1e-6)

    # rama descendente
    state3 = law.initial_state(1)
    s_desc = (law.s2 + law.s3) / 2
    tau_desc, _, _ = law.integrate(np.array([s_desc]), state3, dt=1.0)
    expected = law.tau_max - (law.tau_max - law.tau_f) * (s_desc - law.s2) / (law.s3 - law.s2)
    assert tau_desc[0] == pytest.approx(expected, rel=1e-6)

    # residual
    state4 = law.initial_state(1)
    tau_res, _, _ = law.integrate(np.array([law.s3 * 2]), state4, dt=1.0)
    assert tau_res[0] == pytest.approx(law.tau_f, rel=1e-6)


def test_tangent_matches_finite_difference():
    """La tangente analítica debe coincidir con diferencias finitas en el
    interior de cada rama. En los quiebres entre ramas (s_e, s1, s2, s3) la
    pendiente es genuinamente discontinua, así que se muestrean puntos
    estrictamente interiores a cada tramo, no puntos aleatorios que podrían
    caer arbitrariamente cerca de un quiebre."""
    law = BondLaw(PARAMS)
    interior_points = [
        (law.s_e + law.s1) / 2,
        (law.s1 + law.s2) / 2,
        (law.s2 + law.s3) / 2,
        law.s3 * 1.5,
    ]
    for s0 in interior_points:
        h = 1e-6 * max(s0, 1.0)
        tau0, dtau_ds, _ = law.integrate(np.array([s0]), law.initial_state(1), dt=1.0)
        tau_plus, _, _ = law.integrate(np.array([s0 + h]), law.initial_state(1), dt=1.0)
        tau_minus, _, _ = law.integrate(np.array([s0 - h]), law.initial_state(1), dt=1.0)
        fd = (tau_plus[0] - tau_minus[0]) / (2 * h)
        assert dtau_ds[0] == pytest.approx(fd, rel=1e-3, abs=1e-3)


def test_unloading_uses_secant_with_memory():
    law = BondLaw(PARAMS)
    state = law.initial_state(1)
    s_max = law.s1
    tau_max_reached, _, state = law.integrate(np.array([s_max]), state, dt=1.0)

    # descargar a la mitad del deslizamiento máximo alcanzado
    tau_unload, k_unload, state2 = law.integrate(np.array([s_max / 2]), state, dt=1.0)
    expected_k_sec = tau_max_reached[0] / s_max
    assert tau_unload[0] == pytest.approx(expected_k_sec * (s_max / 2), rel=1e-6)
    assert state2["s_max"][0] == pytest.approx(s_max, rel=1e-8)  # no retrocede


def test_odd_symmetry_for_negative_slip():
    law = BondLaw(PARAMS)
    tau_pos, _, _ = law.integrate(np.array([law.s1 / 2]), law.initial_state(1), dt=1.0)
    tau_neg, _, _ = law.integrate(np.array([-law.s1 / 2]), law.initial_state(1), dt=1.0)
    assert tau_neg[0] == pytest.approx(-tau_pos[0], rel=1e-8)


def test_splitting_failure_mode_rejected_explicitly():
    with pytest.raises(MaterialModelError):
        BondParameters(fcm=30.0, bar_diameter=16.0, failure_mode="splitting").resolve()
