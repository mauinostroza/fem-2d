"""Sesión S7.3: `solver.arc_length.arc_length_solve` (Crisfield, forma
cilíndrica por defecto — ver docstring del módulo).

Caso de validación: el mismo elemento Q4 único con `ConcreteCDP` en
tracción de `test_concrete_q4_integration.py`, pero controlado por carga
(no por desplazamiento) — exactamente el escenario en el que
`LoadControl` puro diverge en el pico y por el que se justifica
arc-length.
"""

import numpy as np
import pytest

from fem2d_nl.bc import dof_index, fixed_dofs_from_nodes
from fem2d_nl.exceptions import ConvergenceError
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.model import ElementGroup, Model
from fem2d_nl.solver.arc_length import ArcLengthOptions, arc_length_solve
from fem2d_nl.solver.controls import ArcLengthControl, DisplacementControl, LoadControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve

LENGTH = 100.0
HEIGHT = 100.0
FCK = 25.0


def _single_element_model(l_ch=LENGTH):
    nodes = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    quads = np.array([[0, 1, 2, 3]])
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK)
    cdp = ConcreteCDP(params, char_length=l_ch)
    group = ElementGroup(kind="quad4", connectivity=quads, material=cdp, elem_kwargs={"thickness": 1.0})
    return Model(nodes=nodes, groups=[group]), cdp


def _supports():
    fixed = fixed_dofs_from_nodes([0, 3], components=(0,))
    fixed = np.union1d(fixed, fixed_dofs_from_nodes([0, 1], components=(1,)))
    return fixed


def _reference_force_pattern(cdp):
    """Patrón de referencia: tracción uniforme ~ft repartida entre los 2
    nodos de la cara cargada (equivalente nodal consistente de una
    tracción uniforme), escalada un 40% por encima de ft para asegurar
    que λ=1 ya esté más allá del pico real (que en tracción del CDP
    ocurre por encima de ft, ver aviso de honestidad de `concrete_cdp.py`)."""
    total = 1.4 * cdp.ft * HEIGHT * 1.0
    pattern = np.zeros(8)
    pattern[dof_index(1, 0)] = total / 2.0
    pattern[dof_index(2, 0)] = total / 2.0
    return pattern


def test_load_control_alone_fails_to_reach_lambda_1_past_the_peak():
    """Confirma el motivo de ser de arc-length: control de carga puro NO
    logra converger hasta λ=1 en este caso (el patrón de referencia ya
    pasa el pico), a diferencia de arc-length en el siguiente test.
    Presupuesto de iteraciones/cutbacks chico a propósito: alcanza para
    demostrar la divergencia sin agotar minutos reintentando cerca del
    punto límite (donde la tangente es casi singular por diseño)."""
    model, cdp = _single_element_model()
    control = LoadControl(fixed_dofs=_supports(), force_pattern=_reference_force_pattern(cdp))
    result = nonlinear_solve(model, control, SolverOptions(n_steps=20, max_iterations=15, max_cutbacks=3))
    assert not result.converged
    assert result.steps  # sí debe conservar los pasos previos al fallo


def test_arc_length_traces_past_the_peak_and_softens():
    model, cdp = _single_element_model()
    control = ArcLengthControl(fixed_dofs=_supports(), force_pattern=_reference_force_pattern(cdp))
    result = arc_length_solve(
        model, control,
        ArcLengthOptions(n_steps=25, max_iterations=40, max_arc_cutbacks=10, max_total_steps=80),
    )
    assert result.steps

    lam_hist = np.array([s.lam for s in result.steps])
    ux_hist = np.array([s.u[dof_index(1, 0)] for s in result.steps])

    # el desplazamiento debe crecer de forma prácticamente monótona (mismo
    # sentido de carga) aunque λ dejará de hacerlo tras el pico
    assert ux_hist[-1] > ux_hist[0]
    assert np.all(np.diff(ux_hist) > -1e-9)

    # la firma de haber pasado el pico: λ sube y luego baja (rama de
    # ablandamiento con carga decreciente a desplazamiento creciente) —
    # justo lo que LoadControl puro no puede trazar (test anterior)
    assert lam_hist.max() > lam_hist[-1]
    assert lam_hist.max() > 0.5  # llegó a una fracción sustancial del pico, no se cortó de entrada


def test_arc_length_matches_displacement_control_pre_peak():
    """En la rama ascendente (antes del pico), arc-length y control de
    desplazamiento deben trazar la MISMA curva fuerza-desplazamiento
    (es la misma física, dos formas de recorrer la misma curva de
    equilibrio) — chequeo cruzado de correctitud, no solo de robustez."""
    model_d, cdp = _single_element_model()
    fixed = _supports()
    prescribed = np.array([dof_index(1, 0), dof_index(2, 0)])
    eps_small = 5e-5  # bien en la rama elástica/inicio de ablandamiento suave, sin complicaciones
    control_d = DisplacementControl(
        fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=np.full(2, eps_small * LENGTH)
    )
    result_d = nonlinear_solve(model_d, control_d, SolverOptions(n_steps=10, max_iterations=40))
    assert result_d.converged
    force_d = result_d.steps[-1].f_int[prescribed].sum()
    ux_d = result_d.steps[-1].u[dof_index(1, 0)]

    model_a, cdp_a = _single_element_model()
    control_a = ArcLengthControl(fixed_dofs=fixed, force_pattern=_reference_force_pattern(cdp_a))
    result_a = arc_length_solve(
        model_a, control_a, ArcLengthOptions(n_steps=40, max_iterations=40, max_total_steps=80)
    )
    assert result_a.steps

    ux_a = np.array([s.u[dof_index(1, 0)] for s in result_a.steps])
    force_a = np.array([s.lam * cdp_a.ft * HEIGHT * 1.4 for s in result_a.steps])  # fuerza total = lam * total del patrón

    idx = np.argmin(np.abs(ux_a - ux_d))
    assert force_a[idx] == pytest.approx(force_d, rel=0.1)
