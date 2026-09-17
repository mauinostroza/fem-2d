"""Test estructural (nivel 1): una única barra truss con acero
multilineal, desplazamiento prescrito en el extremo libre. La fuerza de
reacción debe coincidir con área * sigma(eps) calculado directamente del
material (dos rutas de código independientes: FEM completo vs. llamada
directa al material)."""

import numpy as np
import pytest

from fem2d_nl.bc import dof_index
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel
from fem2d_nl.model import ElementGroup, Model
from fem2d_nl.solver.controls import DisplacementControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve

POINTS = np.array([[0.002, 400.0], [0.02, 450.0], [0.08, 500.0]])
LENGTH = 100.0
AREA = 50.0


def _solve_bar(eps_target: float):
    nodes = np.array([[0.0, 0.0], [LENGTH, 0.0]])
    connectivity = np.array([[0, 1]])
    material = MultilinearSteel(points=POINTS)
    group = ElementGroup(kind="truss2", connectivity=connectivity, material=material, elem_kwargs={"area": AREA})
    model = Model(nodes=nodes, groups=[group])

    fixed = np.array([dof_index(0, 0), dof_index(0, 1), dof_index(1, 1)])
    prescribed = np.array([dof_index(1, 0)])
    control = DisplacementControl(
        fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=np.array([eps_target * LENGTH])
    )
    result = nonlinear_solve(model, control, SolverOptions(n_steps=5))
    assert result.converged
    u = result.steps[-1].u

    from fem2d_nl.assembly import assemble

    f_int, _, _ = assemble(model, u, result.steps[-1].states, dt=1.0)
    reaction = f_int[dof_index(1, 0)]
    return reaction, material


@pytest.mark.parametrize("eps_target", [0.001, 0.002, 0.01, 0.05])
def test_truss_reaction_matches_material_directly(eps_target):
    reaction, material = _solve_bar(eps_target)
    sigma_expected, _, _ = material.integrate(np.array([eps_target]), material.initial_state(1), dt=1.0)
    assert reaction == pytest.approx(AREA * sigma_expected[0], rel=1e-4)
