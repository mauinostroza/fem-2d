"""Nivel 2: ensayo de arrancamiento (pull-out) completo con elementos
`truss2` + `bond_link`, comparado contra la solución de la EDO 1D de
adherencia (resuelta de forma independiente con `scipy.integrate.solve_bvp`
usando la misma ley τ-s). El hormigón se modela rígido (todos sus nodos
fijos) para aislar la no linealidad de la adherencia: el objetivo es
validar el elemento de interfaz y su ensamblaje, no el continuo.
"""

import numpy as np
import pytest
from scipy.integrate import solve_bvp

from fem2d_nl.bc import dof_index
from fem2d_nl.materials.bond_mc2010 import BondLaw, BondParameters
from fem2d_nl.materials.elastic import ElasticUniaxial
from fem2d_nl.model import ElementGroup, Model
from fem2d_nl.solver.controls import DisplacementControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve

BAR_DIAMETER = 16.0
AREA_STEEL = np.pi * BAR_DIAMETER**2 / 4
PERIMETER = np.pi * BAR_DIAMETER
E_STEEL = 200_000.0
EMBED_LENGTH = 150.0
N_ELEM = 30


def _bond_params():
    return BondParameters(fcm=30.0, bar_diameter=BAR_DIAMETER, bond_condition="good")


def _solve_ode_reference(s0: float):
    law = BondLaw(_bond_params())

    def rhs(x, y):
        s, sp = y
        tau, _, _ = law.integrate(s, law.initial_state(s.shape[0]), dt=1.0)
        return np.vstack([sp, (PERIMETER / (AREA_STEEL * E_STEEL)) * tau])

    def bc(ya, yb):
        return np.array([ya[0] - s0, yb[1]])

    x = np.linspace(0.0, EMBED_LENGTH, 200)
    y0 = np.zeros((2, x.size))
    y0[0] = s0 * (1.0 - x / EMBED_LENGTH)
    sol = solve_bvp(rhs, bc, x, y0, max_nodes=20000, tol=1e-8)
    assert sol.success
    # El signo de sol.y[1,0] depende de la convención de dirección de x
    # elegida para la EDO (independiente de la convención de nodos/dofs
    # del FEM); lo que importa físicamente es la magnitud de la fuerza.
    p0 = abs(AREA_STEEL * E_STEEL * sol.y[1, 0])
    return p0


def _solve_fem_pullout(s0: float):
    n_steel_nodes = N_ELEM + 1
    xs = np.linspace(0.0, EMBED_LENGTH, n_steel_nodes)
    steel_nodes = np.column_stack([xs, np.zeros_like(xs)])
    concrete_nodes = steel_nodes.copy()
    nodes = np.vstack([steel_nodes, concrete_nodes])
    concrete_offset = n_steel_nodes

    truss_conn = np.array([[i, i + 1] for i in range(N_ELEM)])
    steel_material = ElasticUniaxial(young_modulus=E_STEEL)
    truss_group = ElementGroup(
        kind="truss2", connectivity=truss_conn, material=steel_material, elem_kwargs={"area": AREA_STEEL}
    )

    link_conn = np.array([[i, concrete_offset + i] for i in range(n_steel_nodes)])
    elem_len = EMBED_LENGTH / N_ELEM
    trib_length = np.full(n_steel_nodes, elem_len)
    trib_length[0] = elem_len / 2
    trib_length[-1] = elem_len / 2
    bond_material = BondLaw(_bond_params())
    link_group = ElementGroup(
        kind="bond_link",
        connectivity=link_conn,
        material=bond_material,
        elem_kwargs={
            "perimeter": PERIMETER,
            "trib_length": trib_length,
            "k_normal": 1e6,
            "axis": (1.0, 0.0),
        },
    )

    model = Model(nodes=nodes, groups=[truss_group, link_group])

    concrete_dofs = np.concatenate(
        [
            [dof_index(concrete_offset + i, 0), dof_index(concrete_offset + i, 1)]
            for i in range(n_steel_nodes)
        ]
    )
    fixed = concrete_dofs
    prescribed = np.array([dof_index(0, 0)])
    control = DisplacementControl(
        fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=np.array([s0])
    )
    result = nonlinear_solve(model, control, SolverOptions(n_steps=10, max_iterations=40))
    assert result.converged
    u = result.steps[-1].u

    from fem2d_nl.assembly import assemble

    f_int, _, _ = assemble(model, u, result.steps[-1].states, dt=1.0)
    reaction = f_int[dof_index(0, 0)]
    return reaction


@pytest.mark.slow
@pytest.mark.parametrize("s0", [0.3, 0.8])
def test_pullout_fem_matches_bond_slip_ode(s0):
    p_fem = _solve_fem_pullout(s0)
    p_ode = _solve_ode_reference(s0)
    assert p_fem == pytest.approx(p_ode, rel=0.05)
