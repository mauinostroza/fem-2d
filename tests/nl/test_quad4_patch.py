"""Patch test del elemento Q4: un campo de desplazamiento lineal (por lo
tanto deformación constante) impuesto en el borde debe reproducirse
exactamente en el nodo interior y dar el esfuerzo constante D*eps en
todos los elementos, sin importar la forma de los cuadriláteros.
"""

import numpy as np
import pytest

from fem2d_nl.bc import dof_index
from fem2d_nl.elements import quad4
from fem2d_nl.materials.elastic import ElasticPlaneStress
from fem2d_nl.model import ElementGroup, Model
from fem2d_nl.solver.controls import DisplacementControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve

# Malla 3x3 nodos (2x2 elementos) con el nodo central perturbado para que
# los elementos no sean rectángulos perfectos (patch test "duro").
NODES = np.array(
    [
        [0.0, 0.0], [15.0, 0.0], [30.0, 0.0],
        [0.0, 10.0], [17.0, 12.0], [30.0, 10.0],
        [0.0, 20.0], [15.0, 20.0], [30.0, 20.0],
    ]
)
QUADS = np.array(
    [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
    ]
)
INTERIOR_NODE = 4
BOUNDARY_NODES = np.array([0, 1, 2, 3, 5, 6, 7, 8])


def _linear_field(coords, a):
    x, y = coords[:, 0], coords[:, 1]
    ux = a[0] + a[1] * x + a[2] * y
    uy = a[3] + a[4] * x + a[5] * y
    return ux, uy


@pytest.mark.parametrize(
    "a",
    [
        [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],       # eps_xx puro
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.02],      # eps_yy puro
        [0.0, 0.0, 0.005, 0.0, 0.003, 0.0],    # corte
        [0.1, 0.01, -0.002, -0.05, 0.004, 0.02],  # combinado + traslación
    ],
)
def test_q4_reproduces_linear_displacement_field_exactly(a):
    material = ElasticPlaneStress(young_modulus=200_000.0, poisson_ratio=0.3)
    group = ElementGroup(kind="quad4", connectivity=QUADS, material=material, elem_kwargs={"thickness": 1.0})
    model = Model(nodes=NODES, groups=[group])

    ux, uy = _linear_field(NODES[BOUNDARY_NODES], a)
    prescribed_dofs = np.array(
        [dof_index(n, c) for n in BOUNDARY_NODES for c in (0, 1)]
    )
    prescribed_values = np.empty(prescribed_dofs.shape[0])
    prescribed_values[0::2] = ux
    prescribed_values[1::2] = uy

    control = DisplacementControl(
        fixed_dofs=np.array([], dtype=int),
        prescribed_dofs=prescribed_dofs,
        prescribed_values=prescribed_values,
    )
    result = nonlinear_solve(model, control, SolverOptions(n_steps=1))
    assert result.converged
    u = result.steps[-1].u

    ux_exp, uy_exp = _linear_field(NODES[INTERIOR_NODE : INTERIOR_NODE + 1], a)
    assert u[dof_index(INTERIOR_NODE, 0)] == pytest.approx(ux_exp[0], abs=1e-10)
    assert u[dof_index(INTERIOR_NODE, 1)] == pytest.approx(uy_exp[0], abs=1e-10)

    eps_xx, eps_yy, gamma_xy = a[1], a[5], a[2] + a[4]
    d_mat = material._d_matrix()
    expected_stress = d_mat @ np.array([eps_xx, eps_yy, gamma_xy])

    for elem in QUADS:
        coords = NODES[elem]
        u_e = np.empty(8)
        u_e[0::2] = u[2 * elem]
        u_e[1::2] = u[2 * elem + 1]
        b_mat, _ = quad4.b_matrix(coords, 0.0, 0.0)
        strain = b_mat @ u_e
        stress = d_mat @ strain
        assert stress == pytest.approx(expected_stress, abs=1e-8)
