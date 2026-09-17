"""Valida el solver no lineal (en régimen puramente elástico, sin
plasticidad) contra: (a) el esfuerzo aplicado analítico, y (b) el
resultado del módulo lineal `fem2d` existente (SolidsPy/CST) para el
mismo problema de tracción uniaxial. Dos formulaciones de elemento
distintas (Q4 vs CST) deben coincidir en el esfuerzo medio lejos de
los bordes cargados/apoyados.
"""

import numpy as np
import pytest

from fem2d.boundary_conditions import (
    Edge,
    EdgeCondition,
    EdgeLoad,
    Support,
    build_cons_array,
    build_loads_array,
    tributary_lengths,
)
from fem2d.geometry import PlateGeometryParams, build_plate_mesh
from fem2d.materials import MaterialProps
from fem2d.solver import run_analysis as run_linear_analysis
from fem2d_nl.bc import dof_index, fixed_dofs_from_nodes
from fem2d_nl.elements import quad4
from fem2d_nl.materials.elastic import ElasticPlaneStress
from fem2d_nl.model import ElementGroup, Model
from fem2d_nl.solver.controls import LoadControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve

LENGTH, HEIGHT = 100.0, 50.0
YOUNG, POISSON = 200_000.0, 0.3
APPLIED_STRESS = 50.0


def _structured_quad_grid(length, height, nx, ny):
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    def nid(i, j):
        return j * (nx + 1) + i

    quads = [
        [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
        for j in range(ny)
        for i in range(nx)
    ]
    return nodes, np.array(quads, dtype=int)


def _solve_nl_uniaxial_tension():
    nodes, quads = _structured_quad_grid(LENGTH, HEIGHT, nx=10, ny=6)
    material = ElasticPlaneStress(young_modulus=YOUNG, poisson_ratio=POISSON)
    group = ElementGroup(kind="quad4", connectivity=quads, material=material, thickness=1.0)
    model = Model(nodes=nodes, groups=[group])

    left = np.nonzero(np.isclose(nodes[:, 0], 0.0))[0]
    bottom = np.nonzero(np.isclose(nodes[:, 1], 0.0))[0]
    right = np.nonzero(np.isclose(nodes[:, 0], LENGTH))[0]

    fixed = np.union1d(
        fixed_dofs_from_nodes(left, components=(0,)),
        fixed_dofs_from_nodes(bottom, components=(1,)),
    )

    order = np.argsort(nodes[right, 1])
    right_sorted = right[order]
    trib = tributary_lengths(nodes[right_sorted, 1])
    force_pattern = np.zeros(model.ndof)
    for node, t in zip(right_sorted, trib):
        force_pattern[dof_index(node, 0)] = APPLIED_STRESS * t

    control = LoadControl(fixed_dofs=fixed, force_pattern=force_pattern)
    result = nonlinear_solve(model, control, SolverOptions(n_steps=1))
    assert result.converged
    u = result.steps[-1].u

    d_mat = material._d_matrix()
    interior_sxx = []
    for elem in quads:
        coords = nodes[elem]
        cx = coords[:, 0].mean()
        if not (0.2 * LENGTH < cx < 0.8 * LENGTH):
            continue
        u_e = np.empty(8)
        u_e[0::2] = u[2 * np.array(elem)]
        u_e[1::2] = u[2 * np.array(elem) + 1]
        b_mat, _ = quad4.b_matrix(coords, 0.0, 0.0)
        stress = d_mat @ (b_mat @ u_e)
        interior_sxx.append(stress[0])
    return np.array(interior_sxx)


def _solve_linear_uniaxial_tension():
    params = PlateGeometryParams(length=LENGTH, height=HEIGHT, hole_radius=0.0, mesh_size=5.0)
    mesh = build_plate_mesh(params)
    material = MaterialProps(young_modulus=YOUNG, poisson_ratio=POISSON)
    conditions = {
        Edge.LEFT: EdgeCondition(support=Support.ROLLER_X),
        Edge.BOTTOM: EdgeCondition(support=Support.ROLLER_Y),
        Edge.RIGHT: EdgeCondition(load=EdgeLoad(magnitude=APPLIED_STRESS, direction="x")),
    }
    cons = build_cons_array(mesh.points, LENGTH, HEIGHT, conditions)
    loads = build_loads_array(mesh.points, LENGTH, HEIGHT, conditions)
    result = run_linear_analysis(mesh, material, cons, loads)
    interior = (mesh.points[:, 0] > 0.2 * LENGTH) & (mesh.points[:, 0] < 0.8 * LENGTH)
    return result.stress[interior, 0]


def test_nl_elastic_matches_applied_stress():
    sxx = _solve_nl_uniaxial_tension()
    assert sxx.mean() == pytest.approx(APPLIED_STRESS, rel=0.02)


def test_nl_elastic_consistent_with_linear_fem2d_module():
    sxx_nl = _solve_nl_uniaxial_tension()
    sxx_linear = _solve_linear_uniaxial_tension()
    # Dos formulaciones de elemento independientes (Q4 vs CST) deben
    # coincidir en el esfuerzo medio lejos de bordes cargados/apoyados.
    assert sxx_nl.mean() == pytest.approx(sxx_linear.mean(), rel=0.05)
