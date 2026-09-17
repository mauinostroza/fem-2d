import numpy as np
import pytest

from fem2d.boundary_conditions import (
    Edge,
    EdgeCondition,
    EdgeLoad,
    Support,
    build_cons_array,
    build_loads_array,
)
from fem2d.geometry import PlateGeometryParams, build_plate_mesh
from fem2d.materials import MaterialProps
from fem2d.solver import run_analysis


def test_uniaxial_tension_matches_applied_stress_away_from_edges():
    length, height = 100.0, 50.0
    params = PlateGeometryParams(length=length, height=height, hole_radius=0.0, mesh_size=5.0)
    mesh = build_plate_mesh(params)
    material = MaterialProps(young_modulus=200_000.0, poisson_ratio=0.3)

    applied_stress = 50.0  # MPa
    conditions = {
        Edge.LEFT: EdgeCondition(support=Support.ROLLER_X),
        Edge.BOTTOM: EdgeCondition(support=Support.ROLLER_Y),
        Edge.RIGHT: EdgeCondition(load=EdgeLoad(magnitude=applied_stress, direction="x")),
    }
    cons = build_cons_array(mesh.points, length, height, conditions)
    loads = build_loads_array(mesh.points, length, height, conditions)

    result = run_analysis(mesh, material, cons, loads)

    interior = (mesh.points[:, 0] > 0.2 * length) & (mesh.points[:, 0] < 0.8 * length)
    sxx_interior = result.stress[interior, 0]
    syy_interior = result.stress[interior, 1]

    assert sxx_interior.mean() == pytest.approx(applied_stress, rel=0.03)
    assert syy_interior.mean() == pytest.approx(0.0, abs=1.0)

    vm_interior = result.von_mises[interior]
    assert vm_interior.mean() == pytest.approx(applied_stress, rel=0.03)


def test_uniaxial_tension_displacement_grows_along_load_direction():
    length, height = 100.0, 50.0
    params = PlateGeometryParams(length=length, height=height, hole_radius=0.0, mesh_size=5.0)
    mesh = build_plate_mesh(params)
    material = MaterialProps(young_modulus=200_000.0, poisson_ratio=0.3)

    conditions = {
        Edge.LEFT: EdgeCondition(support=Support.ROLLER_X),
        Edge.BOTTOM: EdgeCondition(support=Support.ROLLER_Y),
        Edge.RIGHT: EdgeCondition(load=EdgeLoad(magnitude=50.0, direction="x")),
    }
    cons = build_cons_array(mesh.points, length, height, conditions)
    loads = build_loads_array(mesh.points, length, height, conditions)

    result = run_analysis(mesh, material, cons, loads)

    ux_left = result.displacement[np.isclose(mesh.points[:, 0], 0.0), 0]
    ux_right = result.displacement[np.isclose(mesh.points[:, 0], length), 0]
    assert ux_left.mean() == pytest.approx(0.0, abs=1e-8)
    assert ux_right.mean() > ux_left.mean()
