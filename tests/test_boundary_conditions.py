import numpy as np
import pytest

from fem2d.boundary_conditions import (
    Edge,
    EdgeCondition,
    EdgeLoad,
    Support,
    build_cons_array,
    build_loads_array,
    check_rigid_body_constraints,
    find_edge_nodes,
    tributary_lengths,
)
from fem2d.exceptions import BoundaryConditionError


def _rectangle_grid(length, height, nx, ny):
    xs = np.linspace(0, length, nx)
    ys = np.linspace(0, height, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_tributary_lengths_sum_to_edge_length():
    rng = np.random.default_rng(0)
    coord = np.sort(np.concatenate([[0.0, 12.0], rng.uniform(0, 12, size=8)]))
    trib = tributary_lengths(coord)
    assert trib.sum() == pytest.approx(12.0)


def test_tributary_lengths_uniform_grid_matches_spacing():
    coord = np.linspace(0.0, 10.0, 6)  # espaciado uniforme de 2.0
    trib = tributary_lengths(coord)
    # interiores deben valer exactamente el espaciado; extremos la mitad
    assert trib[1:-1] == pytest.approx(2.0)
    assert trib[0] == pytest.approx(1.0)
    assert trib[-1] == pytest.approx(1.0)


def test_find_edge_nodes_identifies_correct_boundary():
    points = _rectangle_grid(100.0, 50.0, 5, 3)
    left = find_edge_nodes(points, Edge.LEFT, 100.0, 50.0, tol=1e-6)
    right = find_edge_nodes(points, Edge.RIGHT, 100.0, 50.0, tol=1e-6)
    assert np.all(points[left, 0] == pytest.approx(0.0))
    assert np.all(points[right, 0] == pytest.approx(100.0))
    assert left.size == 3
    assert right.size == 3


def test_build_loads_array_resultant_matches_applied_stress():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 9)
    magnitude = 20.0  # MPa
    conditions = {
        Edge.RIGHT: EdgeCondition(load=EdgeLoad(magnitude=magnitude, direction="x")),
    }
    loads = build_loads_array(points, length, height, conditions)
    total_fx = loads[:, 1].sum()
    total_fy = loads[:, 2].sum()
    # resultante = esfuerzo * longitud del borde (espesor unitario)
    assert total_fx == pytest.approx(magnitude * height)
    assert total_fy == pytest.approx(0.0)


def test_build_loads_array_accumulates_on_shared_corner():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 4)
    conditions = {
        Edge.RIGHT: EdgeCondition(load=EdgeLoad(magnitude=10.0, direction="x")),
        Edge.TOP: EdgeCondition(load=EdgeLoad(magnitude=5.0, direction="y")),
    }
    loads = build_loads_array(points, length, height, conditions)
    corner_idx = np.argmin(
        np.linalg.norm(points[loads[:, 0].astype(int)] - np.array([length, height]), axis=1)
    )
    corner_row = loads[corner_idx]
    assert corner_row[1] > 0  # tiene componente x del borde derecho
    assert corner_row[2] > 0  # tiene componente y del borde superior


def test_build_cons_array_fixed_edge():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 4)
    conditions = {Edge.LEFT: EdgeCondition(support=Support.FIXED)}
    cons = build_cons_array(points, length, height, conditions)
    left_idx = find_edge_nodes(points, Edge.LEFT, length, height, tol=1e-6)
    assert np.all(cons[left_idx] == -1)
    other_idx = np.setdiff1d(np.arange(points.shape[0]), left_idx)
    assert np.all(cons[other_idx] == 0)


def test_build_cons_array_more_restrictive_wins_on_shared_corner():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 4)
    conditions = {
        Edge.LEFT: EdgeCondition(support=Support.ROLLER_X),
        Edge.BOTTOM: EdgeCondition(support=Support.FIXED),
    }
    cons = build_cons_array(points, length, height, conditions)
    corner_idx = np.argmin(np.linalg.norm(points - np.array([0.0, 0.0]), axis=1))
    assert list(cons[corner_idx]) == [-1, -1]


def test_rigid_body_check_rejects_free_floating_plate():
    points = _rectangle_grid(100.0, 50.0, 4, 4)
    cons = np.zeros((points.shape[0], 2), dtype=int)
    with pytest.raises(BoundaryConditionError):
        check_rigid_body_constraints(points, cons)


def test_rigid_body_check_rejects_roller_y_only_on_vertical_edge():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 4)
    conditions = {Edge.LEFT: EdgeCondition(support=Support.ROLLER_Y)}
    cons = build_cons_array(points, length, height, conditions)
    with pytest.raises(BoundaryConditionError):
        check_rigid_body_constraints(points, cons)


def test_rigid_body_check_accepts_roller_x_left_plus_roller_y_bottom():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 4)
    conditions = {
        Edge.LEFT: EdgeCondition(support=Support.ROLLER_X),
        Edge.BOTTOM: EdgeCondition(support=Support.ROLLER_Y),
    }
    cons = build_cons_array(points, length, height, conditions)
    check_rigid_body_constraints(points, cons)  # no debe lanzar


def test_rigid_body_check_accepts_fixed_edge():
    length, height = 100.0, 50.0
    points = _rectangle_grid(length, height, 4, 4)
    conditions = {Edge.LEFT: EdgeCondition(support=Support.FIXED)}
    cons = build_cons_array(points, length, height, conditions)
    check_rigid_body_constraints(points, cons)  # no debe lanzar
