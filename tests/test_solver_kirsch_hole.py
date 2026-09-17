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


def test_stress_concentration_matches_kirsch_solution_near_hole():
    """Placa con agujero pequeño frente a su tamaño (aproxima placa infinita
    de Kirsch): el esfuerzo en el borde del agujero, en el punto
    perpendicular a la carga, debe acercarse a Kt=3 veces el esfuerzo
    aplicado lejos del agujero."""
    length, height, radius = 200.0, 100.0, 5.0
    params = PlateGeometryParams(
        length=length, height=height, hole_radius=radius, mesh_size=3.0, hole_mesh_size=0.5
    )
    mesh = build_plate_mesh(params)
    material = MaterialProps(young_modulus=200_000.0, poisson_ratio=0.3)

    applied_stress = 10.0  # MPa
    conditions = {
        Edge.LEFT: EdgeCondition(support=Support.ROLLER_X),
        Edge.BOTTOM: EdgeCondition(support=Support.ROLLER_Y),
        Edge.RIGHT: EdgeCondition(load=EdgeLoad(magnitude=applied_stress, direction="x")),
    }
    cons = build_cons_array(mesh.points, length, height, conditions)
    loads = build_loads_array(mesh.points, length, height, conditions)

    result = run_analysis(mesh, material, cons, loads)

    center = np.array([length / 2, height / 2])
    dist_to_center = np.linalg.norm(mesh.points - center, axis=1)
    on_hole_boundary = np.abs(dist_to_center - radius) < 1e-6

    # punto del contorno del agujero más cercano a (cx, cy + R): perpendicular
    # a la dirección de la carga, donde la teoría de Kirsch predice Kt~3
    target = center + np.array([0.0, radius])
    hole_points = np.nonzero(on_hole_boundary)[0]
    closest = hole_points[np.argmin(np.linalg.norm(mesh.points[hole_points] - target, axis=1))]

    kt_fem = result.von_mises[closest] / applied_stress
    assert 2.5 <= kt_fem <= 3.4

    # sanidad física: el máximo esfuerzo de toda la placa debe estar sobre
    # o muy cerca del contorno del agujero, no en una esquina
    max_node = result.max_von_mises_node
    dist_max_to_hole = abs(dist_to_center[max_node] - radius)
    assert dist_max_to_hole < 1.5 * params.hole_mesh_size
