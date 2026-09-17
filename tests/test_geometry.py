import numpy as np
import pytest

from fem2d.exceptions import GeometryError
from fem2d.geometry import PlateGeometryParams, build_plate_mesh, validate_geometry


def test_plate_without_hole_has_triangles_only():
    params = PlateGeometryParams(length=100.0, height=50.0, hole_radius=0.0, mesh_size=10.0)
    mesh = build_plate_mesh(params)

    assert mesh.triangles.shape[1] == 3
    assert mesh.triangles.max() < mesh.points.shape[0]
    # todo nodo debe ser referenciado por al menos un triángulo (sin huérfanos)
    assert set(np.unique(mesh.triangles)) == set(range(mesh.points.shape[0]))


def test_plate_with_hole_filters_line_and_vertex_cells():
    params = PlateGeometryParams(
        length=100.0, height=50.0, hole_radius=10.0, mesh_size=5.0, hole_mesh_size=1.5
    )
    mesh = build_plate_mesh(params)

    assert mesh.triangles.shape[1] == 3
    assert mesh.triangles.min() >= 0
    assert mesh.triangles.max() < mesh.points.shape[0]
    assert set(np.unique(mesh.triangles)) == set(range(mesh.points.shape[0]))

    # el agujero debe haber "vaciado" puntos cerca del centro
    center = np.array([params.length / 2, params.height / 2])
    dists = np.linalg.norm(mesh.points - center, axis=1)
    assert dists.min() >= params.hole_radius - 1e-6


def test_hole_radius_too_large_raises_geometry_error():
    params = PlateGeometryParams(length=100.0, height=50.0, hole_radius=30.0, mesh_size=5.0)
    with pytest.raises(GeometryError):
        validate_geometry(params)


def test_mesh_size_too_coarse_raises_geometry_error():
    params = PlateGeometryParams(length=100.0, height=50.0, hole_radius=0.0, mesh_size=40.0)
    with pytest.raises(GeometryError):
        validate_geometry(params)


def test_mesh_size_too_fine_raises_geometry_error():
    params = PlateGeometryParams(length=100.0, height=50.0, hole_radius=0.0, mesh_size=0.01)
    with pytest.raises(GeometryError):
        validate_geometry(params)


def test_negative_dimensions_raise_geometry_error():
    with pytest.raises(GeometryError):
        validate_geometry(
            PlateGeometryParams(length=-1.0, height=50.0, hole_radius=0.0, mesh_size=5.0)
        )
    with pytest.raises(GeometryError):
        validate_geometry(
            PlateGeometryParams(length=100.0, height=50.0, hole_radius=-1.0, mesh_size=5.0)
        )
