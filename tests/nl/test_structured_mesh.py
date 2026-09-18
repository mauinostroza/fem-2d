"""Sesión S5: sanity de `mesh.structured.build_section_mesh` — conteo de
nodos/elementos, conectividad antihoraria (Jacobiano positivo, verificado
indirectamente ensamblando en u=0), y que las capas de armadura caen
exactamente en la fila de malla más cercana ("snapping", ver docstring del
módulo)."""

import numpy as np
import pytest

from fem2d_nl.assembly import assemble
from fem2d_nl.materials.bond_mc2010 import BondLaw, BondParameters
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel
from fem2d_nl.mesh.section import RebarLayer, SectionGeometry
from fem2d_nl.mesh.structured import build_section_mesh, nominal_char_length


def _concrete(geom):
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=25.0)
    return ConcreteCDP(params, char_length=nominal_char_length(geom))


def _steel():
    return MultilinearSteel(points=np.array([[0.002, 400.0], [0.05, 460.0]]))


def test_perfect_bond_reuses_concrete_nodes():
    geom = SectionGeometry(width=200.0, height=400.0, span=100.0, nx=2, ny=8)
    layer = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=_steel())
    model = build_section_mesh(geom, [layer], _concrete(geom), bond_mode="perfect")

    assert model.n_nodes == (geom.nx + 1) * (geom.ny + 1)  # sin nodos nuevos
    kinds = [g.kind for g in model.groups]
    assert kinds == ["quad4", "truss2"]
    assert model.groups[0].n_elements == geom.nx * geom.ny
    assert model.groups[1].n_elements == geom.nx

    # la fila de acero debe estar exactamente en y=40 (snapping)
    truss_nodes = np.unique(model.groups[1].connectivity)
    assert np.allclose(model.nodes[truss_nodes, 1], 40.0)


def test_bond_slip_duplicates_nodes_and_adds_links():
    geom = SectionGeometry(width=200.0, height=400.0, span=100.0, nx=2, ny=8)
    layer = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=_steel())
    concrete = _concrete(geom)
    bond_law = BondLaw(BondParameters(fcm=concrete.fcm, bar_diameter=16.0))
    model = build_section_mesh(geom, [layer], concrete, bond_mode="bond_slip", bond_params=bond_law)

    n_concrete_nodes = (geom.nx + 1) * (geom.ny + 1)
    assert model.n_nodes == n_concrete_nodes + (geom.nx + 1)  # + fila de acero duplicada
    kinds = [g.kind for g in model.groups]
    assert kinds == ["quad4", "truss2", "bond_link"]
    assert model.groups[2].n_elements == geom.nx + 1  # un link por nodo de esa fila

    # cada par (acero, hormigón) del link debe ser geométricamente coincidente
    bond_conn = model.groups[2].connectivity
    steel_xy = model.nodes[bond_conn[:, 0]]
    concrete_xy = model.nodes[bond_conn[:, 1]]
    assert np.allclose(steel_xy, concrete_xy)


def test_two_layers_collide_on_same_row_raises():
    geom = SectionGeometry(width=200.0, height=400.0, span=100.0, nx=2, ny=4)  # filas cada 100mm
    layer_a = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=_steel())
    layer_b = RebarLayer(depth_y=45.0, n_bars=2, diameter=12.0, steel=_steel())  # misma fila más cercana
    with pytest.raises(ValueError):
        build_section_mesh(geom, [layer_a, layer_b], _concrete(geom), bond_mode="perfect")


def test_assemble_at_zero_displacement_is_well_posed():
    """No debe lanzar (Jacobiano no positivo, etc.) al ensamblar en u=0;
    f_int debe ser exactamente cero (sin deformación, sin esfuerzo)."""
    geom = SectionGeometry(width=200.0, height=400.0, span=100.0, nx=3, ny=6)
    layers = [
        RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=_steel()),
        RebarLayer(depth_y=360.0, n_bars=2, diameter=12.0, steel=_steel()),
    ]
    concrete = _concrete(geom)
    bond_law = BondLaw(BondParameters(fcm=concrete.fcm, bar_diameter=16.0))
    model = build_section_mesh(geom, layers, concrete, bond_mode="bond_slip", bond_params=bond_law)

    states = model.initial_state()
    u0 = np.zeros(model.ndof)
    f_int, k_mat, _ = assemble(model, u0, states, dt=1.0)
    assert np.allclose(f_int, 0.0)
    assert k_mat.shape == (model.ndof, model.ndof)


def test_no_rebar_layers_is_plain_concrete_mesh():
    geom = SectionGeometry(width=200.0, height=400.0, span=100.0, nx=2, ny=4)
    model = build_section_mesh(geom, [], _concrete(geom), bond_mode="perfect")
    assert len(model.groups) == 1
    assert model.n_nodes == (geom.nx + 1) * (geom.ny + 1)
