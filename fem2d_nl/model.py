"""Contenedor del modelo no lineal: nodos, grupos de elementos, materiales.

Un `ElementGroup` agrupa elementos del mismo tipo y material (p. ej.
todos los cuadriláteros de hormigón). El estado de los puntos de
integración se guarda por grupo, como arrays (struct-of-arrays), nunca
como un objeto Python por punto.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from fem2d_nl.elements import quad4

ElementKind = Literal["quad4", "truss2", "bond_link"]

_N_GAUSS_BY_KIND = {"quad4": quad4.N_GAUSS, "truss2": 1, "bond_link": 1}
_N_NODES_BY_KIND = {"quad4": 4, "truss2": 2, "bond_link": 2}


@dataclass
class ElementGroup:
    kind: ElementKind
    connectivity: np.ndarray  # (ne, n_nodes_por_elemento), índices 0-based en Model.nodes
    material: object  # implementa initial_state(n)/integrate(strain,state,dt)
    elem_kwargs: dict = field(default_factory=dict)
    """Parámetros geométricos por elemento, específicos del tipo de
    elemento: `thickness` (quad4), `area` (truss2), `perimeter`/
    `trib_length`/`k_normal`/`axis` (bond_link). Cada valor puede ser un
    escalar (igual para todo el grupo) o un array (ne, ...) para variarlo
    por elemento (p. ej. distintos diámetros de barra en un mismo grupo)."""

    @property
    def n_elements(self) -> int:
        return self.connectivity.shape[0]

    @property
    def n_gauss_per_element(self) -> int:
        return _N_GAUSS_BY_KIND[self.kind]

    @property
    def n_gauss_total(self) -> int:
        return self.n_elements * self.n_gauss_per_element

    def dof_map(self) -> np.ndarray:
        """(ne, n_nodes*2): dofs globales [u,v] de cada nodo del elemento, en orden."""
        conn = self.connectivity
        dofs = np.empty((conn.shape[0], conn.shape[1] * 2), dtype=int)
        dofs[:, 0::2] = 2 * conn
        dofs[:, 1::2] = 2 * conn + 1
        return dofs

    def initial_state(self) -> dict:
        return self.material.initial_state(self.n_gauss_total)


@dataclass
class Model:
    nodes: np.ndarray  # (n_nodes, 2)
    groups: list[ElementGroup] = field(default_factory=list)
    ndof_per_node: int = 2

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def ndof(self) -> int:
        return self.n_nodes * self.ndof_per_node

    def initial_state(self) -> list[dict]:
        """Estado inicial por grupo (lista alineada con `self.groups`)."""
        return [group.initial_state() for group in self.groups]
