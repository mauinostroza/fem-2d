"""Utilidades de condiciones de borde: mapeo nodo/componente -> grado de
libertad global, y construcción de listas de dofs fijos.
"""

import numpy as np


def dof_index(node_id: int, component: int) -> int:
    """component: 0 = x, 1 = y."""
    return 2 * node_id + component


def fixed_dofs_from_nodes(node_ids: np.ndarray, components: tuple[int, ...] = (0, 1)) -> np.ndarray:
    """Dofs globales fijos (u=0) para los nodos y componentes dados."""
    node_ids = np.asarray(node_ids)
    dofs = [dof_index(n, c) for n in node_ids for c in components]
    return np.array(sorted(set(dofs)), dtype=int)
