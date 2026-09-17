"""Ensamblaje del vector de fuerzas internas y la matriz tangente global.

Recorre cada `ElementGroup` del modelo, delega la respuesta de cada
elemento al módulo de elemento correspondiente (`fem2d_nl.elements.*`)
y ensambla por COO (que suma contribuciones duplicadas automáticamente
al convertir a CSR).
"""

import numpy as np
from scipy.sparse import coo_matrix

from fem2d_nl.elements import bond_link, quad4, truss2

_ELEMENT_RESPONSE = {
    "quad4": quad4.element_response,
    "truss2": truss2.element_response,
    "bond_link": bond_link.element_response,
}


def _elem_kwargs_for(group, e: int, ne: int) -> dict:
    """Extrae los kwargs geométricos del elemento `e`: si un valor de
    `elem_kwargs` es un array cuyo primer eje mide `ne`, se toma su fila
    `e`; si no, se pasa tal cual (mismo valor para todo el grupo)."""
    out = {}
    for key, val in group.elem_kwargs.items():
        if isinstance(val, np.ndarray) and val.shape[0] == ne:
            out[key] = val[e]
        else:
            out[key] = val
    return out


def assemble(model, u: np.ndarray, states: list[dict], dt: float):
    """Ensambla (f_int, K_tangente, nuevos_estados) en el desplazamiento `u`.

    `states` es la lista de estados por grupo (alineada con `model.groups`),
    tal como la devuelve `Model.initial_state()`. No se muta: se devuelve
    una lista nueva con los estados de prueba actualizados, que el solver
    solo hace permanentes al converger un paso de carga.
    """
    ndof = model.ndof
    f_int = np.zeros(ndof)
    rows_all, cols_all, vals_all = [], [], []
    new_states: list[dict] = []

    for group, state in zip(model.groups, states):
        response_fn = _ELEMENT_RESPONSE[group.kind]
        dofs = group.dof_map()  # (ne, ndpe)
        ne, ndpe = dofs.shape
        n_gp = group.n_gauss_per_element

        f_local = np.empty((ne, ndpe))
        k_local = np.empty((ne, ndpe, ndpe))
        new_state = {k: np.empty_like(v) for k, v in state.items()}

        for e in range(ne):
            coords = model.nodes[group.connectivity[e]]
            u_e = u[dofs[e]]
            gp = slice(e * n_gp, (e + 1) * n_gp)
            state_e = {k: v[gp] for k, v in state.items()}
            kwargs_e = _elem_kwargs_for(group, e, ne)
            f_e, k_e, new_state_e = response_fn(
                coords, u_e, group.material, state_e, dt, **kwargs_e
            )
            f_local[e] = f_e
            k_local[e] = k_e
            for key, val in new_state_e.items():
                new_state[key][gp] = val

        new_states.append(new_state)
        np.add.at(f_int, dofs.ravel(), f_local.ravel())

        rows = np.broadcast_to(dofs[:, :, None], (ne, ndpe, ndpe))
        cols = np.broadcast_to(dofs[:, None, :], (ne, ndpe, ndpe))
        rows_all.append(rows.ravel())
        cols_all.append(cols.ravel())
        vals_all.append(k_local.ravel())

    if rows_all:
        k_global = coo_matrix(
            (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
            shape=(ndof, ndof),
        ).tocsr()
    else:
        k_global = coo_matrix((ndof, ndof)).tocsr()

    return f_int, k_global, new_states
