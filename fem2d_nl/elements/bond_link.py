"""Elemento de interfaz de adherencia: 2 nodos geométricamente coincidentes
(uno de acero, uno de hormigón), sistema local (t,n) con `t` a lo largo
del eje de la barra.

Convención de conectividad: fila = [nodo_acero, nodo_hormigón].
Convención de dofs del elemento: u_e = [u_s_x, u_s_y, u_c_x, u_c_y].

Material 1D (igual contrato que `steel_uniaxial`):
`material.integrate(slip(n,), state, dt) -> (tau(n,), dtau_ds(n,), new_state)`.
"""

import numpy as np


def element_response(coords, u_e, material, state, dt, perimeter=1.0, trib_length=1.0, k_normal=1.0, axis=(1.0, 0.0)):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])

    rel = u_e[0:2] - u_e[2:4]  # u_acero - u_hormigón
    slip = np.array([rel @ axis])
    gap = float(rel @ normal)

    tau, dtau_ds, new_state = material.integrate(slip, state, dt)
    tau, dtau_ds = tau[0], dtau_ds[0]

    scale = perimeter * trib_length
    f_t = tau * scale
    f_n = k_normal * gap

    b_t = np.array([axis[0], axis[1], -axis[0], -axis[1]])
    b_n = np.array([normal[0], normal[1], -normal[0], -normal[1]])

    f_int = f_t * b_t + f_n * b_n
    k_t = (dtau_ds * scale) * np.outer(b_t, b_t) + k_normal * np.outer(b_n, b_n)
    return f_int, k_t, new_state
