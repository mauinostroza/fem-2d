"""Elemento barra (truss) de 2 nodos: deformación axial constante.

Usa un contrato de material 1D distinto del de los elementos continuos:
`material.integrate(eps(n,), state, dt) -> (sigma(n,), tangent(n,), new_state)`
(escalares, no tensores de esfuerzo plano) — es lo natural para un
material uniaxial como el acero de refuerzo.
"""

import numpy as np

N_GAUSS = 1


def element_response(coords, u_e, material, state, dt, area=1.0):
    """coords: (2,2) coordenadas de los 2 nodos. u_e: (4,) [u1,v1,u2,v2].
    area: área de la sección de la barra (o del conjunto de barras que
    representa la línea de elementos, ver `n_bars*pi*phi^2/4`)."""
    (x1, y1), (x2, y2) = coords
    length = np.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        raise ValueError("Elemento barra de longitud nula.")
    nx, ny = (x2 - x1) / length, (y2 - y1) / length
    b_vec = np.array([-nx, -ny, nx, ny]) / length

    strain = np.array([b_vec @ u_e])
    stress, tangent, new_state = material.integrate(strain, state, dt)

    f_int = area * stress[0] * length * b_vec
    k_t = area * tangent[0] * length * np.outer(b_vec, b_vec)
    return f_int, k_t, new_state
