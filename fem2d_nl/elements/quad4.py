"""Elemento continuo Q4: cuadrilátero bilineal, 2x2 Gauss, tensión plana.

Convención de nodos antihoraria: (-1,-1), (1,-1), (1,1), (-1,1).
Convención de grados de libertad por elemento: [u1,v1,u2,v2,u3,v3,u4,v4]
(igual que la matriz B del triángulo CST en fem2d/solver.py).
"""

import numpy as np

_GP = 1.0 / np.sqrt(3.0)
GAUSS_2X2 = np.array(
    [
        [-_GP, -_GP],
        [_GP, -_GP],
        [_GP, _GP],
        [-_GP, _GP],
    ]
)
GAUSS_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
N_GAUSS = 4


def shape_functions(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (N (4,), dN_dnat (2,4)) evaluadas en (xi, eta)."""
    n = 0.25 * np.array(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ]
    )
    dn_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dn_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return n, np.vstack([dn_dxi, dn_deta])


def b_matrix(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """B (3,8) y det(J) en (xi, eta), para las coordenadas nodales `coords` (4,2)."""
    _, dn_dnat = shape_functions(xi, eta)
    jac = dn_dnat @ coords  # (2,2)
    det_j = np.linalg.det(jac)
    if det_j <= 0:
        raise ValueError(
            "Jacobiano no positivo en un elemento Q4: nodos no antihorarios o elemento degenerado."
        )
    dn_dx = np.linalg.solve(jac, dn_dnat)  # (2,4)
    b_mat = np.zeros((3, 8))
    b_mat[0, 0::2] = dn_dx[0, :]
    b_mat[1, 1::2] = dn_dx[1, :]
    b_mat[2, 0::2] = dn_dx[1, :]
    b_mat[2, 1::2] = dn_dx[0, :]
    return b_mat, det_j


def gauss_point_coords(coords: np.ndarray) -> np.ndarray:
    """Coordenadas físicas (4,2) de los 4 puntos de Gauss del elemento."""
    out = np.empty((N_GAUSS, 2))
    for g, (xi, eta) in enumerate(GAUSS_2X2):
        n, _ = shape_functions(xi, eta)
        out[g] = n @ coords
    return out


def element_response(coords, u_e, material, state, thickness, dt):
    """Respuesta de un elemento Q4.

    coords: (4,2) coordenadas nodales
    u_e: (8,) desplazamientos nodales del elemento
    material: objeto con `.integrate(strain(4,3), state, dt)`
    state: dict[str, np.ndarray] con primer eje = 4 (un estado por punto de Gauss)
    thickness: espesor (ancho fuera del plano)

    Devuelve (f_int (8,), k_t (8,8), new_state).
    """
    b_list = []
    det_j_list = []
    strain = np.empty((N_GAUSS, 3))
    for g, (xi, eta) in enumerate(GAUSS_2X2):
        b_mat, det_j = b_matrix(coords, xi, eta)
        b_list.append(b_mat)
        det_j_list.append(det_j)
        strain[g] = b_mat @ u_e

    stress, tangent, new_state = material.integrate(strain, state, dt)

    f_int = np.zeros(8)
    k_t = np.zeros((8, 8))
    for g in range(N_GAUSS):
        w = GAUSS_WEIGHTS[g] * det_j_list[g] * thickness
        b_mat = b_list[g]
        f_int += w * (b_mat.T @ stress[g])
        k_t += w * (b_mat.T @ tangent[g] @ b_mat)

    return f_int, k_t, new_state
