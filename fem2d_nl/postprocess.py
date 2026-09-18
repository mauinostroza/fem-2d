"""Postprocesamiento estructural: momento-curvatura de una sección vía el
modelo FEM completo (franja de `mesh.structured.build_section_mesh`).

## Condición de borde de flexión

Se impone la cinemática de "secciones planas" como condición esencial en
ambos extremos de la franja (reutiliza `DisplacementControl` tal cual, sin
necesitar multi-point constraints):

- Cara de referencia (`x=0`): `ux=0` en TODOS los nodos de esa cara (fija
  rotación y alargamiento medio ahí) + `uy=0` en un solo nodo (elimina el
  modo de cuerpo rígido de traslación en y). Con esto los 3 modos de
  cuerpo rígido del plano ya quedan bloqueados.
- Cara cargada (`x=span`): `ux_i = -kappa_max*(y_i-y_na)*span*lam`
  prescrito en todos los nodos de esa cara (perfil lineal = secciones
  planas), escalado por el factor de carga `lam` del solver.

`M(kappa)` y `N(kappa)` se extraen de las reacciones en la cara de
referencia (`StepResult.f_int` en esos dofs, ya calculado por el solver):
`N = sum(f_int[ux])`, `M = sum(f_int[ux]*(y-y_na))` — exacto por
equilibrio nodal FEM estándar, sin hipótesis adicionales.

## Aviso de honestidad técnica

`y_na` queda FIJO (por defecto, el centroide elástico transformado inicial)
en vez de buscarse por iteración para forzar `N=0` en cada paso. Es una
condición de "flexión con eje de referencia fijo", no flexión pura exacta:
a medida que la sección fisura, el eje neutro físico se corre y `N(kappa)`
deja de ser cero (queda como un pequeño axial parásito). Por eso esta
función siempre devuelve `N` explícitamente en vez de asumirlo cero —
el llamador debe revisarlo, en particular en curvaturas grandes.
Buscar `y_na` por bisección externa para forzar `N≈0` en cada paso queda
como refinamiento futuro (no implementado).
"""

from dataclasses import dataclass, field

import numpy as np

from fem2d_nl.bc import dof_index, fixed_dofs_from_nodes
from fem2d_nl.solver.controls import DisplacementControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve


@dataclass
class MomentCurvatureResult:
    kappa: np.ndarray  # (n_steps,) curvatura impuesta en cada paso convergido
    moment: np.ndarray  # (n_steps,) momento respecto de y_na
    axial: np.ndarray  # (n_steps,) fuerza axial neta (ver aviso de honestidad arriba)
    y_na: float
    converged: bool
    message: str = ""


def transformed_elastic_centroid(geom, rebar_layers, e_concrete) -> float:
    """Centroide elástico transformado (n=E_acero/E_concreto), usado por
    defecto como `y_na` fijo en `moment_curvature`. Las capas de acero se
    tratan como área adicional (n-1)*A_s en su profundidad (método clásico
    de sección transformada; ignora la reducción de área de hormigón
    desplazada por la barra, error despreciable para armaduras típicas)."""
    a_concrete = geom.width * geom.height
    y_concrete = geom.height / 2.0
    num = a_concrete * y_concrete
    den = a_concrete
    for layer in rebar_layers:
        e_steel = getattr(layer.steel, "e_resolved", None)
        if e_steel is None:
            raise ValueError("El material de la capa debe exponer `e_resolved`.")
        n_ratio = e_steel / e_concrete
        a_extra = (n_ratio - 1.0) * layer.area
        num += a_extra * layer.depth_y
        den += a_extra
    return num / den


def _face_nodes(model, x_target: float, tol: float) -> np.ndarray:
    matches = np.nonzero(np.abs(model.nodes[:, 0] - x_target) < tol)[0]
    if matches.size == 0:
        raise ValueError(f"No se encontraron nodos en x={x_target}.")
    return matches


def moment_curvature(
    model,
    span: float,
    y_na: float,
    kappa_max: float,
    n_steps: int = 20,
    options: SolverOptions | None = None,
    node_match_tol: float | None = None,
) -> MomentCurvatureResult:
    """Analiza `model` (construido por `mesh.structured.build_section_mesh`
    con ese `span`) bajo curvatura creciente hasta `kappa_max`, en
    `n_steps` incrementos de control de desplazamiento.

    Solo considera nodos de HORMIGÓN (x=0 y x=span) para la condición de
    borde y la extracción de M/N — los nodos de acero duplicados (modo
    `bond_slip`) en esos mismos x quedan libres, como corresponde: el
    acero solo transmite fuerza a través de los links de adherencia, no
    directamente a través de la cara de la sección.
    """
    tol = node_match_tol if node_match_tol is not None else span * 1e-9 + 1e-9
    ref_nodes = _face_nodes(model, 0.0, tol)
    load_nodes = _face_nodes(model, span, tol)

    fixed = fixed_dofs_from_nodes(ref_nodes, components=(0,))
    # un solo nodo de referencia fija uy=0 (traslación en y); cualquiera sirve
    fixed = np.union1d(fixed, fixed_dofs_from_nodes(np.array([ref_nodes[0]]), components=(1,)))

    prescribed = np.array([dof_index(int(n), 0) for n in load_nodes])
    y_load = model.nodes[load_nodes, 1]
    prescribed_values = -kappa_max * (y_load - y_na) * span

    control = DisplacementControl(
        fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=prescribed_values
    )
    result = nonlinear_solve(model, control, options or SolverOptions(n_steps=n_steps))

    y_ref = model.nodes[ref_nodes, 1]
    ref_ux_dofs = np.array([dof_index(int(n), 0) for n in ref_nodes])

    kappa_hist = np.array([step.lam * kappa_max for step in result.steps])
    axial_hist = np.array([step.f_int[ref_ux_dofs].sum() for step in result.steps])
    moment_hist = np.array(
        [np.sum(step.f_int[ref_ux_dofs] * (y_ref - y_na)) for step in result.steps]
    )

    return MomentCurvatureResult(
        kappa=kappa_hist,
        moment=moment_hist,
        axial=axial_hist,
        y_na=y_na,
        converged=result.converged,
        message=result.message,
    )
