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
(S7) Existe una variante que SÍ busca `y_na` por bisección en cada paso
para forzar `N≈0` real, `moment_curvature_pure_bending` — mucho más cara
(bisección anidada: cada paso de curvatura implica varias resoluciones no
lineales completas, ~120s/paso medido en una sección chica), por eso no
reemplaza a esta función como default.
"""

from dataclasses import dataclass

import numpy as np

from fem2d_nl.bc import dof_index, fixed_dofs_from_nodes
from fem2d_nl.exceptions import GeometryError
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
    final_states: list | None = None
    """Estado por grupo (alineado con `model.groups`) del último paso
    convergido — p. ej. para dibujar un contorno de daño con
    `visualization_nl.plot_damage_contour`. `None` si ningún paso convergió."""
    y_na_history: np.ndarray | None = None
    """Solo poblado por `moment_curvature_pure_bending` (S7): `y_na` usado
    en cada paso (varía, a diferencia del `y_na` único y fijo del resto de
    los campos). `None` para `moment_curvature` (`y_na` fijo)."""


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
        raise GeometryError(f"No se encontraron nodos en x={x_target}.")
    return matches


def moment_curvature(
    model,
    span: float,
    y_na: float,
    kappa_max: float,
    n_steps: int = 20,
    options: SolverOptions | None = None,
    node_match_tol: float | None = None,
    progress_cb=None,
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
    result = nonlinear_solve(
        model, control, options or SolverOptions(n_steps=n_steps), progress_cb=progress_cb
    )

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
        final_states=result.steps[-1].states if result.steps else None,
    )


def moment_curvature_pure_bending(
    model,
    span: float,
    kappa_max: float,
    n_steps: int = 20,
    y_na_init: float | None = None,
    tol_axial: float | None = None,
    max_bisection: int = 15,
    options: SolverOptions | None = None,
    node_match_tol: float | None = None,
    progress_cb=None,
) -> MomentCurvatureResult:
    """Como `moment_curvature`, pero busca `y_na` por bisección EN CADA
    PASO de curvatura para que la fuerza axial resultante sea ≈0 (flexión
    pura real, no la aproximación de `y_na` fijo — ver el aviso de
    honestidad del módulo).

    Mucho más cara que `moment_curvature`: cada paso de curvatura implica
    resolver el problema no lineal completo ~`max_bisection` veces
    (bisección anidada), en vez de una sola. Cada resolución parte del
    último estado convergido (`nonlinear_solve(..., u_start=, states_start=)`,
    extendido en S7 justo para esto) como "warm start", así que cada
    intento de la bisección es UN solo incremento (no rehace la historia
    de carga completa), pero sigue siendo ~10-20x más lento que
    `moment_curvature` para el mismo `n_steps`.

    `y_na_init` es solo un valor informativo de log/depuración — la
    búsqueda en sí explora todo `[0, height]` en cada paso (no depende de
    un buen punto de partida para converger, solo para el orden de los
    mensajes de progreso).
    """
    tol = node_match_tol if node_match_tol is not None else span * 1e-9 + 1e-9
    ref_nodes = _face_nodes(model, 0.0, tol)
    load_nodes = _face_nodes(model, span, tol)
    ref_ux_dofs = np.array([dof_index(int(n), 0) for n in ref_nodes])
    y_ref = model.nodes[ref_nodes, 1]
    y_load = model.nodes[load_nodes, 1]
    prescribed_dofs = np.array([dof_index(int(n), 0) for n in load_nodes])

    fixed = fixed_dofs_from_nodes(ref_nodes, components=(0,))
    fixed = np.union1d(fixed, fixed_dofs_from_nodes(np.array([ref_nodes[0]]), components=(1,)))

    height = float(model.nodes[:, 1].max())
    step_options = options or SolverOptions(n_steps=1)

    u = np.zeros(model.ndof)
    states = model.initial_state()

    def solve_for_y_na(y_na_trial: float, kappa_target: float):
        prescribed_values = -kappa_target * (y_load - y_na_trial) * span
        control = DisplacementControl(
            fixed_dofs=fixed, prescribed_dofs=prescribed_dofs, prescribed_values=prescribed_values
        )
        sub_result = nonlinear_solve(
            model, control, step_options, u_start=u, states_start=states
        )
        if not sub_result.converged or not sub_result.steps:
            return None
        last = sub_result.steps[-1]
        axial = float(last.f_int[ref_ux_dofs].sum())
        moment = float(np.sum(last.f_int[ref_ux_dofs] * (y_ref - y_na_trial)))
        return last, axial, moment

    kappa_hist, moment_hist, axial_hist, y_na_hist = [], [], [], []
    y_na_current = y_na_init if y_na_init is not None else height / 2.0
    converged_all = True
    message = ""

    for step_idx in range(1, n_steps + 1):
        kappa_target = step_idx / n_steps * kappa_max

        lo, hi = 0.0, height
        res_lo = solve_for_y_na(lo, kappa_target)
        res_hi = solve_for_y_na(hi, kappa_target)
        if res_lo is None or res_hi is None:
            converged_all = False
            message = f"No convergió al evaluar los extremos de la bisección de y_na en kappa={kappa_target:.4g}."
            break
        _, axial_lo, _ = res_lo
        _, axial_hi, _ = res_hi
        if tol_axial is None:
            tol_axial = 1e-3 * max(abs(axial_lo), abs(axial_hi), 1.0)
        if abs(axial_lo) < tol_axial:
            y_na_current, (last, axial, moment) = lo, res_lo
        elif abs(axial_hi) < tol_axial:
            y_na_current, (last, axial, moment) = hi, res_hi
        elif (axial_lo > 0) == (axial_hi > 0):
            converged_all = False
            message = (
                f"La fuerza axial no cambia de signo en y_na∈[0,{height:.4g}] en "
                f"kappa={kappa_target:.4g} (N(0)={axial_lo:.4g}, N({height:.4g})={axial_hi:.4g}): "
                "no se puede acotar la búsqueda de flexión pura en este rango."
            )
            break
        else:
            mid, res_mid = None, None
            for _ in range(max_bisection):
                mid = 0.5 * (lo + hi)
                res_mid = solve_for_y_na(mid, kappa_target)
                if res_mid is None:
                    converged_all = False
                    message = f"No convergió durante la bisección de y_na en kappa={kappa_target:.4g}."
                    break
                _, axial_mid, _ = res_mid
                if abs(axial_mid) < tol_axial:
                    break
                if (axial_mid > 0) == (axial_lo > 0):
                    lo, axial_lo = mid, axial_mid
                else:
                    hi, axial_hi = mid, axial_mid
            if not converged_all:
                break
            y_na_current, (last, axial, moment) = mid, res_mid

        u, states = last.u, last.states
        kappa_hist.append(kappa_target)
        moment_hist.append(moment)
        axial_hist.append(axial)
        y_na_hist.append(y_na_current)
        if progress_cb is not None:
            progress_cb(step_idx - 1, 0, abs(axial), kappa_target / kappa_max)

    return MomentCurvatureResult(
        kappa=np.array(kappa_hist),
        moment=np.array(moment_hist),
        axial=np.array(axial_hist),
        y_na=y_na_hist[-1] if y_na_hist else float("nan"),
        y_na_history=np.array(y_na_hist) if y_na_hist else None,
        converged=converged_all,
        message=message,
        final_states=states if kappa_hist else None,
    )
