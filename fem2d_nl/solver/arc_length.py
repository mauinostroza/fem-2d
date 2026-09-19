"""Control de arco (arc-length, Crisfield) para trazar la curva P-δ más
allá de un pico de carga, donde el control de carga puro
(`solver.controls.LoadControl`) diverge (tangente horizontal/negativa).

Formulación general de Crisfield (esférica o cilíndrica según `psi`, ver
más abajo), Newton-Raphson completo (se re-resuelve el sistema tangente en
cada iteración del corrector, no solo en el predictor). En cada
incremento de longitud de arco `ΔL`:

- **Predictor**: `Δu_t = K_t⁻¹·q` (dirección tangente, `q` = patrón de
  carga de referencia); `Δλ` del predictor se obtiene de la propia
  restricción de arco (con residuo cero al iniciar el incremento), con el
  signo elegido para continuar en la misma dirección que el incremento
  total del paso anterior (o positivo en el primer paso).
- **Corrector**: en cada iteración, `du_bar = K_t⁻¹·R` (R = residuo a
  `λ` actual) y `du_t = K_t⁻¹·q` (misma tangente); se combinan
  `Δu_nuevo = Δu + du_bar + dλ·du_t` eligiendo la raíz `dλ` de la
  restricción de arco (ecuación cuadrática) que mantiene la dirección de
  avance — coseno máximo con el incremento del paso anterior.

**Simplificación deliberada**: se usa la forma *cilíndrica* por defecto
(`psi=0.0`, ignora la contribución de `Δλ` a la restricción de longitud de
arco, solo usa `‖Δu‖²=ΔL²`) en vez de la esférica completa (`psi=1`) —
evita tener que adimensionalizar `Δλ²·‖q‖²` contra `‖Δu‖²` cuando carga y
desplazamiento tienen unidades y órdenes de magnitud muy distintos (N vs
mm en este proyecto), un problema práctico conocido de la forma esférica.
`psi` queda como parámetro por si se quiere la forma completa.

No se integra en `nonlinear_solve`/`_try_step` de `newton.py` (están
construidos alrededor de la interfaz `apply(u,lam)` de
`DisplacementControl`/`LoadControl`, que no encaja con el sistema
*bordered* de dos ecuaciones que necesita arc-length) — función aparte que
reutiliza `_safe_assemble` y las mismas `StepResult`/`NonlinearResult` de
`newton.py` para no romper compatibilidad con el resto del código.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from fem2d_nl.exceptions import ConvergenceError
from fem2d_nl.solver.controls import ArcLengthControl
from fem2d_nl.solver.newton import _safe_assemble
from fem2d_nl.solver.results import NonlinearResult, StepResult


@dataclass(frozen=True)
class ArcLengthOptions:
    n_steps: int = 20            # incrementos de arco objetivo (guía para el ΔL inicial, no un límite estricto)
    target_arc_length: float | None = None  # ΔL inicial; si None, se estima de un predictor a psi=0
    psi: float = 0.0             # 0 = cilíndrico (default, ver docstring del módulo), 1 = esférico completo
    max_iterations: int = 25
    tol_residual: float = 1e-4
    max_arc_cutbacks: int = 6
    max_total_steps: int = 200   # tope duro de incrementos (evita bucles infinitos si nunca converge ni diverge)
    growth_factor: float = 1.3   # crecimiento de ΔL tras convergencia fácil
    cutback_factor: float = 0.5  # reducción de ΔL tras no converger


def _root_select(a_c, b_c, c_c, a_dir, b_dir, du_accum, du_prev):
    """Resuelve `a_c·dλ² + b_c·dλ + c_c = 0` y elige la raíz que mantiene
    la dirección de avance (máximo coseno con `du_prev`, el incremento
    total del paso de arco anterior; si no hay paso anterior, se usa el
    propio `a_dir`+`dλ·b_dir` con signo positivo por defecto)."""
    disc = b_c**2 - 4 * a_c * c_c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    roots = [(-b_c + sqrt_disc) / (2 * a_c), (-b_c - sqrt_disc) / (2 * a_c)]
    if du_prev is None:
        # sin referencia de dirección previa: se prefiere dλ>0 (cargando, no descargando)
        return max(roots)
    best_root, best_cos = None, -np.inf
    for dlam in roots:
        du_candidate = du_accum + a_dir + dlam * b_dir
        cos = float(du_candidate @ du_prev)
        if cos > best_cos:
            best_cos, best_root = cos, dlam
    return best_root


def arc_length_solve(
    model, control: ArcLengthControl, options: ArcLengthOptions | None = None, progress_cb=None
) -> NonlinearResult:
    """Traza la curva de equilibrio (u(λ), λ) con control de arco de
    Crisfield, capaz de continuar más allá de un pico de carga (a
    diferencia de `LoadControl` puro con `newton.nonlinear_solve`, que
    diverge ahí). Se detiene si `λ` sale de `[0, 1+margen]` en la
    dirección de descarga total, si se agota `max_total_steps`, o si un
    incremento no converge tras `max_arc_cutbacks` reducciones de `ΔL`
    (conservando, igual que `nonlinear_solve`, todos los pasos previos ya
    convergidos)."""
    options = options or ArcLengthOptions()
    ndof = model.ndof
    free = control.free_dofs(ndof)
    q = control.force_pattern

    u = np.zeros(ndof)
    lam = 0.0
    states = model.initial_state()
    result = NonlinearResult()

    assembled0 = _safe_assemble(model, u, states, 0.0)
    if assembled0 is None:
        raise ConvergenceError("El modelo no pudo ensamblarse en el estado inicial (u=0).")
    k0 = assembled0[1][np.ix_(free, free)].tocsc()
    try:
        du_t0 = spla.spsolve(k0, q[free])
    except Exception as exc:
        raise ConvergenceError("El sistema tangente inicial es singular; revise apoyos/patrón de carga.") from exc

    delta_l = options.target_arc_length
    if delta_l is None:
        norm_t0 = float(np.linalg.norm(du_t0))
        delta_l = norm_t0 / max(options.n_steps, 1) if norm_t0 > 0 else 1.0
    delta_l_max = delta_l  # el ΔL nunca crece por encima del pedido/estimado inicialmente

    du_prev_total = None  # incremento total (en dofs libres) del último paso de arco convergido
    arc_cutbacks = 0
    total_steps = 0

    while total_steps < options.max_total_steps:
        du_accum = np.zeros(len(free))
        dlam_accum = 0.0

        assembled = _safe_assemble(model, u, states, 0.0)
        if assembled is None:
            ok = False
        else:
            f_int, k_mat, trial_states = assembled
            k_ff = k_mat[np.ix_(free, free)].tocsc()
            try:
                du_t = spla.spsolve(k_ff, q[free])
            except Exception:
                du_t = None

            if du_t is None:
                ok = False
            else:
                denom = float(du_t @ du_t) + options.psi**2 * float(q[free] @ q[free])
                dlam_pred = delta_l / np.sqrt(max(denom, 1e-300))
                if du_prev_total is not None:
                    if float(du_t @ du_prev_total) < 0:
                        dlam_pred = -dlam_pred
                du_accum = dlam_pred * du_t
                dlam_accum = dlam_pred
                ok = True

        if not ok:
            arc_cutbacks += 1
            if arc_cutbacks > options.max_arc_cutbacks:
                result.converged = False
                result.message = "No se pudo evaluar el predictor de arc-length (sistema tangente singular)."
                break
            delta_l *= options.cutback_factor
            continue

        converged_step = False
        iters_used = 0
        trial_states = states
        for it in range(1, options.max_iterations + 1):
            iters_used = it
            u_trial = u.copy()
            u_trial[free] += du_accum
            lam_trial = lam + dlam_accum

            assembled = _safe_assemble(model, u_trial, states, 0.0)
            if assembled is None:
                converged_step = False
                break
            f_int, k_mat, trial_states = assembled
            residual = lam_trial * q - f_int
            res_free = residual[free]
            res_norm = float(np.linalg.norm(res_free))
            ref_norm = max(float(np.linalg.norm(f_int[free])), float(np.linalg.norm((lam_trial * q)[free])), 1e-12)
            if res_norm < options.tol_residual * ref_norm:
                u, lam, states, f_int_converged = u_trial, lam_trial, trial_states, f_int
                converged_step = True
                break

            k_ff = k_mat[np.ix_(free, free)].tocsc()
            try:
                du_bar = spla.spsolve(k_ff, res_free)
                du_t = spla.spsolve(k_ff, q[free])
            except Exception:
                converged_step = False
                break
            if not (np.all(np.isfinite(du_bar)) and np.all(np.isfinite(du_t))):
                converged_step = False
                break

            a_dir = du_accum + du_bar
            a_c = float(du_t @ du_t) + options.psi**2 * float(q[free] @ q[free])
            b_c = 2.0 * (float(a_dir @ du_t) + options.psi**2 * float(q[free] @ q[free]) * dlam_accum)
            c_c = float(a_dir @ a_dir) + options.psi**2 * float(q[free] @ q[free]) * dlam_accum**2 - delta_l**2

            dlam_corr = _root_select(a_c, b_c, c_c, a_dir, du_t, du_accum, du_prev_total)
            if dlam_corr is None:
                converged_step = False
                break

            du_accum = a_dir + dlam_corr * du_t
            dlam_accum += dlam_corr

        if converged_step:
            total_steps += 1
            du_prev_total = du_accum.copy()
            result.steps.append(
                StepResult(
                    lam=lam, u=u.copy(), states=states, iterations=iters_used,
                    residual_norm=0.0, f_int=f_int_converged,
                )
            )
            if progress_cb is not None:
                progress_cb(len(result.steps) - 1, iters_used, 0.0, lam)
            arc_cutbacks = 0
            if iters_used <= max(2, options.max_iterations // 4):
                delta_l = min(delta_l * options.growth_factor, delta_l_max)
            if lam >= 1.0 and (du_prev_total @ q[free]) <= 0:
                # ya se alcanzó/superó λ=1 y el incremento está descargando: fin natural
                break
        else:
            arc_cutbacks += 1
            if arc_cutbacks > options.max_arc_cutbacks:
                result.converged = False
                result.message = (
                    f"No convergió en λ≈{lam + dlam_accum:.4g} tras {options.max_arc_cutbacks} "
                    "reducciones de ΔL. Se conservan los pasos previos."
                )
                break
            delta_l *= options.cutback_factor

    if not result.steps and not result.converged:
        raise ConvergenceError(result.message or "arc_length_solve no pudo converger en ningún paso.")

    return result
