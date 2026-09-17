"""Solver Newton-Raphson con pasos de carga/desplazamiento, line search y
cutback.

Cascada de recuperación ante no-convergencia: line search (backtracking
simple) dentro de cada iteración → cutback de paso (bisección de Δλ) en
`nonlinear_solve`. La subincrementación LOCAL del return mapping de un
material (p. ej. `concrete_cdp.ConcreteCDP`) vive dentro del propio
material; si ese material no logra converger igual, lanza
`MaterialModelError`, que aquí se trata como un fallo de iteración más
(dispara cutback del paso global), no como un crash del solver.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from fem2d_nl.assembly import assemble
from fem2d_nl.exceptions import ConvergenceError, MaterialModelError
from fem2d_nl.solver.results import NonlinearResult, StepResult


@dataclass(frozen=True)
class SolverOptions:
    n_steps: int = 10
    max_iterations: int = 25
    tol_residual: float = 1e-4
    tol_energy: float = 1e-6
    max_cutbacks: int = 6
    line_search: bool = True
    line_search_factors: tuple[float, ...] = (0.5, 0.25, 0.1)


def _safe_assemble(model, u, states, dt):
    """`assemble()`, pero convierte un fallo de material (return mapping
    local no convergido) en `None` en vez de propagar la excepción: el
    llamador lo trata igual que cualquier otro fallo de iteración."""
    try:
        return assemble(model, u, states, dt)
    except MaterialModelError:
        return None


def _try_step(model, control, u_prev, states, lam, dt, options: SolverOptions):
    """Intenta converger un único paso hasta `lam`. Devuelve
    (u, new_states, f_int, iterations, residual_norm, converged).

    `f_int` corresponde exactamente al `u` devuelto (se assemblea al
    principio de cada iteración y se comprueba convergencia con ese mismo
    valor, antes de calcular un nuevo incremento) — así el llamador nunca
    necesita volver a ensamblar fuera del solver para leer reacciones.
    """
    ndof = model.ndof
    free = control.free_dofs(ndof)
    u = control.apply(u_prev.copy(), lam)
    f_ext = control.external_force(ndof, lam)

    ref_norm = None
    energy_0 = None
    prev_energy = None
    trial_states = states
    f_int = np.zeros(ndof)

    for it in range(1, options.max_iterations + 1):
        assembled = _safe_assemble(model, u, states, dt)
        if assembled is None:
            return u, trial_states, f_int, it, float("inf"), False
        f_int, k_mat, trial_states = assembled
        residual = f_ext - f_int
        res_free = residual[free]
        res_norm = float(np.linalg.norm(res_free))

        if ref_norm is None:
            ref_norm = max(float(np.linalg.norm(f_int[free])), float(np.linalg.norm(f_ext[free])), 1e-12)
        residual_ok = res_norm < options.tol_residual * ref_norm
        energy_ok = prev_energy is not None and abs(prev_energy) < options.tol_energy * energy_0
        if residual_ok or energy_ok:
            return u, trial_states, f_int, it, res_norm, True

        k_ff = k_mat[np.ix_(free, free)].tocsc()
        try:
            delta = spla.spsolve(k_ff, res_free)
        except Exception:
            return u, trial_states, f_int, it, res_norm, False
        if not np.all(np.isfinite(delta)):
            return u, trial_states, f_int, it, res_norm, False

        energy = float(delta @ res_free)
        if energy_0 is None:
            energy_0 = max(abs(energy), 1e-30)
        prev_energy = energy

        step_scale = 1.0
        if options.line_search:
            u_full = u.copy()
            u_full[free] += delta
            assembled_full = _safe_assemble(model, u_full, states, dt)
            full_norm = (
                float("inf")
                if assembled_full is None
                else float(np.linalg.norm((f_ext - assembled_full[0])[free]))
            )
            if full_norm > res_norm:
                # el paso completo empeoró (o falló); retroceder con
                # factores fijos y quedarse con el mejor evaluado
                best_norm, best_scale = full_norm, 1.0
                for scale in options.line_search_factors:
                    u_try = u.copy()
                    u_try[free] += scale * delta
                    assembled_try = _safe_assemble(model, u_try, states, dt)
                    if assembled_try is None:
                        continue
                    try_norm = float(np.linalg.norm((f_ext - assembled_try[0])[free]))
                    if try_norm < best_norm:
                        best_norm, best_scale = try_norm, scale
                step_scale = best_scale
                prev_energy = step_scale * energy  # aproximación consistente con el paso reducido

        u[free] += step_scale * delta

    return u, trial_states, f_int, options.max_iterations, float("inf"), False


def nonlinear_solve(model, control, options: SolverOptions | None = None, progress_cb=None) -> NonlinearResult:
    """Resuelve el modelo con pasos de carga/desplazamiento crecientes de
    0 a 1, con cutback (bisección del incremento) ante no convergencia.

    Si un paso no converge tras agotar los cutbacks, se detiene y se
    devuelve `NonlinearResult(converged=False, ...)` conservando todos los
    pasos previos ya convergidos: un fallo en ablandamiento es información
    física válida, no debe descartar resultados previos.
    """
    options = options or SolverOptions()
    ndof = model.ndof
    u = np.zeros(ndof)
    states = model.initial_state()
    result = NonlinearResult()

    lam = 0.0
    d_lam_initial = 1.0 / options.n_steps
    d_lam = d_lam_initial
    step_idx = 0
    cutbacks = 0
    good_streak = 0

    while lam < 1.0 - 1e-12:
        lam_target = min(lam + d_lam, 1.0)
        u_new, new_states, f_int, iters, res_norm, ok = _try_step(
            model, control, u, states, lam_target, dt=1.0, options=options
        )

        if ok:
            lam = lam_target
            u = u_new
            states = new_states
            result.steps.append(
                StepResult(
                    lam=lam, u=u.copy(), states=states, iterations=iters,
                    residual_norm=res_norm, f_int=f_int,
                )
            )
            if progress_cb is not None:
                progress_cb(step_idx, iters, res_norm, lam)
            step_idx += 1
            cutbacks = 0
            # recuperación del tamaño de paso: si viene convergiendo fácil
            # (pocas iteraciones) dos veces seguidas, duplicar d_lam de
            # nuevo (acotado al tamaño inicial). Sin esto, un solo recorte
            # de paso deja el solver permanentemente atascado en pasos
            # microscópicos por el resto del análisis, aunque la
            # dificultad que lo causó ya haya quedado atrás.
            if iters <= max(2, options.max_iterations // 4):
                good_streak += 1
                if good_streak >= 2:
                    d_lam = min(d_lam * 2.0, d_lam_initial)
                    good_streak = 0
            else:
                good_streak = 0
        else:
            cutbacks += 1
            if cutbacks > options.max_cutbacks:
                result.converged = False
                result.message = (
                    f"No convergió en lam={lam_target:.4g} tras {options.max_cutbacks} "
                    "reducciones de paso. Se conservan los pasos previos."
                )
                break
            d_lam /= 2.0

    if not result.steps and not result.converged:
        raise ConvergenceError(result.message or "El solver no pudo converger en ningún paso.")

    return result
