"""Solver Newton-Raphson con pasos de carga/desplazamiento y cutback.

Alcance de esta primera versión (material elástico, sesión S1 del plan):
line search y arc-length quedan reservados (`SolverOptions.line_search`
existe pero todavía no se usa) para cuando haya materiales no lineales
reales que lo necesiten (S3/S4). La cascada de recuperación implementada
por ahora es cutback de paso; subincrementación local y viscosidad se
agregan junto con el material CDP.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from fem2d_nl.assembly import assemble
from fem2d_nl.exceptions import ConvergenceError
from fem2d_nl.solver.results import NonlinearResult, StepResult


@dataclass(frozen=True)
class SolverOptions:
    n_steps: int = 10
    max_iterations: int = 25
    tol_residual: float = 1e-4
    tol_energy: float = 1e-6
    max_cutbacks: int = 6
    line_search: bool = False  # no usado todavía (ver docstring del módulo)


def _try_step(model, control, u_prev, states, lam, dt, options: SolverOptions):
    """Intenta converger un único paso hasta `lam`. Devuelve
    (u, new_states, iterations, residual_norm, converged)."""
    ndof = model.ndof
    free = control.free_dofs(ndof)
    u = control.apply(u_prev.copy(), lam)

    ref_norm = None
    energy_0 = None
    trial_states = states
    for it in range(1, options.max_iterations + 1):
        f_int, k_mat, trial_states = assemble(model, u, states, dt)
        f_ext = control.external_force(ndof, lam)
        residual = f_ext - f_int
        res_free = residual[free]
        res_norm = float(np.linalg.norm(res_free))

        if ref_norm is None:
            ref_norm = max(float(np.linalg.norm(f_int[free])), float(np.linalg.norm(f_ext[free])), 1e-12)
        if res_norm < options.tol_residual * ref_norm:
            return u, trial_states, it, res_norm, True

        k_ff = k_mat[np.ix_(free, free)].tocsc()
        try:
            delta = spla.spsolve(k_ff, res_free)
        except Exception:
            return u, trial_states, it, res_norm, False
        if not np.all(np.isfinite(delta)):
            return u, trial_states, it, res_norm, False

        energy = float(delta @ res_free)
        if energy_0 is None:
            energy_0 = max(abs(energy), 1e-30)
        elif abs(energy) < options.tol_energy * energy_0 and res_norm < options.tol_residual * ref_norm * 10:
            u[free] += delta
            return u, trial_states, it, res_norm, True

        u[free] += delta

    return u, trial_states, options.max_iterations, res_norm, False


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
    d_lam = 1.0 / options.n_steps
    step_idx = 0
    cutbacks = 0

    while lam < 1.0 - 1e-12:
        lam_target = min(lam + d_lam, 1.0)
        u_new, new_states, iters, res_norm, ok = _try_step(model, control, u, states, lam_target, dt=1.0, options=options)

        if ok:
            lam = lam_target
            u = u_new
            states = new_states
            result.steps.append(
                StepResult(lam=lam, u=u.copy(), states=states, iterations=iters, residual_norm=res_norm)
            )
            if progress_cb is not None:
                progress_cb(step_idx, iters, res_norm, lam)
            step_idx += 1
            cutbacks = 0
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
