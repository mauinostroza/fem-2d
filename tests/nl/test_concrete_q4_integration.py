"""Sesión S4 del plan: `ConcreteCDP` funcionando dentro de un elemento
Q4 real, a través del solver no lineal completo (no solo a nivel de
punto material como en `test_concrete_cdp.py`).

Nota de alcance honesta: un test de "objetividad de malla" cuantitativo
(comparar la energía disipada entre varias mallas, o contra la energía
de fractura analítica G_F·área) se intentó en esta sesión y se descartó
por ahora. Se confirmó que:
- Una barra perfectamente uniforme sin imperfección no tiene un patrón
  de localización bien definido (todos los elementos intentan fisurar
  a la vez); es el error clásico de este tipo de test, documentado en
  la propia literatura de crack-band — hace falta una imperfección
  explícita para forzar la localización.
- Incluso con una imperfección, comparar el área bajo la curva
  carga-desplazamiento GLOBAL contra G_F·área no funciona directamente:
  esa área incluye la energía elástica (recuperable) del resto de la
  estructura, no solo la energía disipada por la fisura. Una
  verificación correcta necesita extraer la energía disipada de las
  variables de estado del material (daño/deformación plástica), no
  inferirla del área bajo la curva global — trabajo pendiente, más
  apropiado para `postprocess.py` (sesión S5).
- Empujar cualquier malla a daño casi totalmente saturado (rama de
  ablandamiento casi completa) es numéricamente muy exigente incluso
  con line search y cutback (matriz tangente casi singular); requiere
  arc-length (sesión S7, no implementada todavía) para trazarse con
  robustez.

Lo que este archivo sí valida con confianza: el modelo CDP conectado a
través de la cadena completa (elemento Q4 → ensamblaje → solver
Newton-Raphson con line search/cutback) da una respuesta físicamente
razonable (endurece elástico, ablanda tras fisurar, converge) en un
caso de tracción y uno de compresión moderados.
"""

import numpy as np
import pytest

from fem2d_nl.bc import dof_index, fixed_dofs_from_nodes
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.model import ElementGroup, Model
from fem2d_nl.solver.controls import DisplacementControl
from fem2d_nl.solver.newton import SolverOptions, nonlinear_solve

LENGTH = 100.0
HEIGHT = 100.0
FCK = 25.0


def _single_element_model(l_ch=LENGTH):
    nodes = np.array([[0.0, 0.0], [LENGTH, 0.0], [LENGTH, HEIGHT], [0.0, HEIGHT]])
    quads = np.array([[0, 1, 2, 3]])
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK)
    cdp = ConcreteCDP(params, char_length=l_ch)
    group = ElementGroup(kind="quad4", connectivity=quads, material=cdp, elem_kwargs={"thickness": 1.0})
    return Model(nodes=nodes, groups=[group]), cdp


def test_q4_with_cdp_tension_softens_without_crashing():
    model, cdp = _single_element_model()
    fixed = fixed_dofs_from_nodes([0, 3], components=(0,))
    fixed = np.union1d(fixed, fixed_dofs_from_nodes([0, 1], components=(1,)))
    prescribed = np.array([dof_index(1, 0), dof_index(2, 0)])
    eps_target = 0.002  # bien dentro de la rama de ablandamiento, ver módulo CDP
    prescribed_values = np.full(2, eps_target * LENGTH)

    control = DisplacementControl(fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=prescribed_values)
    result = nonlinear_solve(model, control, SolverOptions(n_steps=20, max_iterations=50))

    assert result.converged
    forces = np.array([s.f_int[prescribed].sum() for s in result.steps])
    assert forces.max() > 0.0
    # la fuerza debe caer sustancialmente respecto del pico (ablandamiento real)
    assert forces[-1] < 0.5 * forces.max()


def test_q4_with_cdp_compression_hardens_toward_fcm():
    model, cdp = _single_element_model()
    fixed = fixed_dofs_from_nodes([0, 3], components=(0,))
    fixed = np.union1d(fixed, fixed_dofs_from_nodes([0, 1], components=(1,)))
    prescribed = np.array([dof_index(1, 0), dof_index(2, 0)])
    eps_target = -0.0015  # cerca del pico de compresión
    prescribed_values = np.full(2, eps_target * LENGTH)

    control = DisplacementControl(fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=prescribed_values)
    result = nonlinear_solve(model, control, SolverOptions(n_steps=20, max_iterations=50))

    assert result.converged
    forces = np.array([s.f_int[prescribed].sum() for s in result.steps])
    peak_stress = abs(forces.min()) / HEIGHT  # thickness=1
    assert peak_stress == pytest.approx(cdp.fcm, rel=0.15)


def test_material_convergence_failure_triggers_cutback_not_crash():
    """Un solo paso enorme (bien más allá de la ductilidad razonable del
    material) puede hacer fallar el return mapping local. El solver debe
    recuperarse por cutback (mismo mecanismo que una no-convergencia
    global ordinaria), no propagar la excepción y morir."""
    model, cdp = _single_element_model()
    fixed = fixed_dofs_from_nodes([0, 3], components=(0,))
    fixed = np.union1d(fixed, fixed_dofs_from_nodes([0, 1], components=(1,)))
    prescribed = np.array([dof_index(1, 0), dof_index(2, 0)])
    prescribed_values = np.full(2, 0.003 * LENGTH)

    # un solo paso (n_steps=1) fuerza un salto de deformación grande de
    # una sola vez, el escenario más propenso a un fallo de convergencia
    # local que en sesiones anteriores habría propagado la excepción.
    control = DisplacementControl(fixed_dofs=fixed, prescribed_dofs=prescribed, prescribed_values=prescribed_values)
    result = nonlinear_solve(model, control, SolverOptions(n_steps=1, max_iterations=30, max_cutbacks=10))

    # no debe lanzar excepción (era el bug real antes de esta sesión);
    # converja o no del todo, debe haber avanzado con pasos más chicos
    assert len(result.steps) > 0
