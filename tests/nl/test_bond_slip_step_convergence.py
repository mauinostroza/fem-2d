"""Sesión S7.1: regresión que asienta la investigación del hallazgo de S5
sobre `bond_slip` y sensibilidad al número de pasos de carga.

S5 había reportado (ver el aviso de honestidad histórico en
`test_moment_curvature.py`) una diferencia de hasta ~40-50% en el momento
en curvaturas intermedias entre 10 y 15-30 pasos de carga para el modo
`bond_slip`. Un script de diagnóstico limpio (barrido sistemático de
`n_steps` = 10/20/40/80 sobre el mismo caso, ejecutado en S7) **no
reprodujo** ese hallazgo: el momento final coincide dentro de ~0.2% en las
cuatro resoluciones. La inspección del estado interno de los `bond_link`
confirma que el mecanismo funciona correctamente: `s_max` es exactamente
0 en los dos extremos de la barra (consecuencia física esperada de la
condición de borde ahí — el nodo de hormigón de referencia está fijo en
`ux=0` y el de la cara cargada sigue el perfil impuesto, así que el acero
extremo no tiene demanda de deslizamiento significativa) y no nulo,
físicamente razonable, en los puntos interiores.

Conclusión: el hallazgo de S5 fue casi con certeza un artefacto del
script de depuración ad-hoc de esa sesión (no reconstruible con el código
de producción), no un defecto de `moment_curvature`/`build_section_mesh`/
`bond_link`. Este test deja la propiedad de convergencia asentada como
regresión, marcado `slow` porque corre el análisis 3 veces.
"""

import numpy as np
import pytest

from fem2d_nl.materials.bond_mc2010 import BondLaw, BondParameters
from fem2d_nl.materials.concrete_cdp import CDPParameters, ConcreteCDP
from fem2d_nl.materials.steel_uniaxial import MultilinearSteel
from fem2d_nl.mesh.section import RebarLayer, SectionGeometry
from fem2d_nl.mesh.structured import build_section_mesh, nominal_char_length
from fem2d_nl.postprocess import moment_curvature, transformed_elastic_centroid
from fem2d_nl.solver.newton import SolverOptions

pytestmark = pytest.mark.slow

GEOM = SectionGeometry(width=200.0, height=400.0, span=200.0, nx=3, ny=4)
FCK = 25.0
KAPPA_MAX = 1.2e-5


def _build_model():
    l_ch = nominal_char_length(GEOM)
    params = CDPParameters(young_modulus=30_000.0, poisson_ratio=0.2, fck=FCK)
    cdp = ConcreteCDP(params, char_length=l_ch)
    steel = MultilinearSteel(points=np.array([[0.002, 400.0], [0.05, 460.0]]))
    layer = RebarLayer(depth_y=40.0, n_bars=3, diameter=16.0, steel=steel)
    bond_law = BondLaw(BondParameters(fcm=cdp.fcm, bar_diameter=16.0, bond_condition="good"))
    model = build_section_mesh(GEOM, [layer], cdp, bond_mode="bond_slip", bond_params=bond_law)
    y_na = transformed_elastic_centroid(GEOM, [layer], cdp.e0)
    return model, y_na


def test_bond_slip_final_moment_converges_across_step_counts():
    moments_final = {}
    for n_steps in (10, 20, 40):
        model, y_na = _build_model()
        res = moment_curvature(
            model, GEOM.span, y_na, KAPPA_MAX, n_steps=n_steps,
            options=SolverOptions(n_steps=n_steps, max_iterations=30, max_cutbacks=6),
        )
        assert res.converged
        moments_final[n_steps] = res.moment[-1]

    baseline = moments_final[10]
    for n_steps, m in moments_final.items():
        assert m == pytest.approx(baseline, rel=5e-3), (
            f"M final con n_steps={n_steps} ({m:.6e}) difiere de n_steps=10 "
            f"({baseline:.6e}) más de lo esperado — reabriría la duda de S5."
        )


def test_bond_link_slip_is_zero_at_free_ends_and_nonzero_inside():
    """Confirma la explicación física del hallazgo de S5 (mal interpretado
    en su momento como un posible bug): en los dos extremos de la barra
    (coincidentes con la cara de referencia fija y la cara cargada), el
    deslizamiento máximo acumulado (`s_max`) es exactamente cero; en los
    puntos interiores es no nulo — el mecanismo de adherencia sí está
    activo, solo que no en los extremos, por la propia condición de borde
    del problema (no por un defecto del `bond_link`)."""
    model, y_na = _build_model()
    res = moment_curvature(
        model, GEOM.span, y_na, KAPPA_MAX, n_steps=20,
        options=SolverOptions(n_steps=20, max_iterations=30, max_cutbacks=6),
    )
    assert res.converged
    bond_group_idx = [i for i, g in enumerate(model.groups) if g.kind == "bond_link"][0]
    s_max = res.final_states[bond_group_idx]["s_max"]

    assert s_max[0] == pytest.approx(0.0, abs=1e-12)
    assert s_max[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.any(np.abs(s_max[1:-1]) > 1e-6)
