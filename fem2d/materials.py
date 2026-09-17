"""Propiedades de material y presets comunes.

Unidades del proyecto: longitud en mm, fuerza en N, esfuerzo y módulo de
Young en MPa (N/mm^2).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialProps:
    young_modulus: float  # MPa
    poisson_ratio: float


MATERIAL_PRESETS: dict[str, MaterialProps | None] = {
    "Acero estructural": MaterialProps(young_modulus=200_000.0, poisson_ratio=0.30),
    "Aluminio": MaterialProps(young_modulus=70_000.0, poisson_ratio=0.33),
    "Personalizado": None,
}
