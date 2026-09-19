"""Datos de entrada para una sección de hormigón armado (dataclasses puras,
sin lógica de malla — ver `mesh/structured.py` para eso).

Convención geométrica (ver `structured.py`): el modelo FEM es una franja
(elevación) de una viga en el plano x-y de `fem2d_nl.model.Model`:
`x` es la dirección del eje de la viga (0..span), `y` es la altura de la
sección (0..height, con y=0 en la fibra inferior), y `width` (breadth de
la sección) se usa como espesor fuera de plano de los elementos `quad4`.
"""

from dataclasses import dataclass

from fem2d_nl.exceptions import GeometryError


@dataclass(frozen=True)
class SectionGeometry:
    width: float   # ancho/breadth de la sección (espesor fuera de plano)
    height: float  # altura de la sección, y=0 fibra inferior
    span: float    # longitud de la franja analizada, a lo largo de x
    nx: int        # elementos a lo largo de x (span)
    ny: int        # elementos a lo largo de y (height)

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.span <= 0:
            raise GeometryError("width, height y span deben ser positivos.")
        if self.nx < 1 or self.ny < 1:
            raise GeometryError("nx y ny deben ser >= 1.")


@dataclass(frozen=True)
class RebarLayer:
    depth_y: float   # altura y de la capa (0 = fibra inferior)
    n_bars: int      # número de barras que representa la capa (se colapsan en una línea)
    diameter: float  # diámetro de cada barra
    steel: object    # MultilinearSteel (o cualquier material con el mismo contrato 1D)

    def __post_init__(self):
        if self.n_bars < 1:
            raise GeometryError("n_bars debe ser >= 1.")
        if self.diameter <= 0:
            raise GeometryError("diameter debe ser positivo.")

    @property
    def area(self) -> float:
        import math
        return self.n_bars * math.pi * self.diameter**2 / 4.0

    @property
    def perimeter(self) -> float:
        import math
        return self.n_bars * math.pi * self.diameter
