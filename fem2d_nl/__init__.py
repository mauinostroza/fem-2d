"""fem2d_nl: solver FEM 2D no lineal (hormigón armado) para fem-2d.

Independiente del módulo lineal `fem2d` (placa con agujero, SolidsPy).
No comparte solver ni malla con él; solo hereda la jerarquía de
excepciones para que la UI maneje errores de forma uniforme.
"""
