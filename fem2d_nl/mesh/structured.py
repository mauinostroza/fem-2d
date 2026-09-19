"""Malla estructurada para una franja de sección de hormigón armado, en
NumPy puro (sin gmsh): la coincidencia nodo a nodo entre hormigón y acero
(necesaria tanto en modo de adherencia perfecta como para los links de
adherencia) queda garantizada por construcción, y evita el problema de
threading de gmsh/Streamlit que ya se resolvió para el modo lineal (no
hace falta resolverlo de nuevo aquí, porque no se necesita).

Convención (ver `mesh/section.py`): grilla rectangular `nx+1` columnas en
`x` (0..span) por `ny+1` filas en `y` (0..height, y=0 fibra inferior),
`quad4` de hormigón con conectividad antihoraria estándar.

Aviso de honestidad técnica: en vez de INSERTAR filas adicionales para
cada capa de armadura (lo que distorsionaría el tamaño de elemento
asumido por `char_length` del material de hormigón, que es un escalar
fijado al construir `ConcreteCDP`), cada capa se ubica en la fila de
malla ya existente más cercana a `depth_y` ("snapping"): la fila se
desplaza a la altura exacta pedida. Esto mantiene el número de filas y el
tamaño de elemento nominal (`nominal_char_length`) prácticamente
constante, a costa de una distorsión geométrica leve (a lo sumo media
altura de fila) entre la posición pedida y la real. Si dos capas caen en
la misma fila más cercana, se lanza un error pidiendo aumentar `ny` o
separar las capas.
"""

import math

import numpy as np

from fem2d_nl.elements import quad4
from fem2d_nl.exceptions import GeometryError, MaterialModelError
from fem2d_nl.mesh.section import RebarLayer, SectionGeometry
from fem2d_nl.model import ElementGroup, Model


def nominal_char_length(geom: SectionGeometry) -> float:
    """Longitud característica de crack-band para un elemento `quad4` de
    tamaño nominal de esta malla (`span/nx` x `height/ny`), a pasar al
    construir el material de hormigón (`ConcreteCDP(params, char_length=...)`)
    ANTES de llamar a `build_section_mesh` — la malla no construye el
    material, solo lo usa."""
    area = (geom.span / geom.nx) * (geom.height / geom.ny)
    return math.sqrt(area / quad4.N_GAUSS)


def _snap_rows(geom: SectionGeometry, rebar_layers: list[RebarLayer]) -> np.ndarray:
    y_rows = np.linspace(0.0, geom.height, geom.ny + 1)
    used_rows: dict[int, float] = {}
    for layer in rebar_layers:
        if not (0.0 <= layer.depth_y <= geom.height):
            raise GeometryError(
                f"depth_y={layer.depth_y} fuera de la sección (0..{geom.height})."
            )
        idx = int(np.argmin(np.abs(y_rows - layer.depth_y)))
        if idx in used_rows and used_rows[idx] != layer.depth_y:
            raise GeometryError(
                f"Dos capas de armadura (y={used_rows[idx]} e y={layer.depth_y}) caen en "
                f"la misma fila de malla; aumente ny o separe las capas."
            )
        used_rows[idx] = layer.depth_y
        y_rows[idx] = layer.depth_y
    return y_rows


def build_section_mesh(
    geom: SectionGeometry,
    rebar_layers: list[RebarLayer],
    concrete_material,
    bond_mode: str = "bond_slip",
    bond_params=None,
) -> Model:
    """Construye el `Model` completo (hormigón + capas de armadura) para
    una franja de sección.

    `bond_mode`:
    - `"perfect"`: adherencia perfecta, las barras comparten directamente
      los nodos de hormigón de su fila (sin deslizamiento posible).
    - `"bond_slip"`: se duplican los nodos de acero en cada fila de
      armadura y se conectan a los nodos de hormigón coincidentes con
      elementos `bond_link` (ley τ-s de `bond_params`, uno por capa o
      compartido entre todas).

    `concrete_material` ya debe estar construido (con su `char_length`
    coherente con esta malla, ver `nominal_char_length`); esta función
    solo arma la malla, no construye materiales.

    `bond_params` (solo si `bond_mode="bond_slip"`): una `BondLaw` ya
    construida, compartida por todas las capas, o un `dict[float, BondLaw]`
    que mapea `depth_y` -> `BondLaw` para dar una ley distinta por capa.
    """
    if bond_mode not in ("perfect", "bond_slip"):
        raise MaterialModelError(f"bond_mode debe ser 'perfect' o 'bond_slip', no {bond_mode!r}.")
    if bond_mode == "bond_slip" and rebar_layers and bond_params is None:
        raise MaterialModelError("bond_mode='bond_slip' requiere bond_params.")

    n_cols = geom.nx + 1
    n_rows_ = geom.ny + 1
    x_cols = np.linspace(0.0, geom.span, n_cols)
    y_rows = _snap_rows(geom, rebar_layers)
    row_of_depth = {layer.depth_y: int(np.argmin(np.abs(y_rows - layer.depth_y))) for layer in rebar_layers}

    nodes = np.empty((n_rows_ * n_cols, 2))
    for iy, y in enumerate(y_rows):
        for ix, x in enumerate(x_cols):
            nodes[iy * n_cols + ix] = (x, y)

    quads = []
    for iy in range(geom.ny):
        for ix in range(geom.nx):
            n00 = iy * n_cols + ix
            n10 = iy * n_cols + ix + 1
            n11 = (iy + 1) * n_cols + ix + 1
            n01 = (iy + 1) * n_cols + ix
            quads.append([n00, n10, n11, n01])
    concrete_group = ElementGroup(
        kind="quad4",
        connectivity=np.array(quads, dtype=int),
        material=concrete_material,
        elem_kwargs={"thickness": geom.width},
    )

    groups = [concrete_group]
    node_list = [nodes]

    for layer in rebar_layers:
        iy = row_of_depth[layer.depth_y]
        concrete_row = [iy * n_cols + ix for ix in range(n_cols)]

        if bond_mode == "perfect":
            steel_row = concrete_row
        else:
            base = sum(arr.shape[0] for arr in node_list)
            steel_row = [base + ix for ix in range(n_cols)]
            steel_coords = np.array([(x_cols[ix], y_rows[iy]) for ix in range(n_cols)])
            node_list.append(steel_coords)

        truss_conn = np.array([[steel_row[ix], steel_row[ix + 1]] for ix in range(geom.nx)], dtype=int)
        groups.append(
            ElementGroup(
                kind="truss2",
                connectivity=truss_conn,
                material=layer.steel,
                elem_kwargs={"area": layer.area},
            )
        )

        if bond_mode == "bond_slip":
            trib = np.empty(n_cols)
            trib[0] = (x_cols[1] - x_cols[0]) / 2.0
            trib[-1] = (x_cols[-1] - x_cols[-2]) / 2.0
            trib[1:-1] = (x_cols[2:] - x_cols[:-2]) / 2.0
            e_steel = getattr(layer.steel, "e_resolved", None)
            if e_steel is None:
                raise MaterialModelError(
                    "bond_mode='bond_slip' requiere que el material de la capa exponga "
                    "`e_resolved` (p. ej. MultilinearSteel) para dimensionar la rigidez normal del link."
                )
            bond_conn = np.array(
                [[steel_row[ix], concrete_row[ix]] for ix in range(n_cols)], dtype=int
            )
            k_normal = e_steel * layer.area / trib
            bond_law = bond_params[layer.depth_y] if isinstance(bond_params, dict) else bond_params
            groups.append(
                ElementGroup(
                    kind="bond_link",
                    connectivity=bond_conn,
                    material=bond_law,
                    elem_kwargs={
                        "perimeter": layer.perimeter,
                        "trib_length": trib,
                        "k_normal": k_normal,
                        "axis": (1.0, 0.0),
                    },
                )
            )

    all_nodes = np.concatenate(node_list, axis=0)
    return Model(nodes=all_nodes, groups=groups)
