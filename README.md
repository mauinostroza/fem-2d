# fem-2d — Análisis FEM de placa con agujero

Interfaz local para navegador, basada en [SolidsPy](https://github.com/AppliedMechanics-EAFIT/SolidsPy),
para analizar una placa rectangular con un agujero circular central
sometida a cargas en sus bordes. Pensada para evaluar rápidamente la
concentración de esfuerzos alrededor del agujero (resistencia de borde).

Permite:

- Generar la geometría y la malla (largo, alto, radio del agujero y
  tamaño de elemento) con refinamiento automático cerca del agujero.
- Definir condiciones de apoyo y cargas de forma independiente en cada
  uno de los 4 bordes de la placa.
- Ver la malla, la deformada y el contorno de esfuerzos de von Mises,
  con el punto de esfuerzo máximo resaltado y, opcionalmente, un factor
  de utilización respecto a un esfuerzo admisible.

## Alcance y limitaciones

- Elasticidad lineal 2D, formulación de tensión plana con espesor
  unitario (una "carga" en MPa equivale a una fuerza distribuida en
  N/mm de borde).
- Material isótropo lineal (sin plasticidad ni fatiga).
- Geometría: solo rectángulo con un agujero circular centrado.

## Instalación

Requiere Python 3.10+ y, en Linux, la librería del sistema `libGLU`
(usada internamente por gmsh, incluso en modo sin interfaz gráfica):

```bash
sudo apt-get install libglu1-mesa   # Debian/Ubuntu
```

Luego:

```bash
python -m venv .venv
source .venv/bin/activate
./install.sh
```

`install.sh` instala las dependencias en dos pasos porque
`solidspy==1.1.0.post1` declara en su empaquetado una dependencia
obsoleta (`meshio==3.0`) que en realidad nunca usa en su código; ese pin
choca con la versión de `meshio` que necesita `pygmsh`. Por eso
`solidspy` se instala aparte con `--no-deps`, después de sus
dependencias reales (`numpy`, `scipy`, `matplotlib`, `easygui`), que sí
están en `requirements.txt`.

## Ejecución

```bash
streamlit run app.py
```

Abre automáticamente `http://localhost:8501` en el navegador.

## Nota técnica: bug de SolidsPy en la recuperación de esfuerzos

La versión publicada de SolidsPy (`1.1.0.post1`) tiene un bug confirmado
en `solidspy.postprocesor.strain_nodes` para elementos triangulares
(tipo 3): su función interna `str_el3` reserva espacio para 3 puntos de
evaluación por elemento, pero solo calcula 1 (correcto para un triángulo
de deformación constante), dejando los otros 2 en cero. El resultado es
que dos de cada tres nodos de cada elemento reciben esfuerzo cero.

Por eso `fem2d/solver.py` no usa esa función: calcula el esfuerzo en
cada elemento con la fórmula estándar del triángulo de deformación
constante (CST) y promedia entre los elementos que comparten cada nodo,
usando únicamente los desplazamientos que devuelve SolidsPy (esos sí son
correctos). Ver `fem2d/solver.py:compute_nodal_stress` para el detalle.

## Tests

```bash
pytest tests/
```

Incluyen una validación física: para una placa con un agujero pequeño
respecto a su tamaño, el esfuerzo en el borde del agujero (en el punto
perpendicular a la carga) debe acercarse al factor de concentración
teórico de Kirsch (Kt ≈ 3).

## Estructura del proyecto

```
app.py                        # interfaz Streamlit
fem2d/
├── exceptions.py             # errores de dominio (mensajes accionables en la UI)
├── materials.py               # presets de material
├── geometry.py                # malla (pygmsh) y validaciones
├── boundary_conditions.py     # bordes -> condiciones/cargas para SolidsPy
├── solver.py                  # integración con SolidsPy y cálculo de esfuerzos
└── visualization.py           # gráficos matplotlib
tests/                         # pytest, sin dependencia de la UI
```
