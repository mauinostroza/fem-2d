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

## En desarrollo: modo no lineal (hormigón armado)

Hay un segundo motor de cálculo, `fem2d_nl/`, en construcción para un
modo de análisis nuevo (sección/elemento de hormigón armado con no
linealidad de material) que todavía no está conectado a la interfaz. Es
independiente del módulo lineal de arriba: solver Newton-Raphson propio
(SolidsPy no sirve para esto), con su propio elemento continuo (Q4).

Avance hasta ahora (ver `tests/nl/`):

- Infraestructura no lineal: elemento Q4, ensamblaje, condiciones de
  borde, control de carga/desplazamiento, solver Newton-Raphson con
  cutback — validada con un patch test y contra el módulo lineal
  existente en régimen elástico.
- Barras de refuerzo (`elements/truss2.py`) con acero de plasticidad
  multilineal (`materials/steel_uniaxial.py`, endurecimiento isótropo,
  retorno cerrado sin iteración).
- Adherencia acero-hormigón (`elements/bond_link.py` +
  `materials/bond_mc2010.py`, ley τ-s del fib Model Code 2010),
  validada con un ensayo de arrancamiento (pull-out) completo contra
  la solución independiente de la EDO de adherencia
  (`scipy.integrate.solve_bvp`).
- Modelo de daño-plasticidad del hormigón, Concrete Damaged Plasticity
  tipo Abaqus (`materials/concrete_cdp.py`, Lubliner 1989 / Lee &
  Fenves 1998), con return mapping en espacio de esfuerzo principal.
  Validado a nivel de punto material (sin FEM todavía): la envolvente
  biaxial reproduce fb0/fc0=1.16 y el pico de compresión uniaxial
  reproduce f'cm, ambos dentro del 2% — el chequeo más limpio y
  decisivo de la geometría de la superficie de fluencia. Ver
  `tests/nl/test_concrete_cdp.py` para el detalle y las limitaciones
  conocidas (abajo).

Pendiente: integrar el CDP dentro del elemento Q4 (hoy solo corre a
nivel de punto material, con su propio return mapping pero sin la
regularización crack-band ni la robustez de solver que necesita para
mallas reales), la malla de sección con barras embebidas, y la interfaz
de usuario. El plan completo (formulación, arquitectura, riesgos) está
en el historial de la sesión de desarrollo, no versionado en el repo.

**Avisos de honestidad técnica (no ocultar antes de usar en producción)**:

- Algunos valores numéricos por defecto de la ley de adherencia
  (`materials/bond_mc2010.py`, Tabla 6.1-1 del fib Model Code 2010) no
  pudieron verificarse contra el texto oficial del código durante el
  desarrollo (documentado en el docstring del módulo); la forma de la
  curva es correcta, pero cotejar esos números antes de un cálculo de
  producción.
- El modelo CDP reproduce el UMBRAL de fluencia en tracción exactamente
  en `ft` (verificado analítica y numéricamente), pero el PICO nominal
  de la curva de tracción uniaxial puede superar `ft` en un ~30-40%
  antes de ablandar (ver docstring de `concrete_cdp.py` y
  `test_tension_softens_toward_zero_eventually`): es una consecuencia de
  cómo el esfuerzo "efectivo" σ̄_t crece con el daño de Birtel & Mark en
  esta calibración concreta, no un pico físico esperado en un hormigón
  real (que es mucho más frágil). No se pudo resolver con confianza sin
  acceso al PDF original de Lee & Fenves para verificar si falta algún
  término de normalización. La compresión (pico, envolvente biaxial) no
  tiene este problema y está validada con precisión.
- La curva de compresión usa una deformación de pico (`eps_c1`)
  aproximada por fórmula genérica, no la tabla por clase de resistencia
  del MC2010 (no verificable sin el texto original), y el ablandamiento
  post-pico es una rama lineal simplificada calibrada solo para disipar
  la energía de fractura dada — no una curva normativa.
