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
app.py                         # router Streamlit: elige entre los 2 modos
ui/
├── linear_plate.py             # UI del modo lineal (placa con agujero)
└── rc_section.py               # UI del modo no lineal (sección de hormigón armado)
fem2d/
├── exceptions.py             # errores de dominio (mensajes accionables en la UI)
├── materials.py               # presets de material
├── geometry.py                # malla (pygmsh) y validaciones
├── boundary_conditions.py     # bordes -> condiciones/cargas para SolidsPy
├── solver.py                  # integración con SolidsPy y cálculo de esfuerzos
└── visualization.py           # gráficos matplotlib
fem2d_nl/                       # motor no lineal (ver la sección de abajo)
├── mesh/                       # SectionGeometry/RebarLayer + malla estructurada
├── materials/                  # CDP, acero multilineal, adherencia MC2010
├── elements/, solver/          # Q4/truss2/bond_link, Newton-Raphson propio
├── postprocess.py              # momento-curvatura
└── visualization_nl.py         # gráficos del modo no lineal
tests/                         # pytest, sin dependencia de la UI (tests/nl/ para fem2d_nl)
```

## Modo no lineal: sección de hormigón armado (momento-curvatura)

Hay un segundo motor de cálculo, `fem2d_nl/`, para un modo de análisis
nuevo (sección de hormigón armado con no linealidad de material completa),
ya conectado a la interfaz (`ui/rc_section.py`, seleccionable desde
"Modo de análisis" en la barra lateral). Es independiente del modo lineal
de arriba: solver Newton-Raphson propio (SolidsPy no sirve para esto), con
su propio elemento continuo (Q4). Es un modo **experimental**: revisar los
avisos de honestidad técnica más abajo antes de usar los resultados en
producción.

Permite:

- Definir la geometría de una franja de sección (ancho, alto, longitud de
  la franja analizada, malla) y capas de armadura (profundidad, número de
  barras, diámetro) con una tabla editable.
- Elegir hormigón (f'ck, con overrides opcionales de `ft`/`GF`/`eps_c1`
  para quien tenga valores verificados) y acero (fy, fu, deformación
  última) con curvas derivadas automáticamente por el fib Model Code 2010.
- Elegir adherencia perfecta o adherencia según el fib MC2010 (barras
  discretas + elementos de interfaz).
- Correr un análisis de momento-curvatura hasta una deformación máxima
  objetivo, con barra de progreso, y ver la curva M-κ, un preview de la
  malla con las capas de armadura, y el contorno de daño del hormigón en
  el último paso.
- Descargar la curva M-κ en CSV.

Nota de rendimiento: es un solver no lineal en Python puro, sin
aceleración — una malla chica (~15-20 elementos) tarda del orden de un
minuto para una curva completa hasta la deformación última del hormigón
(0.0035); mallas más finas o más pasos de carga pueden tardar varios
minutos. La UI lo advierte junto al campo de malla.

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
- CDP integrado en el elemento Q4 real, a través del solver completo
  (no solo a nivel de punto material): converge en tracción (ablanda
  sustancialmente tras el pico) y en compresión (pico dentro del 15% de
  f'cm) en casos de deformación moderada. Robustez del solver: line
  search (backtracking) dentro de cada iteración, recuperación
  adaptativa del tamaño de paso tras un recorte (antes, un solo recorte
  dejaba el solver atascado en pasos microscópicos para siempre), y un
  fallo de convergencia del return mapping local ya no hace crashear el
  solver global — se trata como cualquier otro fallo de iteración y
  dispara un recorte de paso. Ver `tests/nl/test_concrete_q4_integration.py`.

- Malla estructurada de sección (`mesh/section.py`, `mesh/structured.py`),
  en NumPy puro (sin gmsh — no hace falta resolver otra vez el problema de
  threading gmsh/Streamlit del modo lineal): franja rectangular con
  hormigón `quad4` + capas de armadura como `truss2`, en modo adherencia
  perfecta (comparten nodo con el hormigón) o `bond_slip` (nodos de acero
  duplicados + `bond_link` con la ley del fib MC2010). Las filas de malla
  se ajustan ("snapping") a la profundidad exacta de cada capa.
- Momento-curvatura (`postprocess.py::moment_curvature`) vía FEM completo:
  cinemática de secciones planas impuesta como condición esencial en
  ambos extremos de la franja (perfil lineal de `ux`, reutilizando
  `DisplacementControl` sin necesitar multi-point constraints), M y N
  extraídos de las reacciones nodales (exacto por equilibrio FEM, sin
  hipótesis adicionales). Validado en régimen elástico contra la teoría
  de vigas de sección transformada (dentro del 2%) y muestra ablandamiento
  claro de la rigidez secante tras la fisuración. Ver
  `tests/nl/test_moment_curvature.py` (marcado `slow`) y el aviso de
  honestidad técnica sobre `y_na` fijo, abajo.
- Cierre del ítem abierto de S4 (energía disipada): en vez de reintentar
  el enfoque fallido de comparar el área bajo la curva P-δ global entre
  mallas, se agregó `ConcreteCDP.dissipated_tensile_energy_density()`,
  la integral correcta a nivel de punto material (sin FEM) que vale
  exactamente `G_F/l_ch` para cualquier tamaño de elemento asumido —
  verificado numéricamente objetivo de malla (< 1% de dispersión) en
  `tests/nl/test_dissipated_energy_objective.py`. Este es el test de
  objetividad de malla que S4 no pudo cerrar limpiamente; queda cerrado
  a nivel material, no a nivel de localización de fisura en un FEM
  completo (eso sigue abierto, ver debajo).

- Interfaz Streamlit (`ui/rc_section.py`, `fem2d_nl/visualization_nl.py`)
  conectada al router de `app.py`: geometría/armadura/materiales/adherencia
  por formulario, barra de progreso durante el análisis (usa el
  `progress_cb` que ahora expone `moment_curvature`), curva M-κ, preview
  de malla y contorno de daño (`ConcreteCDP.damage_at`, envoltorio público
  agregado en esta sesión). Probada en vivo en el navegador (Playwright):
  el modo de adherencia perfecta corrió un caso completo de principio a
  fin (geometría por defecto, 3⌀16 a 40mm, εcu=0.0035) y mostró una curva
  M-κ con la forma esperada (M máx ≈169 kN·m) y un contorno de daño a
  tracción físicamente razonable (más daño en la fibra inferior, en
  tracción). El modo `bond_slip` está cubierto por
  `test_bond_slip_mode_runs_and_is_monotonic`; la verificación manual en
  vivo de ese modo específico quedó corriendo en segundo plano al cerrar
  la sesión sin confirmar su resultado — si alguien la retoma, revisar que
  termine sin error antes de darla por probada en la UI.

El plan completo (formulación, arquitectura, riesgos) está en el
historial de la sesión de desarrollo, no versionado en el repo.

**Alcance honesto de lo que NO se logró en la sesión S4** (se intentó y
se descartó, en vez de forzar un test que pasara sin decir la verdad):
un test cuantitativo de "objetividad de malla" a nivel de FEM completo
(comparar la energía disipada entre mallas con distinto refinamiento, o
contra la energía de fractura analítica G_F·área, a partir de la curva
carga-desplazamiento global). Se encontraron dos problemas reales:
(1) una barra perfectamente uniforme sin imperfección no tiene un
patrón de localización bien definido — es el error clásico de este tipo
de test, ya documentado en la literatura de crack-band, y hace falta
una imperfección explícita para forzar la localización; (2) el área
bajo la curva carga-desplazamiento GLOBAL incluye energía elástica
recuperable del resto de la estructura, no solo la energía disipada por
la fisura, así que compararla contra G_F·área no es válido sin antes
extraer la energía disipada de las variables de estado del material.
La sesión S5 resolvió el punto (2) a nivel material (ver arriba), pero el
punto (1) — un test de objetividad de malla end-to-end sobre localización
real en un FEM completo — sigue sin resolverse: empujar cualquier malla a
daño casi totalmente saturado sigue siendo numéricamente muy exigente
incluso con line search y cutback (matriz tangente casi singular) y
probablemente necesite arc-length (sesión S7, no implementada). Quien
continúe este trabajo debería tratar la objetividad de malla a nivel FEM
como un ítem abierto, no como algo ya verificado.

**Alcance honesto de lo que NO se logró en la sesión S5**: el plan
original pedía verificar que el modo de adherencia perfecta y el modo
`bond_slip` converjan al mismo momento último `M_u`. Al investigarlo se
encontró que, en la franja corta usada para `moment_curvature`, el
resultado de `bond_slip` en curvaturas intermedias resultó sorprendentemente
sensible al número de pasos de carga usados para llegar ahí (con 10 pasos
coincide con adherencia perfecta dentro del 0.1%; con 15-30 pasos difiere
hasta ~40-50% en curvaturas intermedias, aunque ambos casos reportan
"convergido"). La sospecha más probable, sin confirmar con el tiempo
disponible: los nodos de acero duplicados en la cara de referencia quedan
completamente libres (`moment_curvature` no les impone ninguna condición,
ver su docstring), representando una barra que "termina" ahí en vez de
continuar más allá del tramo modelado — un artefacto de usar una franja
corta, no necesariamente un error de la ley de adherencia. En vez de
forzar una comparación cuantitativa de `M_u` que podría estar ocultando
este problema, se descartó y se dejaron en su lugar chequeos más modestos
(`tests/nl/test_moment_curvature.py::test_bond_slip_mode_runs_and_is_monotonic`).
Diagnosticar esto con confianza (posiblemente alargando `span` o agregando
una condición de continuidad al acero en la cara de referencia) queda
como ítem abierto.

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
