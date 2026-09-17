#!/usr/bin/env bash
# Instala las dependencias del proyecto en dos pasos.
#
# solidspy==1.1.0.post1 declara en su setup.py una dependencia obsoleta
# (meshio==3.0) que en realidad nunca importa en su código; ese pin choca
# con la versión de meshio que necesita pygmsh. Por eso se instala aparte
# con --no-deps, después de que sus dependencias reales (numpy, scipy,
# matplotlib, easygui) ya están instaladas desde requirements.txt.
set -euo pipefail

pip install -r requirements.txt
pip install --no-deps solidspy==1.1.0.post1
