"""Curvas de tracción y compresión del hormigón a partir de f'ck, y su
conversión a las tablas de endurecimiento/daño (Birtel & Mark) que usa
`concrete_cdp.ConcreteCDP`.

Fórmulas de resistencia media, módulo elástico y energía de fractura
tomadas del fib Model Code 2010 (verificadas contra múltiples fuentes
secundarias durante la investigación). La curva de compresión usa la
forma de Sargin del MC2010, pero con `eps_c1` aproximado por una fórmula
empírica genérica (no la tabla por clase de resistencia del código, que
no se pudo verificar contra el texto original) — ver `mc2010_eps_c1_approx`.
El ablandamiento post-pico en compresión se regulariza con una rama
lineal simple (no una curva de la literatura), calibrada solo para
disipar la energía de fractura en compresión dada; es una simplificación
deliberada, documentada, no una fórmula normativa.
"""

import numpy as np

from fem2d_nl.exceptions import MaterialModelError


def mc2010_fcm(fck: float) -> float:
    return fck + 8.0


def mc2010_eci(fcm: float, alpha_e: float = 1.0) -> float:
    return 21_500.0 * alpha_e * (fcm / 10.0) ** (1.0 / 3.0)


def mc2010_fctm(fck: float) -> float:
    if fck <= 50.0:
        return 0.30 * fck ** (2.0 / 3.0)
    fcm = mc2010_fcm(fck)
    return 2.12 * np.log(1.0 + 0.1 * fcm)


def mc2010_fracture_energy(fcm: float) -> float:
    """G_F en N/mm."""
    return 73.0 * fcm**0.18 / 1000.0


def mc2010_eps_c1_approx(fcm: float) -> float:
    """Aproximación genérica de la deformación de pico en compresión
    (‰, convertida a deformación adimensional). No es la tabla por clase
    de resistencia del MC2010 (no verificada contra el texto original);
    es una fórmula de uso común, razonable como valor por defecto, pero
    debe poder sobreescribirse (`CDPParameters.compression_curve`)."""
    return min(0.0007 * fcm**0.31, 0.0028)


def check_crack_band_snapback(ft: float, gf: float, e0: float, l_ch: float) -> None:
    """Bažant-Oh: si el elemento es demasiado grande para la energía de
    fractura dada, la curva de ablandamiento regularizada deja de ser
    monótona decreciente (snap-back local) y el material se vuelve
    inestable de forma espuria. Nunca debe pasarse por alto en silencio.
    """
    limit = 2.0 * e0 * gf / ft**2
    if l_ch > limit:
        raise MaterialModelError(
            f"La malla es demasiado gruesa para la energía de fractura a tracción dada "
            f"(l_ch={l_ch:.3g} mm > {limit:.3g} mm): la curva de ablandamiento regularizada "
            f"tendría snap-back local. Refine la malla o aumente G_F."
        )


def _strictly_increasing(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-12
    return x


def build_tension_law(ft: float, gf: float, l_ch: float, e0: float, b_t: float = 0.1, n: int = 24):
    """Tabla (kappa_t, sigma_bar_t, d_t) para la ley de tracción, a partir
    de la curva bilineal esfuerzo-apertura de fisura del MC2010,
    regularizada por longitud característica (crack band) y convertida a
    variables de daño/endurecimiento por la regla de Birtel & Mark."""
    check_crack_band_snapback(ft, gf, e0, l_ch)
    w1 = gf / ft
    wc = 5.0 * w1
    w = np.linspace(0.0, wc, n)
    sigma = np.where(w <= w1, ft * (1.0 - 0.8 * w / w1), ft * (0.25 - 0.05 * w / w1))
    sigma = np.maximum(sigma, 1e-6 * ft)  # evita sigma=0 exacto (división por cero más abajo)

    eps_ck = w / l_ch
    damage = (1.0 - b_t) * eps_ck * e0 / (sigma + (1.0 - b_t) * eps_ck * e0)
    damage[0] = 0.0
    damage = np.clip(damage, 0.0, 0.999)

    kappa = _strictly_increasing(b_t * eps_ck)
    sigma_bar = sigma / (1.0 - damage)
    return kappa, sigma_bar, damage


def build_compression_law(
    fcm: float,
    eci: float,
    gc: float,
    l_ch: float,
    e0: float,
    eps_c1: float | None = None,
    b_c: float = 0.7,
    n: int = 40,
    residual_fraction: float = 0.05,
    initial_yield_fraction: float = 0.4,
):
    """Tabla (kappa_c, sigma_bar_c, d_c) para la ley de compresión: rama
    ascendente de Sargin (MC2010) desde el punto de fluencia inicial
    (`initial_yield_fraction*fcm`, valor aproximado de uso común, no una
    tabla del código) hasta el pico, rama descendente lineal simplificada
    y regularizada por energía de fractura en compresión.

    `kappa_c=0` corresponde al INICIO de la fluencia (no a deformación
    cero): igual que `sigma_bar_t(kappa_t=0)=ft` en tracción, aquí
    `sigma_bar_c(kappa_c=0)≈initial_yield_fraction*fcm`. Confundir esto
    con kappa_c medido desde deformación cero hace que la superficie de
    fluencia nunca se active en estados de tracción dominante (el término
    β de la función de fluencia usa sigma_bar_c(0) como referencia).
    """
    eps_c1 = eps_c1 if eps_c1 is not None else mc2010_eps_c1_approx(fcm)
    ec1 = fcm / eps_c1
    k = eci / ec1

    def sargin(eta):
        return fcm * (k * eta - eta**2) / (1.0 + (k - 2.0) * eta)

    eta_fine = np.linspace(1e-5, 1.0, 4000)
    sigma_fine = sargin(eta_fine)
    eta0 = float(np.interp(initial_yield_fraction * fcm, sigma_fine, eta_fine))

    eta = np.linspace(eta0, 1.0, n)
    sigma_pre = sargin(eta)
    eps_pre = eta * eps_c1

    kappa_ult = 2.0 * (gc / l_ch) / fcm
    if kappa_ult < 0.1 * eps_c1:
        raise MaterialModelError(
            f"La malla es demasiado gruesa para la energía de fractura en compresión dada "
            f"(l_ch={l_ch:.3g} mm): la rama de ablandamiento quedaría prácticamente vertical. "
            "Refine la malla o aumente Gc."
        )
    xi = np.linspace(0.0, 1.0, n)[1:]
    sigma_post = fcm * (1.0 - xi) + fcm * residual_fraction * xi
    eps_post = eps_c1 + xi * kappa_ult

    eps_total = np.concatenate([eps_pre, eps_post])
    sigma_total = np.concatenate([sigma_pre, sigma_post])

    eps_in = np.maximum(eps_total - sigma_total / e0, 0.0)
    eps_in = np.maximum.accumulate(eps_in)
    eps_in -= eps_in[0]  # kappa_c=0 en el inicio de la fluencia, no en deformación cero

    damage = (1.0 - b_c) * eps_in * e0 / (sigma_total + (1.0 - b_c) * eps_in * e0)
    damage[0] = 0.0
    damage = np.clip(damage, 0.0, 0.999)

    kappa = _strictly_increasing(b_c * eps_in)
    sigma_bar = sigma_total / (1.0 - damage)
    return kappa, sigma_bar, damage
