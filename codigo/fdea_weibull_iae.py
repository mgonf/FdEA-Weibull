#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FdEA: modelo de fórmula abierta (IAE) frente a modelos paramétricos de supervivencia.
Versión 1.3

Escenarios:
  - 2020 (ajuste): IAE (2020), Tabla 1 del informe IAE (2025).
  - 2025 (validación fuera de muestra): IAE (2025), Tablas 2 y 5 del informe.

Supuestos (IAE, 2025, Tabla 4): tipo real de las aportaciones = crecimiento real pasado del
PIB; tipo real de las prestaciones = crecimiento real futuro del PIB; pensión revalorizada con
el IPC (constante en términos reales); tablas de pensionistas de la Seguridad Social (unisex),
resumidas por su esperanza de vida a los 65 años (21,14 en 2020 y 21,52 en 2025).

Base reguladora (art. 209.1 y disposición transitoria 8.ª del texto refundido de la LGSS):
276 meses divididos entre 322 en 2020 y 300 meses entre 350 en 2025. Las bases de los dos
últimos años se computan por su valor nominal y las anteriores se actualizan con el IPC hasta
dos años antes de la jubilación (Devesa Carpio et al., 2020, nota 3). Como el modelo trabaja en
términos reales, la inflación de esos dos años reduce el valor real de la base reguladora. Se
usa el IPC de diciembre sobre diciembre del INE (2018 y 2019; 2023 y 2024), repartido de forma
uniforme entre los meses de cada año. Como en el IAE, la jubilación se produce el 1 de enero.

Como las tablas de la Seguridad Social no se publican, el modelo de fórmula abierta usa la
forma de la PER2020_Ind_2ndo.orden (unisex) reescalada para reproducir la e65 del IAE.
Los modelos paramétricos se estiman sobre esa tabla en 2020 y, para 2025, solo se actualiza
su parámetro de escala con la nueva e65 (un único dato).

Uso:
    python fdea_weibull_iae.py            # necesita per2020_ind_2orden.csv en la misma carpeta
Incluye, además, el FdEA medio ponderado con los pesos del IAE (2020, tabla A1), la
descomposición de los cambios de 2020 a 2025, el contraste de la regla de reescalado y el
factor de ajuste de la pensión por longevidad, también con las demás leyes y con la tabla
proyectada.

Salidas: fdea_resultados.xlsx, mallas/*.png, resumen.json

Cambios respecto a la versión 1.2 (no cambia ningún resultado anterior):
  - Contraste de la regla de reescalado con todas las leyes, no solo con la de Weibull
    (sección 4.5 del artículo).
  - Factor de ajuste por longevidad de la tabla proyectada y de cada ley cuando la esperanza de
    vida a los 65 años de la tabla proyectada aumenta un año desde 2025 (sección 4.6).
  - Ganancia de esperanza de vida por edad de jubilación entre 2025 y 2045 en la tabla
    proyectada y en cada ley (sección 4.5).

Cambios de la versión 1.2 respecto a la 1.1:
  - Base reguladora conforme a la norma: periodo de 23 años en 2020 (276/322) y desfase
    nominal de las 24 últimas bases (antes: 300/350 en ambos años y todas las bases en
    términos reales).
  - Crecimiento real de las bases del 0,22% anual en ambos escenarios, el que documenta el IAE
    (2025, p. 16). Se elimina el valor de -0,13% que la versión 1.1 fijaba para 2025.
  - La ley de Weibull se denomina por sus ecuaciones en el artículo (Ec. 7-8).
  - Se añaden la descomposición de 2020 a 2025 (Tabla 10) y otras cifras del texto.
"""
import copy
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.special import gamma as G

__version__ = "1.3"

ANIOS = list(range(33, 44))

# ------------------------------------------------------------------ datos IAE
EDADES_20 = list(range(60, 71))
CAEJ_20 = np.array([  # Tabla 3 del artículo (IAE, 2020, tabla 1)
    [0.565, 0.565, 0.617, 0.687, 0.774, 0.862, 0.932, 0.959, 0.987, 1.014, 1.042],
    [0.597, 0.668, 0.668, 0.704, 0.793, 0.883, 0.954, 0.982, 1.009, 1.037, 1.064],
    [0.611, 0.684, 0.757, 0.794, 0.812, 0.904, 0.977, 1.005, 1.032, 1.060, 1.087],
    [0.625, 0.700, 0.775, 0.850, 0.906, 0.925, 1.000, 1.028, 1.055, 1.083, 1.110],
    [0.625, 0.700, 0.775, 0.850, 0.925, 1.000, 1.000, 1.028, 1.055, 1.083, 1.110],
    [0.625, 0.700, 0.775, 0.850, 0.925, 1.000, 1.028, 1.028, 1.055, 1.083, 1.110],
    [0.650, 0.720, 0.790, 0.860, 0.930, 1.000, 1.040, 1.055, 1.055, 1.083, 1.110],
    [0.650, 0.720, 0.790, 0.860, 0.930, 1.000, 1.040, 1.080, 1.083, 1.110, 1.110],
    [0.650, 0.720, 0.790, 0.860, 0.930, 1.000, 1.040, 1.080, 1.120, 1.110, 1.138],
    [0.675, 0.740, 0.805, 0.870, 0.935, 1.000, 1.040, 1.080, 1.120, 1.160, 1.138],
    [0.675, 0.740, 0.805, 0.870, 0.935, 1.000, 1.040, 1.080, 1.120, 1.160, 1.200]])
FDEA_IAE_20 = np.array([
    [1.43, 1.39, 1.48, 1.59, 1.74, 1.87, 1.95, 1.94, 1.92, 1.90, 1.87],
    [1.44, 1.57, 1.53, 1.56, 1.70, 1.83, 1.91, 1.90, 1.88, 1.85, 1.82],
    [1.41, 1.54, 1.65, 1.68, 1.67, 1.79, 1.87, 1.85, 1.83, 1.81, 1.78],
    [1.38, 1.51, 1.62, 1.72, 1.78, 1.75, 1.83, 1.81, 1.79, 1.77, 1.74],
    [1.33, 1.44, 1.55, 1.65, 1.74, 1.82, 1.75, 1.74, 1.72, 1.69, 1.67],
    [1.27, 1.38, 1.49, 1.58, 1.67, 1.74, 1.73, 1.67, 1.65, 1.63, 1.60],
    [1.27, 1.37, 1.46, 1.54, 1.61, 1.67, 1.68, 1.64, 1.58, 1.56, 1.54],
    [1.22, 1.31, 1.40, 1.48, 1.55, 1.61, 1.62, 1.62, 1.56, 1.54, 1.48],
    [1.17, 1.26, 1.35, 1.42, 1.49, 1.55, 1.55, 1.55, 1.55, 1.48, 1.45],
    [1.17, 1.25, 1.32, 1.38, 1.44, 1.49, 1.49, 1.49, 1.49, 1.48, 1.40],
    [1.12, 1.20, 1.27, 1.33, 1.38, 1.43, 1.43, 1.43, 1.43, 1.43, 1.41]])
EDADES_25 = list(range(63, 71))
TS_25 = np.array([  # IAE (2025), Tabla 2 (%)
    [57.97, 65.33, 78.52, 87.97, 92.02, 96.02, 100.02, 108.02],
    [59.41, 66.95, 80.47, 90.15, 94.30, 98.30, 102.30, 110.30],
    [64.71, 68.57, 82.41, 92.33, 96.58, 100.58, 104.58, 112.58],
    [74.15, 74.15, 84.36, 94.51, 98.86, 102.86, 106.86, 114.86],
    [79.00, 93.23, 93.23, 95.60, 100.00, 104.00, 108.00, 116.00],
    [79.00, 94.50, 96.48, 96.48, 100.00, 104.00, 108.00, 116.00],
    [81.00, 94.75, 100.00, 104.00, 104.00, 104.00, 108.00, 116.00],
    [81.00, 94.75, 100.00, 104.00, 108.00, 108.00, 108.00, 116.00],
    [81.00, 94.75, 100.00, 104.00, 108.00, 112.00, 112.00, 116.00],
    [83.00, 95.00, 100.00, 104.00, 108.00, 112.00, 116.00, 116.00],
    [83.00, 95.00, 100.00, 104.00, 108.00, 112.00, 116.00, 120.00]]) / 100
FDEA_IAE_25 = np.array([  # IAE (2025), Tabla 5
    [1.39, 1.52, 1.76, 1.90, 1.92, 1.97, 2.02, 2.19],
    [1.37, 1.49, 1.73, 1.87, 1.89, 1.93, 1.98, 2.14],
    [1.43, 1.47, 1.70, 1.84, 1.85, 1.90, 1.94, 2.09],
    [1.58, 1.52, 1.67, 1.81, 1.82, 1.86, 1.90, 2.04],
    [1.61, 1.84, 1.78, 1.76, 1.77, 1.81, 1.84, 1.97],
    [1.55, 1.79, 1.77, 1.70, 1.70, 1.73, 1.76, 1.89],
    [1.53, 1.73, 1.76, 1.80, 1.73, 1.66, 1.69, 1.81],
    [1.47, 1.66, 1.69, 1.73, 1.76, 1.69, 1.62, 1.73],
    [1.41, 1.60, 1.63, 1.66, 1.69, 1.71, 1.64, 1.66],
    [1.39, 1.54, 1.57, 1.60, 1.62, 1.65, 1.67, 1.60],
    [1.34, 1.49, 1.51, 1.54, 1.57, 1.59, 1.60, 1.62]])


# IAE (2020), tabla A1: gasto de las altas de jubilación (MCVL 2018), en %, filas 33..43 años
# cotizados y columnas <=60..>=70 años de edad. Se usan como ponderaciones del FdEA medio.
PESOS_MCVL_2018 = np.array([
    [0.03, 1.56, 0.48, 0.48, 0.43, 12.50, 1.06, 0.52, 0.28, 0.19, 0.50],
    [0.00, 0.51, 0.10, 0.11, 0.15, 0.90, 0.09, 0.05, 0.03, 0.00, 0.05],
    [0.05, 0.71, 0.13, 0.64, 0.38, 1.27, 0.17, 0.04, 0.04, 0.01, 0.10],
    [0.07, 0.74, 0.18, 0.48, 0.39, 1.53, 0.09, 0.04, 0.02, 0.05, 0.11],
    [0.02, 1.10, 0.21, 0.98, 0.42, 1.45, 0.05, 0.02, 0.03, 0.01, 0.04],
    [0.19, 1.09, 0.29, 1.00, 0.42, 1.59, 0.05, 0.04, 0.05, 0.00, 0.05],
    [0.13, 1.41, 0.26, 1.04, 0.52, 1.81, 0.06, 0.05, 0.01, 0.02, 0.07],
    [0.13, 1.60, 0.26, 1.43, 0.57, 1.76, 0.03, 0.07, 0.03, 0.02, 0.05],
    [0.19, 1.68, 0.36, 1.44, 0.66, 2.51, 0.08, 0.08, 0.02, 0.02, 0.06],
    [0.11, 2.17, 0.27, 1.66, 0.69, 2.50, 0.06, 0.04, 0.03, 0.01, 0.07],
    [1.29, 6.67, 1.52, 7.55, 4.18, 15.07, 0.39, 0.14, 0.18, 0.09, 0.56]]) / 100
# Mismos pesos con las edades de 60 a 62 agrupadas en 63 (rango de 2025, como hace el IAE)
PESOS_63_70 = PESOS_MCVL_2018[:, 3:].copy()
PESOS_63_70[:, 0] += PESOS_MCVL_2018[:, :3].sum(1)


def media_ponderada(M, w):
    """FdEA medio y desviación típica ponderados (criterio del IAE)."""
    w = w / w.sum()
    m = float((M * w).sum())
    return m, float(np.sqrt((w * (M - m) ** 2).sum()))


def tc_2020(anio):
    return 0.1615


def tc_2025(anio):
    return {2023: 0.1672, 2024: 0.1678}.get(anio, 0.1637 if anio <= 2022 else 0.1683)


@dataclass
class Escenario:
    nombre: str
    anio_jub: int
    edades: list
    caej: np.ndarray
    fdea_iae: np.ndarray
    tc: callable
    r_aport: float      # crecimiento real pasado del PIB
    i_prest: float      # crecimiento real futuro del PIB
    e65: float          # esperanza de vida a los 65 (tablas SS, unisex)
    meses_br: int       # meses de bases que entran en la base reguladora
    divisor_br: int     # divisor de la base reguladora
    ipc: tuple          # IPC dic./dic. (INE) de los años anteriores a la jubilación: (Y-2, Y-1)
    g_sal: float = 0.0022   # crecimiento real de las bases medias (IAE, 2025, p. 16: 0,22%)
    pagas: int = 14
    exencion: bool = False  # exención de cuotas tras la edad ordinaria (IAE 2025)
    edad_max: int = 120


# 2020: DT 8.ª LGSS (23 años: 276 meses / 322); IPC de 2018 (1,2%) y 2019 (0,8%)
ESC20 = Escenario("2020", 2020, EDADES_20, CAEJ_20, FDEA_IAE_20, tc_2020, 0.0248, 0.0159, 21.14,
                  meses_br=276, divisor_br=322, ipc=(0.012, 0.008))
# 2025: art. 209.1 LGSS (25 años: 300 meses / 350); IPC de 2023 (3,1%) y 2024 (2,8%)
ESC25 = Escenario("2025", 2025, EDADES_25, TS_25, FDEA_IAE_25, tc_2025, 0.0224, 0.0122, 21.52,
                  meses_br=300, divisor_br=350, ipc=(0.031, 0.028), exencion=True)


def mensual(x):
    return (1 + x) ** (1 / 12) - 1


# ------------------------------------------------------------ aportaciones y BR
def indice_ipc(k, ipc):
    """Nivel de precios del mes situado k meses antes de la jubilación, con el de diciembre del
    año anterior (k = 1) igual a 1 y variación mensual uniforme dentro de cada año natural.
    Desde k = 25 (diciembre de tres años antes) se mantiene constante: las bases anteriores se
    actualizan con el IPC hasta esa fecha y las 24 últimas se toman por su valor nominal."""
    p2, p1 = ipc                     # IPC dic./dic. de los años Y-2 y Y-1
    k = np.minimum(np.asarray(k, dtype=float), 25)
    return np.where(k <= 13, (1 + p1) ** (-(k - 1) / 12),
                    (1 + p1) ** -1 * (1 + p2) ** (-(k - 13) / 12))


def base_reguladora(esc: Escenario, nominal=True):
    """Base reguladora en términos reales (precios de la jubilación; base real del último mes = 1).
    Con nominal=False, todas las bases en términos reales (criterio de la versión 1.1)."""
    k = np.arange(1, esc.meses_br + 1)
    base = (1 + mensual(esc.g_sal)) ** (-(k - 1))
    if nominal:
        base = base * indice_ipc(k, esc.ipc)
    return float(np.sum(base) / esc.divisor_br)


def aportaciones_y_br(anios, edad, esc: Escenario):
    """Valores en términos reales; el FdEA no depende del nivel salarial."""
    n = 12 * anios
    t = np.arange(n)
    meses_antes = n - t                                   # meses hasta la jubilación
    base = (1 + mensual(esc.g_sal)) ** (t - (n - 1))      # última base = 1
    anio_cal = esc.anio_jub - np.ceil(meses_antes / 12).astype(int)
    tc = np.array([esc.tc(a) for a in anio_cal])
    if esc.exencion:
        edad_ord = 65 if anios >= 38.25 else 66 + 8 / 12
        edad_mes = edad - meses_antes / 12
        tc = np.where(edad_mes >= edad_ord, 0.0, tc)
    vaa_aport = np.sum(base * tc * (1 + mensual(esc.r_aport)) ** meses_antes)
    return vaa_aport, base_reguladora(esc)


def fdea_desde_renta(renta_anual_por_edad, esc: Escenario):
    """renta_anual_por_edad[e] = valor actual de 1 u.m. anual vitalicia desde la edad e."""
    out = np.zeros((len(ANIOS), len(esc.edades)))
    for j, e in enumerate(esc.edades):
        for i, a in enumerate(ANIOS):
            aport, br = aportaciones_y_br(a, e, esc)
            out[i, j] = br * esc.caej[i, j] * esc.pagas * renta_anual_por_edad[e] / aport
    return out


def delta(esc):
    return np.log(1 + esc.i_prest)  # pensión constante en términos reales


# ---------------------------------------------- tabla (fórmula abierta, PER2020)
def cargar_per2020(ruta):
    df = pd.read_csv(ruta, sep=";", decimal=",")
    df.columns = [c.strip().lower() for c in df.columns]
    return df.set_index("edad").sort_index()


def kpx_tabla(per, edad, anio_jub, c=1.0, edad_max=120):
    """Supervivencia anual generacional, unisex 50/50, con qx multiplicados por c."""
    def sx(s):
        q = [min(c * per.loc[x, f"qx_base_{s}"] *
                 np.exp(-per.loc[x, f"lambda_{s}"] * (anio_jub + k - 2012)), 1.0)
             for k, x in enumerate(range(edad, edad_max))]
        return np.concatenate([[1.0], np.cumprod(1 - np.array(q))])
    return 0.5 * sx("h") + 0.5 * sx("m")


def mensualizar(kpx):
    k = np.arange(len(kpx))
    tm = np.arange(12 * (len(kpx) - 1) + 1) / 12
    return tm, np.interp(tm, k, kpx)


def e_x(tm, p):
    return np.trapezoid(p, tm) if hasattr(np, "trapezoid") else np.trapz(p, tm)


def calibrar_tabla(per, esc):
    f = lambda c: e_x(*mensualizar(kpx_tabla(per, 65, esc.anio_jub, c))) - esc.e65
    return brentq(f, 0.2, 6)


def rentas_tabla(per, esc, c):
    d = delta(esc)
    out = {}
    for e in esc.edades:
        tm, p = mensualizar(kpx_tabla(per, e, esc.anio_jub, c))
        out[e] = np.sum(np.exp(-d * tm) * p) / 12   # pagos mensuales anticipados
    return out


# ------------------------------------------------------- leyes paramétricas
# Cada ley define tpx(x, t) (t en años) y un parámetro de escala que se reajusta con e65.
class WeibullPaper:
    """Ec. 7-8 del artículo: p(t) = exp(-(t / a(edad))^b), t en meses."""
    nombre, n_par = "Weibull (Ec. 7-8)", 4

    def __init__(self, a0=300., a1=0., a2=0., b=3.):
        self.th = np.array([a0, a1, a2, b])
        self.phi = 1.0  # factor de escala común (actualización por e65)

    def a(self, e):
        a0, a1, a2, _ = self.th
        return self.phi * (a0 + a1 * (e - 60) + a2 * (e - 60) ** 2)

    def tpx(self, x, t):
        return np.exp(-(np.asarray(t) * 12 / self.a(x)) ** self.th[3])

    def e65(self):  # fórmula cerrada: E[T] = a(65)·Γ(1+1/b), en meses
        return self.a(65) * G(1 + 1 / self.th[3]) / 12

    def reescalar(self, e65):
        self.phi *= e65 / self.e65()


class WeibullEdad:
    """S(x) = exp(-(x/lam)^k); tpx = S(x+t)/S(x)."""
    nombre, n_par = "Weibull edad alcanzada", 2

    def __init__(self, lam=90., k=10.):
        self.th = np.array([lam, k])

    def tpx(self, x, t):
        lam, k = self.th
        return np.exp(-(((x + t) / lam) ** k - (x / lam) ** k))

    def reescalar(self, e65):
        k = self.th[1]
        self.th[0] = brentq(lambda l: quad(lambda t: np.exp(-(((65 + t) / l) ** k - (65 / l) ** k)),
                                           0, 60)[0] - e65, 60, 130)


class Gompertz:
    """mu(x) = B e^{c x}."""
    nombre, n_par = "Gompertz", 2

    def __init__(self, lB=-10., c=0.1):
        self.th = np.array([lB, c])

    def tpx(self, x, t):
        lB, c = self.th
        return np.exp(-np.exp(lB) / c * np.exp(c * x) * (np.exp(c * np.asarray(t)) - 1))

    def reescalar(self, e65):
        c = self.th[1]
        self.th[0] = brentq(lambda lB: quad(lambda t: np.exp(-np.exp(lB) / c * np.exp(c * 65) *
                                                              (np.exp(c * t) - 1)), 0, 60)[0] - e65,
                            -30, 0)


class Makeham(Gompertz):
    """mu(x) = A + B e^{c x}."""
    nombre, n_par = "Gompertz-Makeham", 3

    def __init__(self, A=1e-3, lB=-10., c=0.1):
        self.th = np.array([A, lB, c])

    def tpx(self, x, t):
        A, lB, c = self.th
        t = np.asarray(t)
        return np.exp(-A * t - np.exp(lB) / c * np.exp(c * x) * (np.exp(c * t) - 1))

    def reescalar(self, e65):
        A, _, c = self.th
        f = lambda lB: quad(lambda t: np.exp(-A * t - np.exp(lB) / c * np.exp(c * 65) *
                                             (np.exp(c * t) - 1)), 0, 60)[0] - e65
        self.th[1] = brentq(f, -30, 0)


class Exponencial:
    """mu constante = 1/e65 (referencia ingenua, 1 parámetro)."""
    nombre, n_par = "Exponencial", 1

    def __init__(self, mu=0.05):
        self.th = np.array([mu])

    def tpx(self, x, t):
        return np.exp(-self.th[0] * np.asarray(t))

    def reescalar(self, e65):
        self.th[0] = 1 / e65


def estimar(ley, objetivo, x0, bounds):
    """Mínimos cuadrados entre tpx de la ley y la supervivencia mensual de la tabla (60-70)."""
    def sce(th):
        ley.th = np.array(th, dtype=float)
        return sum(np.sum((ley.tpx(e, tm) - p) ** 2) for e, (tm, p) in objetivo.items())
    res = minimize(sce, x0, method="L-BFGS-B", bounds=bounds)
    ley.th = res.x
    return ley


def rentas_ley(ley, esc):
    d = delta(esc)
    T = esc.edad_max
    return {e: quad(lambda t: np.exp(-d * t) * ley.tpx(e, t), 0, T - e, limit=400)[0]
            for e in esc.edades}


# ------------------------------------------------------------- métricas y mallas
def metricas(ref, mod):
    d = mod - ref
    return dict(MAPE=100 * np.mean(np.abs(d / ref)), MAE=np.mean(np.abs(d)),
                max_abs=np.max(np.abs(d)), max_rel=100 * np.max(np.abs(d / ref)),
                sesgo=np.mean(d), pct_le_005=100 * np.mean(np.abs(d) <= 0.05 + 1e-9),
                corr=np.corrcoef(ref.ravel(), mod.ravel())[0, 1], media=np.mean(mod))


ROJO, AMARILLO, VERDE = "F8696B", "FFEB84", "63BE7B"


def _mezcla(c1, c2, f):
    a = [int(c1[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(a, b))


def colores_escala(m, invertir=False):
    """Escala de 3 colores de Excel (rojo-amarillo-verde, mediana como punto medio), como el IAE."""
    lo, hi = (VERDE, ROJO) if invertir else (ROJO, VERDE)
    vmin, vmed, vmax = np.min(m), np.median(m), np.max(m)
    out = np.empty(m.shape, dtype=object)
    for idx, v in np.ndenumerate(m):
        if v <= vmed:
            out[idx] = _mezcla(lo, AMARILLO, 0 if vmed == vmin else (v - vmin) / (vmed - vmin))
        else:
            out[idx] = _mezcla(AMARILLO, hi, 0 if vmax == vmed else (v - vmed) / (vmax - vmed))
    return out


def malla_png(m, edades, titulo, ruta, invertir=False, colorear=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = colores_escala(m if colorear is None else colorear, invertir)
    fig, ax = plt.subplots(figsize=(0.75 * len(edades) + 1.5, 4.8))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#" + cols[i, j], edgecolor="black", lw=0.5))
            v = 0.0 if abs(m[i, j]) < 0.005 else m[i, j]
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}".replace(".", ",").replace("-", "−"),
                    ha="center", va="center", fontsize=8)
    ax.set_xlim(0, m.shape[1]); ax.set_ylim(m.shape[0], 0)
    ax.set_xticks(np.arange(m.shape[1]) + 0.5); ax.set_xticklabels(edades)
    ax.set_yticks(np.arange(m.shape[0]) + 0.5); ax.set_yticklabels(ANIOS)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Edad de jubilación"); ax.set_ylabel("Años cotizados"); ax.tick_params(length=0)
    ax.set_title(titulo, fontsize=10, pad=28)
    fig.tight_layout(); fig.savefig(ruta, dpi=200); plt.close(fig)


def hoja_malla(ws, m, edades, invertir=False, colorear=None):
    from openpyxl.styles import PatternFill, Alignment, Border, Side, Font
    s = Side(style="thin", color="000000"); b = Border(left=s, right=s, top=s, bottom=s)
    cols = colores_escala(m if colorear is None else colorear, invertir)
    ws.cell(1, 1, "Años cotizados").font = Font(bold=True)
    ws.cell(1, 2, "Edad de jubilación").font = Font(bold=True)
    for j, e in enumerate(edades):
        c = ws.cell(2, j + 2, e); c.border = b; c.alignment = Alignment(horizontal="center")
    for i, a in enumerate(ANIOS):
        c = ws.cell(3 + i, 1, a); c.border = b; c.alignment = Alignment(horizontal="center")
        for j in range(len(edades)):
            c = ws.cell(3 + i, j + 2, round(float(m[i, j]), 2))
            c.number_format = "0.00"; c.border = b; c.alignment = Alignment(horizontal="center")
            c.fill = PatternFill("solid", start_color=cols[i, j], end_color=cols[i, j])
    ws.column_dimensions["A"].width = 15


# ------------------------------------------------------------------------ main
def main(ruta_per="per2020_ind_2orden.csv", salida="fdea_resultados.xlsx"):
    per = cargar_per2020(ruta_per)
    res, tablas, mets = {"version": __version__}, {}, []

    # 0) Base reguladora (común a todas las celdas de cada escenario)
    res["base_reguladora"] = {
        esc.nombre: {"meses": esc.meses_br, "divisor": esc.divisor_br, "ipc": list(esc.ipc),
                     "g_sal": esc.g_sal, "BR": base_reguladora(esc),
                     "BR_todo_real": base_reguladora(esc, nominal=False),
                     "factor_nominal": base_reguladora(esc) / base_reguladora(esc, nominal=False)}
        for esc in (ESC20, ESC25)}

    # 1) Tabla PER2020 reescalada a la e65 del IAE (fórmula abierta)
    c20, c25 = calibrar_tabla(per, ESC20), calibrar_tabla(per, ESC25)
    res["factor_qx"] = {"2020": c20, "2025": c25}
    for esc, c in [(ESC20, c20), (ESC25, c25)]:
        tablas[f"Abierta_{esc.nombre}"] = fdea_desde_renta(rentas_tabla(per, esc, c), esc)

    # 2) Leyes paramétricas: estimadas sobre la tabla 2020; en 2025 solo cambia la escala (e65)
    objetivo = {e: mensualizar(kpx_tabla(per, e, 2020, c20)) for e in EDADES_20}
    leyes = [
        (WeibullPaper(), [400., -10., 0., 3.], [(1, None), (None, None), (None, None), (0.05, 30)]),
        (WeibullEdad(), [90., 10.], [(60, 130), (1.01, 40)]),
        (Gompertz(), [-10., 0.1], [(-30, 0), (0.001, 0.5)]),
        (Makeham(), [1e-3, -10., 0.1], [(0, 0.05), (-30, 0), (0.001, 0.5)]),
        (Exponencial(), [0.05], [(1e-4, 1)]),
    ]
    res["parametros"] = {}
    estados20 = {}                 # cada ley tal como queda en 2020 (para 5d-5f)
    for ley, x0, bnd in leyes:
        estimar(ley, objetivo, x0, bnd)
        ley.reescalar(ESC20.e65)   # todas las leyes reproducen exactamente la e65 del IAE
        estados20[ley.nombre] = copy.deepcopy(ley)
        th20 = ley.th.copy(); phi20 = getattr(ley, "phi", None)
        tablas[f"{ley.nombre}_2020"] = fdea_desde_renta(rentas_ley(ley, ESC20), ESC20)
        ley.reescalar(ESC25.e65)
        tablas[f"{ley.nombre}_2025"] = fdea_desde_renta(rentas_ley(ley, ESC25), ESC25)
        res["parametros"][ley.nombre] = {"2020": th20.tolist(),
                                         "2025": ley.th.tolist(),
                                         "phi_2020": phi20,
                                         "phi_2025": getattr(ley, "phi", None)}

    # 3) Aplicación: sensibilidad del FdEA 2025 a la longevidad (Weibull Ec. 7-8)
    wp = leyes[0][0]
    sens = {}
    for e65 in [21.52, 22.52, 22.63, 23.11]:
        wp.reescalar(e65)
        sens[e65] = fdea_desde_renta(rentas_ley(wp, ESC25), ESC25)
    wp.reescalar(ESC25.e65)
    tablas["Sens_+1anio_2025"] = sens[22.52] - sens[21.52]
    res["sensibilidad"] = {str(k): float(np.mean(v)) for k, v in sens.items()}
    res["elasticidad_e65"] = float(np.mean((sens[22.52] / sens[21.52] - 1) / (1 / 21.52)))

    # 4) Métricas frente al IAE
    for nombre, m in list(tablas.items()):
        if nombre.startswith("Sens"):
            continue
        esc = ESC20 if nombre.endswith("2020") else ESC25
        d = metricas(esc.fdea_iae, m); d.update(modelo=nombre.rsplit("_", 1)[0], escenario=esc.nombre)
        mets.append(d)
    dfm = pd.DataFrame(mets)[["modelo", "escenario", "MAPE", "MAE", "max_abs", "max_rel",
                              "sesgo", "pct_le_005", "corr", "media"]].round(4)
    res["metricas"] = dfm.to_dict(orient="records")
    print(dfm.to_string(index=False))

    # 5) Análisis adicionales
    wp_ = WeibullPaper(*wp.th); wp_.phi = wp.phi * ESC20.e65 / ESC25.e65   # Weibull en 2020
    W20, W25 = tablas["Weibull (Ec. 7-8)_2020"], tablas["Weibull (Ec. 7-8)_2025"]
    A20, A25 = tablas["Abierta_2020"], tablas["Abierta_2025"]
    res["ponderados_2020"] = {
        "IAE_60_70": media_ponderada(FDEA_IAE_20, PESOS_MCVL_2018),
        "Weibull_60_70": media_ponderada(W20, PESOS_MCVL_2018),
        "IAE_63_70": media_ponderada(FDEA_IAE_20[:, 3:], PESOS_63_70),
        "Weibull_63_70": media_ponderada(W20[:, 3:], PESOS_63_70)}
    # Con los mismos pesos de 2018 (el IAE pondera 2025 con la MCVL de 2023, que no publica)
    res["ponderados_2025_pesos_2018"] = {
        "IAE": media_ponderada(FDEA_IAE_25, PESOS_63_70),
        "Weibull": media_ponderada(W25, PESOS_63_70)}

    # Cifras del texto: ajuste sin el segundo paso (escala sin calibrar a la e65 del IAE),
    # perfil por edad, distancia entre Weibull y la fórmula abierta, y celda de mayor diferencia
    ls = WeibullPaper(*wp.th)                     # phi = 1: parámetros de mínimos cuadrados
    W20_ls = fdea_desde_renta(rentas_ley(ls, ESC20), ESC20)
    res["sin_calibrar_2020"] = {"e65": float(ls.e65()),
                                **{k: float(v) for k, v in metricas(FDEA_IAE_20, W20_ls).items()}}
    res["perfil_edad"] = {
        "2020": dict(zip(map(str, EDADES_20), (W20 / FDEA_IAE_20).mean(0).tolist())),
        "2025": dict(zip(map(str, EDADES_25), (W25 / FDEA_IAE_25).mean(0).tolist()))}
    res["weibull_vs_abierta"] = {
        "2020": {"MAPE": float(100 * np.mean(np.abs(W20 / A20 - 1))), "max_abs": float(np.abs(W20 - A20).max())},
        "2025": {"MAPE": float(100 * np.mean(np.abs(W25 / A25 - 1))), "max_abs": float(np.abs(W25 - A25).max())}}
    res["leyes_vs_abierta"] = {
        ley.nombre: {T: float(100 * np.mean(np.abs(tablas[f"{ley.nombre}_{T}"] / tablas[f"Abierta_{T}"] - 1)))
                     for T in ("2020", "2025")} for ley, _, _ in leyes}
    res["mayor_diferencia"] = {}
    for T, M, I, ed in (("2020", W20, FDEA_IAE_20, EDADES_20), ("2025", W25, FDEA_IAE_25, EDADES_25)):
        i, j = np.unravel_index(np.argmax(np.abs(M - I)), M.shape)
        res["mayor_diferencia"][T] = {"anios": ANIOS[i], "edad": ed[j], "modelo": float(M[i, j]),
                                      "IAE": float(I[i, j]), "dif": float(M[i, j] - I[i, j])}

    # 5a) Descomposición de 2020 a 2025 (Tabla 10): se introducen los cambios de uno en uno,
    # como en IAE (2025, tabla 7). Media simple de la malla 63-70 y media con los pesos de 2018.
    def malla_W(esc, e65):
        l = WeibullPaper(*wp.th); l.phi = wp_.phi * e65 / ESC20.e65
        return fdea_desde_renta(rentas_ley(l, esc), esc)
    b0 = copy.deepcopy(ESC20); b0.edades = EDADES_25; b0.caej = CAEJ_20[:, 3:]
    sa = copy.deepcopy(b0); sa.anio_jub = ESC25.anio_jub; sa.r_aport = ESC25.r_aport
    sa.meses_br, sa.divisor_br, sa.ipc = ESC25.meses_br, ESC25.divisor_br, ESC25.ipc
    sb = copy.deepcopy(sa); sb.caej = TS_25; sb.tc = tc_2025
    sd = copy.deepcopy(sb); sd.i_prest = ESC25.i_prest
    se = copy.deepcopy(sd); se.exencion = True
    pasos = [("2020 (63-70)", b0, ESC20.e65),
             ("(a) año de jubilación, IPC, base reguladora y crecimiento pasado del PIB", sa, ESC20.e65),
             ("(b) tasas de sustitución y tipo de cotización", sb, ESC20.e65),
             ("(c) esperanza de vida 21,52", sb, ESC25.e65),
             ("(d) crecimiento futuro del PIB 1,22%", sd, ESC25.e65),
             ("exención de cuotas", se, ESC25.e65),
             ("esperanza de vida de 2045 (22,63)", se, 22.63)]
    desc, prev = [], None
    for nombre, esc, e65 in pasos:
        M = malla_W(esc, e65)
        m, mp = float(M.mean()), media_ponderada(M, PESOS_63_70)[0]
        desc.append({"paso": nombre, "media": m, "media_pond": mp,
                     "var_media": None if prev is None else 100 * (m / prev[0] - 1),
                     "var_pond": None if prev is None else 100 * (mp / prev[1] - 1)})
        prev = (m, mp)
    res["descomposicion"] = desc
    res["descomposicion_IAE"] = {"2020 (63-70)": 1.551, "(a)": 1.582, "(b)": 1.528, "(c)": 1.552,
                                 "(d)": 1.619, "(d) sin exención": 1.613}

    # 5b) Contraste del reescalado: la tabla aproximada se proyecta con las mejoras de la
    # PER2020 (factor de escala de 2020 fijo); Weibull solo recibe su e65. Supuestos de 2025.
    contraste = []
    for T in (2025, 2035, 2045, 2065):
        esc = copy.deepcopy(ESC25); esc.anio_jub = T
        tm, p = mensualizar(kpx_tabla(per, 65, T, c20)); e65_T = e_x(tm, p)
        A = fdea_desde_renta(rentas_tabla(per, esc, c20), esc)
        l = WeibullPaper(*wp.th); l.phi = wp_.phi * e65_T / ESC20.e65
        Wr = fdea_desde_renta(rentas_ley(l, esc), esc)
        W0 = fdea_desde_renta(rentas_ley(wp_, esc), esc)
        contraste.append(dict(anio=T, e65=e65_T, MAPE_reescalado=100 * np.mean(np.abs(Wr / A - 1)),
                              max_reescalado=100 * np.max(np.abs(Wr / A - 1)),
                              MAPE_sin_actualizar=100 * np.mean(np.abs(W0 / A - 1))))
    res["contraste_reescalado"] = contraste

    # 5c) Factor de ajuste por longevidad: coeficiente sobre la pensión inicial que mantiene el
    # FdEA cuando e65 pasa de 21,52 (2025) a e65_T, con los supuestos financieros de 2025.
    def renta_W(e65):
        l = WeibullPaper(*wp.th); l.phi = wp_.phi * e65 / ESC20.e65
        return rentas_ley(l, ESC25)
    r0 = renta_W(ESC25.e65)
    res["factor_ajuste"] = {str(e): {str(x): r0[x] / r[x] for x in ESC25.edades}
                            for e, r in ((e, renta_W(e)) for e in (22.52, 22.63, 23.11))}

    # 5d) Contraste del reescalado con todas las leyes (sección 4.5): cada ley, estimada en 2020,
    # se actualiza solo con la e65 de la tabla proyectada. Error medio y máximo (%) de su malla
    # frente a la de la tabla. Para la ley de Weibull (Ec. 7-8) coincide con 5b.
    def e65_proy(T):
        return e_x(*mensualizar(kpx_tabla(per, 65, T, c20)))

    contraste_leyes = []
    for T in (2025, 2035, 2045, 2065):
        esc = copy.deepcopy(ESC25); esc.anio_jub = T
        e65_T = e65_proy(T)
        A = fdea_desde_renta(rentas_tabla(per, esc, c20), esc)
        fila = {"anio": T, "e65": e65_T}
        for nombre, l20 in estados20.items():
            l = copy.deepcopy(l20); l.reescalar(e65_T)
            M = fdea_desde_renta(rentas_ley(l, esc), esc)
            fila[nombre] = {"MAPE": float(100 * np.mean(np.abs(M / A - 1))),
                            "max": float(100 * np.max(np.abs(M / A - 1)))}
        contraste_leyes.append(fila)
    res["contraste_leyes"] = contraste_leyes

    # 5e) Factor de ajuste con la tabla proyectada (sección 4.6): la e65 de la tabla proyectada
    # pasa de su valor en 2025 al de un año más (año fraccionario en que lo alcanza). Se compara
    # el factor de la tabla con el de cada ley actualizada con esas dos e65. Supuestos de 2025.
    e0 = e65_proy(ESC25.anio_jub)
    T1 = brentq(lambda T: e65_proy(T) - (e0 + 1), ESC25.anio_jub, 2120)
    esc1 = copy.deepcopy(ESC25); esc1.anio_jub = T1
    A0, A1 = rentas_tabla(per, ESC25, c20), rentas_tabla(per, esc1, c20)
    fal = {"e65_inicial": e0, "e65_final": e0 + 1, "anio_final": T1,
           "Tabla proyectada": {str(x): A0[x] / A1[x] for x in ESC25.edades}}
    for nombre, l20 in estados20.items():
        L0 = copy.deepcopy(l20); L0.reescalar(e0)
        L1 = copy.deepcopy(l20); L1.reescalar(e0 + 1)
        ra, rb = rentas_ley(L0, ESC25), rentas_ley(L1, ESC25)
        fal[nombre] = {str(x): ra[x] / rb[x] for x in ESC25.edades}
    res["factor_ajuste_tabla_proyectada"] = fal

    # 5f) Ganancia de esperanza de vida entre 2025 y 2045 por edad de jubilación (sección 4.5),
    # en la tabla proyectada y en cada ley actualizada con la e65 de la tabla en esos dos años.
    e45 = e65_proy(2045)

    def ex_ley(l, x):   # sin truncar, como la e65 con la que se actualiza cada ley
        with np.errstate(over="ignore"):    # exp(-inf) = 0 en la cola: sin efecto en el resultado
            return quad(lambda t: l.tpx(x, t), 0, np.inf, limit=400)[0]

    gan = {"Tabla proyectada": {str(x): e_x(*mensualizar(kpx_tabla(per, x, 2045, c20)))
                                - e_x(*mensualizar(kpx_tabla(per, x, 2025, c20)))
                                for x in EDADES_20}}
    for nombre, l20 in estados20.items():
        L0 = copy.deepcopy(l20); L0.reescalar(e0)
        L1 = copy.deepcopy(l20); L1.reescalar(e45)
        gan[nombre] = {str(x): ex_ley(L1, x) - ex_ley(L0, x) for x in EDADES_20}
    res["ganancia_ex_2025_2045"] = gan

    # 6) Exportación
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active; ws.title = "metricas"
    ws.append(list(dfm.columns))
    for r in dfm.itertuples(index=False):
        ws.append(list(r))
    ws2 = wb.create_sheet("base_reguladora")
    ws2.append(["escenario", "meses", "divisor", "IPC año -2", "IPC año -1", "g real", "BR",
                "BR todo real", "factor"])
    for k, d in res["base_reguladora"].items():
        ws2.append([k, d["meses"], d["divisor"], *d["ipc"], d["g_sal"], round(d["BR"], 6),
                    round(d["BR_todo_real"], 6), round(d["factor_nominal"], 6)])
    ws3 = wb.create_sheet("ponderados_2020"); ws3.append(["malla", "FdEA medio", "desv. típica"])
    for k, (m, d) in res["ponderados_2020"].items():
        ws3.append([k, round(m, 4), round(d, 4)])
    ws6 = wb.create_sheet("descomposicion")
    ws6.append(["paso", "media simple", "variación (%)", "media ponderada", "variación (%)"])
    for d in desc:
        ws6.append([d["paso"], round(d["media"], 4), None if d["var_media"] is None else round(d["var_media"], 2),
                    round(d["media_pond"], 4), None if d["var_pond"] is None else round(d["var_pond"], 2)])
    ws4 = wb.create_sheet("contraste_reescalado"); ws4.append(list(res["contraste_reescalado"][0].keys()))
    for d in res["contraste_reescalado"]:
        ws4.append([round(v, 4) for v in d.values()])
    ws5 = wb.create_sheet("factor_ajuste"); ws5.append(["e65"] + ESC25.edades)
    for e, d in res["factor_ajuste"].items():
        ws5.append([float(e)] + [round(d[str(x)], 4) for x in ESC25.edades])
    nombres = list(estados20)
    ws7 = wb.create_sheet("contraste_leyes")
    ws7.append(["año", "e65"] + [f"{n}: error {k}" for n in nombres for k in ("medio (%)", "máximo (%)")])
    for d in contraste_leyes:
        ws7.append([d["anio"], round(d["e65"], 4)] + [round(d[n][k], 4) for n in nombres for k in ("MAPE", "max")])
    ws8 = wb.create_sheet("factor_ajuste_tabla")
    coma = lambda v, d: f"{v:.{d}f}".replace(".", ",")
    ws8.append([f"e65 de la tabla proyectada: {coma(fal['e65_inicial'], 4)} en 2025 y "
                f"{coma(fal['e65_final'], 4)} en el año {coma(fal['anio_final'], 2)}"])
    ws8.append(["modelo"] + ESC25.edades)
    for n in ["Tabla proyectada"] + nombres:
        ws8.append([n] + [round(fal[n][str(x)], 4) for x in ESC25.edades])
    ws9 = wb.create_sheet("ganancia_ex_2025_2045"); ws9.append(["modelo"] + EDADES_20)
    for n, d in gan.items():
        ws9.append([n] + [round(d[str(x)], 4) for x in EDADES_20])
    os.makedirs("mallas", exist_ok=True)
    todas = {"IAE_2020": FDEA_IAE_20, "IAE_2025": FDEA_IAE_25, **tablas}
    for nombre, m in todas.items():
        edades = EDADES_20 if nombre.endswith("2020") else EDADES_25
        seguro = nombre.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        hoja_malla(wb.create_sheet(seguro[:31]), m, edades)
        malla_png(m, edades, nombre.replace("_", " "), f"mallas/{seguro}.png")
        if not nombre.startswith(("IAE", "Sens")):
            ref = FDEA_IAE_20 if nombre.endswith("2020") else FDEA_IAE_25
            d = ref - m
            hoja_malla(wb.create_sheet(("dif_" + seguro)[:31]), d, edades, invertir=True, colorear=np.abs(d))
            malla_png(d, edades, "Diferencia IAE − " + nombre.replace("_", " "),
                      f"mallas/dif_{seguro}.png", invertir=True, colorear=np.abs(d))
    wb.save(salida)
    res["tablas"] = {k: np.round(v, 4).tolist() for k, v in todas.items()}
    with open("resumen.json", "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("base_reguladora", "factor_qx", "parametros", "sensibilidad",
                                          "elasticidad_e65", "ponderados_2020", "sin_calibrar_2020",
                                          "descomposicion", "contraste_reescalado", "factor_ajuste",
                                          "contraste_leyes", "factor_ajuste_tabla_proyectada",
                                          "ganancia_ex_2025_2045")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
