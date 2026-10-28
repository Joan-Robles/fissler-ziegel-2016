"""Pruebas de var_es.py. Ejecutar desde la raíz: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import var_es as ve  # noqa: E402


@pytest.fixture(scope="module")
def datos():
    import os
    os.chdir(ROOT)
    price = ve.load_prices()["Close"]
    _, L = ve.returns_and_losses(price)
    fc, L_eval = ve.rolling_forecasts(L)
    return L, fc, L_eval, ve.scores_fz0(fc, L_eval)


def test_verificaciones_independientes(datos):
    tabla = ve.verificaciones(*datos)
    assert (tabla["diferencia abs."] < 1e-5).all(), tabla[tabla["diferencia abs."] >= 1e-5]


def test_es_con_empates():
    toy = np.array([0, 1, 2, 3, 3, 3, 3, 3, 5, 10.0])
    v, e = ve.var_es_hs(toy, 0.3)
    assert v == 3 and e == pytest.approx(6.0)


def test_contraejemplo_discreto():
    for lam, esperado in [(0.4, -10.4), (0.5, -10.5)]:
        assert ve.var_es_discrete(*ve.mezcla(ve.F0_DISCRETA, ve.F1_DISCRETA, lam), 0.5)[1] == pytest.approx(esperado)


def test_kupiec_christoffersen_a_mano():
    h = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    n, x, a = 20, 4, 0.1
    lr_uc = -2 * np.log(((1 - a) ** (n - x) * a**x) / ((1 - x / n) ** (n - x) * (x / n) ** x))
    pi01, pi11, pi1 = 3 / 15, 1 / 4, 4 / 19
    lr_ind = -2 * np.log(((1 - pi1) ** 15 * pi1**4) / ((1 - pi01) ** 12 * pi01**3 * (1 - pi11) ** 3 * pi11))
    assert ve.kupiec_test(h, a)["LR_uc"] == pytest.approx(lr_uc)
    assert ve.christoffersen_test(h, a)["LR_ind"] == pytest.approx(lr_ind)


def test_resultados_principales(datos):
    tabla = ve.score_table(datos[3])
    assert tabla.loc["95%", "DM"] == pytest.approx(-2.5043, abs=1e-3)
    assert tabla.loc["99%", "DM"] == pytest.approx(-2.7797, abs=1e-3)


def test_dm_contra_statsmodels(datos):
    sm = pytest.importorskip("statsmodels.api")
    s = datos[3][0.05]
    d = (s["HS"] - s["Normal"]).to_numpy()
    res = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 8, "use_correction": False})
    assert ve.dm_test(d)["DM"] == pytest.approx(float(res.tvalues[0]))
