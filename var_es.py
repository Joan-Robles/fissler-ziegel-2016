"""Cálculos del proyecto: datos, VaR/ES, backtesting y score FZ0 (Fissler y Ziegel, 2016).

Convención: alpha es la probabilidad de cola (0.05, 0.01). En la teoría la variable
es el rendimiento Y y el par se escribe x = (x1, x2) = (VaR, ES), ambos negativos.
En la aplicación se reportan pérdidas L = -Y y VaR/ES positivos: v = -x1, e = -x2.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import integrate, optimize, stats
from scipy.special import xlogy

TICKER = "SPY"
START, END = "2010-01-01", "2026-01-01"  # END exclusivo en yfinance
OOS_START = "2016-01-01"
WINDOW = 500
LEVELS = (0.05, 0.01)
DATA_FILE = Path("data") / "spy_2010_2025.csv"


def nivel(a):
    """Etiqueta de confianza para una probabilidad de cola: 0.05 -> '95%'."""
    return f"{round(100 * (1 - a), 2):g}%"


# Datos ----------------------------------------------------------------------

def load_prices(path=DATA_FILE, force=False):
    if Path(path).exists() and not force:
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    import yfinance as yf
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    Path(path).parent.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def returns_and_losses(price):
    r = (100 * np.log(price).diff()).dropna().rename("r")
    return r, (-r).rename("L")


def descriptivas(x):
    x = np.asarray(x, dtype=float)
    jb = stats.jarque_bera(x)
    return pd.Series({
        "N": x.size, "Media": x.mean(), "Desv. estándar": x.std(ddof=1),
        "Vol. anualizada": x.std(ddof=1) * np.sqrt(252), "Mínimo": x.min(),
        "Percentil 1": np.percentile(x, 1), "Percentil 5": np.percentile(x, 5),
        "Percentil 95": np.percentile(x, 95), "Percentil 99": np.percentile(x, 99),
        "Máximo": x.max(), "Asimetría": stats.skew(x), "Curtosis en exceso": stats.kurtosis(x),
        "p-valor Jarque-Bera": jb.pvalue,
    })


def ljung_box(x, lags=10):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    rho = np.array([x[k:] @ x[:-k] for k in range(1, lags + 1)]) / (x @ x)
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)


def tail_table(L, levels=LEVELS):
    mu, sd = L.mean(), L.std(ddof=1)
    rows = {}
    for a in levels:
        rows[f"Cuantil {nivel(a)} de L"] = {"Empírico": np.quantile(L, 1 - a, method="inverted_cdf"),
                                           "Normal": mu + sd * stats.norm.ppf(1 - a)}
    for thr in (4, 5):
        rows[f"Días con L > {thr}%"] = {"Empírico": int((L > thr).sum()),
                                       "Normal": len(L) * stats.norm.sf(thr, mu, sd)}
    return pd.DataFrame(rows).T


# Teoría: conjuntos de nivel (sección 2.2) ------------------------------------

def var_es_discrete(values, probs, alpha):
    """VaR y ES inferiores (convención de rendimientos) de una distribución discreta."""
    order = np.argsort(values)
    y, p = np.asarray(values, float)[order], np.asarray(probs, float)[order]
    cdf = np.cumsum(p)
    i = int(np.searchsorted(cdf, alpha - 1e-12))
    x1 = y[i]
    below = cdf[i] - p[i]
    return x1, (np.sum(y[:i] * p[:i]) + x1 * (alpha - below)) / alpha


def mezcla(F0, F1, lam):
    return np.r_[F0[0], F1[0]], np.r_[(1 - lam) * np.asarray(F0[1]), lam * np.asarray(F1[1])]


F0_DISCRETA = ([-10, -6], [0.5, 0.5])
F1_DISCRETA = ([-16, -4, 10], [0.25, 0.25, 0.5])


def var_es_from_cdf(cdf, alpha, bracket=(-60.0, 60.0)):
    """VaR y ES inferiores de una distribución continua: ES = x1 - (1/alpha) * int_{-inf}^{x1} F."""
    x1 = optimize.brentq(lambda y: cdf(y) - alpha, *bracket, xtol=1e-12)
    tail = integrate.quad(cdf, -np.inf, x1, limit=200)[0]
    return x1, x1 - tail / alpha


def demo_distribuciones(alpha=0.05, nu=4):
    """F1 = N(0,1); F2 con el mismo ES; F3 con el mismo VaR; F4 (t de Student) con el mismo par."""
    z = stats.norm.ppf(alpha)
    k = stats.norm.pdf(z) / alpha
    F1 = stats.norm(0, 1)
    F2 = stats.norm(-k * 0.5, 0.5)
    F3 = stats.norm(z * 0.5, 0.5)
    q = stats.t.ppf(alpha, nu)
    es_t = -stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1) * alpha)
    x1n, x2n = var_es_from_cdf(F1.cdf, alpha)
    s4 = (x2n - x1n) / (es_t - q)
    F4 = stats.t(nu, loc=x1n - s4 * q, scale=s4)
    return {"F1": F1, "F2": F2, "F3": F3, "F4": F4}


def camino_mezclas(alpha=0.05, lams=np.linspace(0, 1, 21)):
    """VaR y ES de las mezclas (1 - lambda) F1 + lambda Fk para k = 2, 3, 4."""
    d = demo_distribuciones(alpha)
    rows = []
    for k in ("F2", "F3", "F4"):
        for lam in lams:
            cdf = lambda y, k=k, lam=lam: (1 - lam) * d["F1"].cdf(y) + lam * d[k].cdf(y)
            x1, x2 = var_es_from_cdf(cdf, alpha)
            rows.append({"pareja": k, "lambda": lam, "VaR": x1, "ES": x2})
    return pd.DataFrame(rows)


# Score FZ0 y principio de Osband (secciones 3 y 5) --------------------------

def G2(x2):
    return -1 / x2


def calG2(x2):
    return -np.log(-x2)


T_DEMO = stats.t(4, loc=-0.05, scale=0.8)
A_DEMO = 0.05


def tail_int(x1, dist=T_DEMO):
    return integrate.quad(dist.cdf, -np.inf, x1, limit=200, epsabs=1e-13, epsrel=1e-13)[0]


def sbar_fz0(x1, x2, dist=T_DEMO, alpha=A_DEMO):
    """Score FZ0 esperado (G1 = 0, calG2 = -log(-x2)) en la convención de rendimientos."""
    if x2 >= 0 or x1 < x2:
        return np.inf
    return G2(x2) * (x2 - x1 + tail_int(x1, dist) / alpha) - calG2(x2)


def vbar(x1, x2, dist=T_DEMO, alpha=A_DEMO):
    """Función de identificación esperada del par (VaR, ES)."""
    return np.array([dist.cdf(x1) - alpha, x2 - x1 + tail_int(x1, dist) / alpha])


def puntos_demo(n=40, seed=1):
    """Reportes x = (x1, x2) alrededor del valor verdadero, dentro de A0 = {x1 >= x2}."""
    t1, t2 = var_es_from_cdf(T_DEMO.cdf, A_DEMO)
    g = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        x1, x2 = t1 + g.uniform(-0.6, 0.6), t2 + g.uniform(-0.8, 0.8)
        if x1 >= x2 and x2 < 0:
            pts.append((x1, x2))
    return (t1, t2), pts


def osband_check(n=40, eps=1e-4):
    """Gradiente numérico de S̄ frente a h(x) V̄(x, F) en puntos al azar (sección 3)."""
    _, pts = puntos_demo(n)
    rows = []
    for x1, x2 in pts:
        grad = [(sbar_fz0(x1 + eps, x2) - sbar_fz0(x1 - eps, x2)) / (2 * eps),
                (sbar_fz0(x1, x2 + eps) - sbar_fz0(x1, x2 - eps)) / (2 * eps)]
        hv = np.diag([-1 / (A_DEMO * x2), 1 / x2**2]) @ vbar(x1, x2)
        rows.append({"x1": x1, "x2": x2, "grad1": grad[0], "hV1": hv[0], "grad2": grad[1], "hV2": hv[1]})
    return pd.DataFrame(rows)


def descomposicion_check(n=40):
    """S̄(x) - S̄(t) frente a R1 + R2 = Bregman + término cruzado (Corolario 5.5)."""
    (t1, t2), pts = puntos_demo(n, seed=2)
    rows = []
    for x1, x2 in pts:
        bregman = calG2(t2) - calG2(x2) - G2(x2) * (t2 - x2)
        cross = G2(x2) / A_DEMO * integrate.quad(lambda y: T_DEMO.cdf(y) - A_DEMO, t1, x1, epsabs=1e-13)[0]
        rows.append({"x1": x1, "x2": x2, "dS": sbar_fz0(x1, x2) - sbar_fz0(t1, t2),
                     "bregman": bregman, "cruzado": cross})
    return pd.DataFrame(rows)


def superficie_fz0(n=61):
    t1, t2 = var_es_from_cdf(T_DEMO.cdf, A_DEMO)
    xx1 = np.linspace(t1 - 0.8, t1 + 0.8, n)
    xx2 = np.linspace(t2 - 0.9, t2 + 0.9, n)
    I1 = np.array([tail_int(x1) for x1 in xx1])
    Z = np.array([[G2(x2) * (x2 - x1 + I / A_DEMO) - calG2(x2) if x1 >= x2 else np.nan
                   for x1, I in zip(xx1, I1)] for x2 in xx2])
    return {"t": (t1, t2), "x1": xx1, "x2": xx2, "Z": Z, "x2_star": xx1 - I1 / A_DEMO}


def pinball(x1, y, alpha):
    return ((np.asarray(y) <= x1).astype(float) - alpha) * (x1 - y)


def fz0_score(L, var, es, alpha):
    """FZ0 en pérdidas: 1{L > v}(L - v)/(alpha e) + v/e + ln e - 1."""
    L, var, es = (np.asarray(x, dtype=float) for x in (L, var, es))
    if np.any(es <= 0):
        raise ValueError("FZ0 requiere ES > 0 en la convención de pérdidas")
    return (L > var) * (L - var) / (alpha * es) + var / es + np.log(es) - 1


def fz_general(y, x1, x2, alpha, G1, G2_, calG2_):
    """Familia del Corolario 5.5 en la convención de rendimientos."""
    ind = (y <= x1).astype(float)
    return (ind - alpha) * G1(x1) - ind * G1(y) + G2_(x2) * (x2 - x1 + ind * (x1 - y) / alpha) - calG2_(x2)


# Estimadores (sección 4) -----------------------------------------------------

def var_es_hs(losses, alpha):
    """VaR y ES por simulación histórica, con tratamiento explícito del átomo en el cuantil."""
    x = np.sort(np.asarray(losses, dtype=float))
    n = x.size
    k = int(np.ceil(n * (1 - alpha) - 1e-9))
    var = x[k - 1]
    atom = np.mean(x <= var) - (1 - alpha)
    return var, (np.sum(x[x > var]) / n + var * atom) / alpha


def es_order_stats(losses, alpha):
    x = np.sort(np.asarray(losses, dtype=float))
    n = x.size
    k = int(np.ceil(n * (1 - alpha) - 1e-9))
    return ((k - n * (1 - alpha)) * x[k - 1] + x[k:].sum()) / (n * alpha)


def es_quantile_integral(losses, alpha, m=200_000):
    u = (1 - alpha) + (np.arange(m) + 0.5) * alpha / m
    return np.quantile(np.asarray(losses, dtype=float), u, method="inverted_cdf").mean()


def normal_var_es(mu, sd, alpha):
    z = stats.norm.ppf(1 - alpha)
    return mu + sd * z, mu + sd * stats.norm.pdf(z) / alpha


def rolling_forecasts(L, window=WINDOW, oos_start=OOS_START, levels=LEVELS):
    """Pronósticos a un día con ventana móvil: dict modelo -> dict alpha -> DataFrame(VaR, ES)."""
    Lv = L.to_numpy()
    pos = np.flatnonzero(L.index >= oos_start)
    assert pos[0] >= window
    dates = L.index[pos]
    win = sliding_window_view(Lv[pos[0] - window: pos[-1]], window)
    mu, sd = win.mean(axis=1), win.std(axis=1, ddof=1)
    out = {"HS": {}, "Normal": {}}
    for a in levels:
        hs = np.array([var_es_hs(w, a) for w in win])
        out["HS"][a] = pd.DataFrame(hs, index=dates, columns=["VaR", "ES"])
        nv, ne = normal_var_es(mu, sd, a)
        out["Normal"][a] = pd.DataFrame({"VaR": nv, "ES": ne}, index=dates)
    return out, L.iloc[pos]


def resumen_pronosticos(fc):
    rows = {}
    for model, by_a in fc.items():
        for a, df in by_a.items():
            for m in ("VaR", "ES"):
                rows[(model, nivel(a), m)] = {"Media": df[m].mean(), "Mínimo": df[m].min(), "Máximo": df[m].max()}
    return pd.DataFrame(rows).T


# Backtesting (sección 4) -----------------------------------------------------

def transition_counts(hits):
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    return (int(np.sum((prev == 0) & (curr == 0))), int(np.sum((prev == 0) & (curr == 1))),
            int(np.sum((prev == 1) & (curr == 0))), int(np.sum((prev == 1) & (curr == 1))))


def lr_from_counts(x, n, n00, n01, n10, n11, alpha):
    pi_hat = x / n
    lr_uc = -2 * (xlogy(n - x, 1 - alpha) + xlogy(x, alpha) - xlogy(n - x, 1 - pi_hat) - xlogy(x, pi_hat))
    with np.errstate(invalid="ignore", divide="ignore"):
        pi01 = np.where(n00 + n01 > 0, n01 / np.maximum(n00 + n01, 1), 0.0)
        pi11 = np.where(n10 + n11 > 0, n11 / np.maximum(n10 + n11, 1), 0.0)
    pi1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_markov = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    ll_iid = xlogy(n00 + n10, 1 - pi1) + xlogy(n01 + n11, pi1)
    ll_null = xlogy(n00 + n10, 1 - alpha) + xlogy(n01 + n11, alpha)
    return lr_uc, -2 * (ll_iid - ll_markov), -2 * (ll_null - ll_markov)


def kupiec_test(hits, alpha):
    h = np.asarray(hits, dtype=int)
    n, x = h.size, int(h.sum())
    lr = lr_from_counts(x, n, *transition_counts(h), alpha)[0]
    return {"n": n, "excepciones": x, "esperadas": n * alpha, "tasa": x / n,
            "LR_uc": float(lr), "p_uc": stats.chi2.sf(lr, 1)}


def christoffersen_test(hits, alpha):
    n00, n01, n10, n11 = transition_counts(hits)
    _, lr_ind, lr_cc = lr_from_counts(int(np.sum(hits)), len(hits), n00, n01, n10, n11, alpha)
    return {"pi01": n01 / max(n00 + n01, 1), "pi11": n11 / max(n10 + n11, 1),
            "LR_ind": float(lr_ind), "p_ind": stats.chi2.sf(lr_ind, 1),
            "LR_cc": float(lr_cc), "p_cc": stats.chi2.sf(lr_cc, 2)}


def simulate_null(T, alpha, nsim=20_000, seed=0):
    g = np.random.default_rng(seed)
    H = g.random((nsim, T)) < alpha
    prev, curr = H[:, :-1], H[:, 1:]
    n11 = np.sum(prev & curr, axis=1)
    n10 = np.sum(prev & ~curr, axis=1)
    n01 = np.sum(~prev & curr, axis=1)
    n00 = (T - 1) - n11 - n10 - n01
    return np.column_stack(lr_from_counts(H.sum(axis=1), T, n00, n01, n10, n11, alpha))


def backtest_table(fc, L_eval, levels=LEVELS, nsim=20_000):
    nulls = {a: simulate_null(len(L_eval), a, nsim, seed=int(round(100 * (1 - a)))) for a in levels}
    rows = {}
    for model, by_a in fc.items():
        for a in levels:
            hits = (L_eval > by_a[a]["VaR"]).astype(int).to_numpy()
            k, c = kupiec_test(hits, a), christoffersen_test(hits, a)
            obs = (k["LR_uc"], c["LR_ind"], c["LR_cc"])
            mc = [(1 + np.sum(nulls[a][:, j] >= obs[j] - 1e-12)) / (nsim + 1) for j in range(3)]
            rows[(model, nivel(a))] = {
                "excepciones": k["excepciones"], "esperadas": k["esperadas"], "tasa": k["tasa"],
                "p Kupiec": k["p_uc"], "p binomial exacta": stats.binomtest(k["excepciones"], k["n"], a).pvalue,
                "pi01": c["pi01"], "pi11": c["pi11"], "p independencia": c["p_ind"],
                "p condicional": c["p_cc"], "p Kupiec MC": mc[0], "p independencia MC": mc[1],
                "p condicional MC": mc[2]}
    return pd.DataFrame(rows).T, nulls


def null_size(nulls):
    return pd.DataFrame({nivel(a): {
        "Kupiec": np.mean(stats.chi2.sf(lr[:, 0], 1) < 0.05),
        "Independencia": np.mean(stats.chi2.sf(lr[:, 1], 1) < 0.05),
        "Condicional": np.mean(stats.chi2.sf(lr[:, 2], 2) < 0.05)} for a, lr in nulls.items()})


def hits_by_year(fc, L_eval, levels=LEVELS):
    out = {}
    for model, by_a in fc.items():
        for a in levels:
            out[(model, nivel(a))] = (L_eval > by_a[a]["VaR"]).groupby(L_eval.index.year).sum()
    df = pd.DataFrame(out)
    days = L_eval.groupby(L_eval.index.year).size()
    for a in levels:
        df[("Esperadas", nivel(a))] = days * a
    return df


# Ranking con FZ0 (sección 5) ---------------------------------------------------

def empirical_fz0_profile(w, alpha, grid=400):
    """Score FZ0 promedio en la ventana como función de v, con e óptimo para cada v."""
    w = np.asarray(w, dtype=float)
    v_hs, e_hs = var_es_hs(w, alpha)
    vs = np.linspace(v_hs - 0.5, v_hs + 0.5, grid)
    e_star = vs + np.array([np.mean(np.maximum(w - v, 0)) for v in vs]) / alpha
    prof = np.array([np.mean(fz0_score(w, v, e, alpha)) for v, e in zip(vs, e_star)])
    xs = np.sort(w)
    k = int(np.ceil(w.size * (1 - alpha) - 1e-9))
    return {"v": vs, "score": prof, "e_star": e_star, "v_hs": v_hs, "e_hs": e_hs,
            "intervalo": (xs[k - 1], xs[k]), "score_hs": np.mean(fz0_score(w, v_hs, e_hs, alpha))}


def dm_test(d, lag=None):
    d = np.asarray(d, dtype=float)
    T = d.size
    if lag is None:
        lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    u = d - d.mean()
    lrv = u @ u / T + 2 * sum((1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / T for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(lrv / T)
    return {"media ΔS": d.mean(), "rezagos": lag, "DM": stat,
            "p bilateral": 2 * stats.norm.sf(abs(stat)), "p unilateral": stats.norm.cdf(stat)}


def scores_fz0(fc, L_eval, levels=LEVELS):
    return {a: pd.DataFrame({m: fz0_score(L_eval, fc[m][a]["VaR"], fc[m][a]["ES"], a) for m in fc},
                            index=L_eval.index) for a in levels}


def score_table(scores):
    rows = {}
    for a, s in scores.items():
        rows[nivel(a)] = {"FZ0 HS": s["HS"].mean(), "FZ0 Normal": s["Normal"].mean(),
                          **dm_test(s["HS"] - s["Normal"]), "% días HS mejor": np.mean(s["HS"] < s["Normal"])}
    return pd.DataFrame(rows).T


def score_decomposition(scores, fc, L_eval):
    rows = []
    for a, s in scores.items():
        for m in s:
            h = (L_eval > fc[m][a]["VaR"]).to_numpy()
            rows.append({"nivel": nivel(a), "modelo": m,
                         "sin excepción": s[m][~h].sum() / len(s), "con excepción": s[m][h].sum() / len(s),
                         "FZ0 por día sin excepción": s[m][~h].mean(), "FZ0 por excepción": s[m][h].mean()})
    return pd.DataFrame(rows)


def es_ratio(fc, L_eval, levels=LEVELS):
    rows = {}
    for m, by_a in fc.items():
        for a in levels:
            hit = L_eval > by_a[a]["VaR"]
            rows[(m, nivel(a))] = {"Pérdida media en excepción": L_eval[hit].mean(),
                                   "ES medio reportado": by_a[a]["ES"][hit].mean(),
                                   "Razón": L_eval[hit].mean() / by_a[a]["ES"][hit].mean()}
    return pd.DataFrame(rows).T


def robustness(scores, fc, L_eval, levels=LEVELS):
    rows = []
    y = -L_eval.to_numpy()
    for a in levels:
        d = scores[a]["HS"] - scores[a]["Normal"]
        cases = {f"HAC {lag} rezagos": (d, lag) for lag in (0, 5, 8, 15, 30)}
        cases.update({"Sin 2020": (d[d.index.year != 2020], None),
                      "2016-2019": (d[d.index.year <= 2019], None),
                      "2021-2025": (d[d.index.year >= 2021], None),
                      "Sin 10 días más favorables a HS": (d[~d.index.isin(d.nsmallest(10).index)], None)})
        d_alt = d + (pinball(-fc["HS"][a]["VaR"].to_numpy(), y, a) - pinball(-fc["Normal"][a]["VaR"].to_numpy(), y, a))
        cases["Otro miembro: G1(x) = x"] = (d_alt, None)
        for name, (dd, lag) in cases.items():
            rows.append({"nivel": nivel(a), "caso": name, **dm_test(dd, lag)})
    return pd.DataFrame(rows)


def final_table(fc, bt, scores, L, window=WINDOW, levels=LEVELS):
    w_last = L.to_numpy()[-window:]
    nxt = {"HS": lambda a: var_es_hs(w_last, a),
           "Normal": lambda a: normal_var_es(w_last.mean(), w_last.std(ddof=1), a)}
    names = {"HS": "Simulación histórica", "Normal": "Normal paramétrico"}
    out = {}
    for m in fc:
        rows = {}
        for meas, j in (("VaR", 0), ("ES", 1)):
            for a in levels:
                rows[(f"{meas} {nivel(a)}", "promedio fuera de muestra")] = fc[m][a][meas].mean()
                rows[(f"{meas} {nivel(a)}", "pronóstico 2026-01-02")] = nxt[m](a)[j]
        for col, lab in (("tasa", "Tasa de excepciones"), ("p Kupiec", "Kupiec, p-valor"),
                         ("p independencia", "Christoffersen independencia, p-valor"),
                         ("p condicional", "Christoffersen condicional, p-valor")):
            for a in levels:
                rows[(lab, nivel(a))] = bt.loc[(m, nivel(a)), col]
        for a in levels:
            rows[("Score FZ0 promedio", nivel(a))] = scores[a][m].mean()
        out[names[m]] = rows
    df = pd.DataFrame(out)
    df.index.names = ["Indicador", "Detalle"]
    return df


# Verificaciones independientes ---------------------------------------------

def verificaciones(L, fc, L_eval, scores, window=WINDOW, levels=LEVELS):
    """Contrasta cada cálculo con una forma independiente. Diferencia ~ 0 = correcto."""
    pos0 = np.flatnonzero(L.index >= OOS_START)[0]
    w0 = L.to_numpy()[pos0 - window: pos0]
    win = sliding_window_view(L.to_numpy()[pos0 - window: pos0 + len(L_eval) - 1], window)
    rows = []

    def add(nombre, a, b):
        rows.append({"verificación": nombre, "propio": float(a), "independiente": float(b),
                     "diferencia abs.": abs(float(a) - float(b))})

    for a in levels:
        lab = nivel(a)
        q = np.quantile(win, 1 - a, axis=1, method="inverted_cdf")
        add(f"VaR HS {lab}: todas las ventanas vs np.quantile (máx. dif.)",
            np.max(np.abs(fc["HS"][a]["VaR"].to_numpy() - q)), 0)
        add(f"ES HS {lab}: fórmula con átomo vs integral de la función cuantil",
            var_es_hs(w0, a)[1], es_quantile_integral(w0, a))
        mu, sd = w0.mean(), w0.std(ddof=1)
        x1, x2 = var_es_from_cdf(stats.norm(-mu, sd).cdf, a)
        add(f"ES normal {lab}: fórmula cerrada vs integración numérica", normal_var_es(mu, sd, a)[1], -x2)
        s_gen = fz_general(-L_eval.to_numpy(), -fc["HS"][a]["VaR"].to_numpy(), -fc["HS"][a]["ES"].to_numpy(), a,
                           lambda x: np.zeros_like(x), G2, calG2)
        add(f"FZ0 {lab}: forma en pérdidas vs Corolario 5.5 (máx. dif.)",
            np.max(np.abs(scores[a]["HS"].to_numpy() - s_gen)), 0)

    t1, t2 = var_es_from_cdf(T_DEMO.cdf, A_DEMO)
    opt = optimize.minimize(lambda x: sbar_fz0(x[0], x[1]), x0=[-1.0, -1.5], method="Nelder-Mead",
                            options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
    add("t_4: argmin de E[FZ0] vs VaR verdadero", opt.x[0], t1)
    add("t_4: argmin de E[FZ0] vs ES verdadero", opt.x[1], t2)
    dec = descomposicion_check(20)
    add("Corolario 5.5: S̄(x) - S̄(t) vs R1 + R2 (máx. dif.)",
        np.max(np.abs(dec["dS"] - dec["bregman"] - dec["cruzado"])), 0)
    osb = osband_check(20)
    add("Osband: gradiente numérico vs h(x) V̄ (máx. dif.)",
        np.max(np.abs(np.r_[osb["grad1"] - osb["hV1"], osb["grad2"] - osb["hV2"]])), 0)
    hits = (L_eval > fc["HS"][0.05]["VaR"]).astype(int).to_numpy()
    add("LR_cc = LR_uc(T-1) + LR_ind (HS 95%)", christoffersen_test(hits, 0.05)["LR_cc"],
        kupiec_test(hits[1:], 0.05)["LR_uc"] + christoffersen_test(hits, 0.05)["LR_ind"])
    return pd.DataFrame(rows)
