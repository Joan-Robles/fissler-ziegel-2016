"""Gráficos del proyecto. Cada función devuelve la figura de matplotlib."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy import stats

import var_es as ve

C_HS, C_N, C_L = "#1f4e79", "#c55a11", "#7f7f7f"
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})
COLOR = {"HS": C_HS, "Normal": C_N}


def _ylim_perdidas(ax):
    ax.set_ylim(-6, 12)
    ax.set_ylabel("% del valor")


def precio_rendimientos(price, r):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(price, color=C_HS, lw=1)
    ax[0].set_yscale("log")
    ax[0].set_yticks([75, 100, 150, 200, 300, 400, 600])
    ax[0].yaxis.set_major_formatter(ScalarFormatter())
    ax[0].yaxis.set_minor_formatter(plt.NullFormatter())
    ax[0].set_ylabel("Precio ajustado (USD, log)")
    ax[1].plot(r, color=C_L, lw=0.6)
    ax[1].set_ylabel("Y_t = r_t (%)")
    for a in ax:
        a.axvline(pd.Timestamp(ve.OOS_START), color="k", ls="--", lw=1)
    ax[1].text(pd.Timestamp(ve.OOS_START), r.max() * 0.9, "  inicio de la evaluación", fontsize=9)
    ax[0].set_title("SPY: precio ajustado y rendimiento logarítmico diario")
    fig.tight_layout()
    return fig


def distribucion_perdidas(L):
    mu, sd = L.mean(), L.std(ddof=1)
    grid = np.linspace(L.min(), L.max(), 600)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(L, bins=120, density=True, color=C_L, alpha=0.6, label="Pérdidas observadas")
    ax[0].plot(grid, stats.norm.pdf(grid, mu, sd), color=C_N, lw=1.5, label="Normal con la misma media y desv.")
    for a, ls in ((0.05, "--"), (0.01, "-")):
        ax[0].axvline(np.quantile(L, 1 - a, method="inverted_cdf"), color=C_L, ls=ls, lw=1,
                      label=f"cuantil {ve.nivel(a)} empírico")
        ax[0].axvline(mu + sd * stats.norm.ppf(1 - a), color=C_N, ls=ls, lw=1, label=f"cuantil {ve.nivel(a)} normal")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-4, 1)
    ax[0].set_xlabel("L_t (%)")
    ax[0].set_title("Densidad de las pérdidas (escala log)")
    ax[0].legend(fontsize=7, loc="upper left")
    stats.probplot(L, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(markersize=2, color=C_L)
    ax[1].get_lines()[1].set(color=C_N)
    ax[1].set_title("Gráfico QQ contra la normal")
    ax[1].set_xlabel("Cuantiles teóricos N(0,1)")
    ax[1].set_ylabel("Cuantiles observados de L_t")
    fig.tight_layout()
    return fig


def mezcla_discreta():
    lams = np.linspace(0, 1, 301)
    path = np.array([ve.var_es_discrete(*ve.mezcla(ve.F0_DISCRETA, ve.F1_DISCRETA, lam), 0.5) for lam in lams])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(lams, path[:, 0], color=C_HS, drawstyle="steps-post")
    ax[0].set_title("VaR_0.5(F_λ): el cuantil de la mezcla salta")
    ax[1].plot(lams, path[:, 1], color=C_N, label="ES_0.5(F_λ)")
    ax[1].axhline(-10, color="k", ls=":", label="t = -10, ES de F0 y de F1")
    ax[1].set_title("ES_0.5(F_λ): el conjunto de nivel no es convexo")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("λ")
    fig.tight_layout()
    return fig


def mezclas_continuas(camino):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    specs = [("F2", "ES", "F1 y F2: mismo ES"), ("F3", "VaR", "F1 y F3: mismo VaR"),
             ("F4", None, "F1 y F4 (t de Student): mismo par")]
    for a, (k, meas, title) in zip(ax, specs):
        d = camino[camino["pareja"] == k]
        if meas is None:
            a.plot(d["lambda"], d["VaR"], color=C_HS, label="VaR")
            a.plot(d["lambda"], d["ES"], color=C_N, label="ES")
            a.legend(fontsize=8)
        else:
            a.plot(d["lambda"], d[meas], color=C_N if meas == "ES" else C_HS)
            base = d[meas].iloc[0]
            a.axhline(base, color="k", ls=":")
            if meas == "VaR":
                a.set_ylim(base - 0.05, base + 0.05)
            a.set_ylabel(meas)
        a.set_title(title, fontsize=10)
        a.set_xlabel("λ")
    fig.tight_layout()
    return fig


def osband(osb):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, (g, h, lab) in zip(ax, [("grad1", "hV1", "componente x1 (VaR)"), ("grad2", "hV2", "componente x2 (ES)")]):
        a.scatter(osb[h], osb[g], s=14, color=C_HS)
        lim = [min(osb[h].min(), osb[g].min()), max(osb[h].max(), osb[g].max())]
        a.plot(lim, lim, color="k", lw=0.8, ls=":")
        a.set_xlabel(r"$h(x)\,\bar V(x, F)$")
        a.set_ylabel(r"$\partial \bar S / \partial x$ (diferencias finitas)")
        a.set_title(lab)
    fig.suptitle(r"Principio de Osband: $\nabla \bar S = h(x)\,\bar V(x, F)$ en 40 reportes al azar", fontsize=11)
    fig.tight_layout()
    return fig


def pronosticos(fc, L_eval, models=("HS",)):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, a in zip(axes, ve.LEVELS):
        ax.plot(L_eval, color=C_L, lw=0.5, label="Pérdida realizada")
        for m in models:
            ax.plot(fc[m][a]["VaR"], color=COLOR[m], lw=1.2, label=f"VaR {m}")
            ax.plot(fc[m][a]["ES"], color=COLOR[m], lw=1.2, ls="--", label=f"ES {m}")
        _ylim_perdidas(ax)
        ax.set_title(f"Pronósticos a un día, {ve.nivel(a)} (α = {a})")
        ax.legend(loc="upper left", fontsize=8, ncol=5)
    fig.tight_layout()
    return fig


def excepciones(fc, L_eval, model="HS"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, a in zip(axes, ve.LEVELS):
        df = fc[model][a]
        hit = L_eval > df["VaR"]
        ax.plot(L_eval, color=C_L, lw=0.5, label="Pérdida realizada")
        ax.plot(df["VaR"], color=COLOR[model], lw=1.2, label=f"VaR {ve.nivel(a)}")
        ax.scatter(L_eval.index[hit], L_eval[hit], color="crimson", s=12, zorder=3,
                   label=f"Excepciones ({hit.sum()}; esperadas {len(hit) * a:.1f})")
        _ylim_perdidas(ax)
        ax.set_title(f"{model}: excepciones del VaR {ve.nivel(a)}")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def excepciones_por_anio(hy):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, a in zip(axes, ve.LEVELS):
        lab = ve.nivel(a)
        years = hy.index.astype(str)
        x = np.arange(len(years))
        ax.bar(x - 0.2, hy[("HS", lab)], width=0.4, color=C_HS, label="HS")
        ax.bar(x + 0.2, hy[("Normal", lab)], width=0.4, color=C_N, label="Normal")
        ax.step(np.r_[x - 0.5, x[-1] + 0.5], np.r_[hy[("Esperadas", lab)], hy[("Esperadas", lab)].iloc[-1]],
                where="post", color="k", lw=1, label="Esperadas")
        ax.set_xticks(x, years, rotation=45)
        ax.set_title(f"Excepciones por año, VaR {lab}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def superficie(sup):
    t1, t2 = sup["t"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].contour(sup["x1"], sup["x2"], sup["Z"], levels=25, cmap="viridis")
    ax[0].plot(t1, t2, "r*", ms=12, label="t = (VaR, ES) verdadero")
    ax[0].plot(sup["x1"], sup["x2_star"], color="k", lw=1, ls="--", label=r"mejor $x_2$ para cada $x_1$ ($\bar V_2 = 0$)")
    ax[0].set_xlabel("x1 (reporte de VaR)")
    ax[0].set_ylabel("x2 (reporte de ES)")
    ax[0].set_title("Score FZ0 esperado, Y ~ t_4, α = 0.05")
    ax[0].legend(fontsize=8, loc="upper left")
    Z = sup["Z"]
    ax[1].plot(sup["x1"], Z[np.argmin(np.abs(sup["x2"] - t2))], color=C_HS, label="x2 = t2, x1 variable")
    ax[1].plot(sup["x2"], Z[:, np.argmin(np.abs(sup["x1"] - t1))], color=C_N, label="x1 = t1, x2 variable")
    ax[1].axvline(t1, color=C_HS, ls=":")
    ax[1].axvline(t2, color=C_N, ls=":")
    ax[1].set_xlabel("valor del reporte")
    ax[1].set_ylabel(r"$\bar S_{FZ0}$")
    ax[1].set_title("Cortes del score esperado")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def descomposicion_r(dec):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = dec["bregman"] + dec["cruzado"]
    ax[0].scatter(r, dec["dS"], s=14, color=C_HS)
    lim = [0, max(r.max(), dec["dS"].max())]
    ax[0].plot(lim, lim, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("R1 + R2 (Bregman + término cruzado)")
    ax[0].set_ylabel(r"$\bar S(x, F) - \bar S(t, F)$")
    ax[0].set_title("La diferencia de score es R1 + R2")
    order = np.argsort(r.to_numpy())
    ax[1].bar(np.arange(len(dec)), dec["bregman"].to_numpy()[order], color=C_N, label="Distancia de Bregman")
    ax[1].bar(np.arange(len(dec)), dec["cruzado"].to_numpy()[order], bottom=dec["bregman"].to_numpy()[order],
              color=C_HS, label="Término cruzado (VaR)")
    ax[1].set_xlabel("reportes ordenados por R1 + R2")
    ax[1].set_title("Ambos componentes son ≥ 0")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def perfil_mestimacion(perfiles):
    fig, axes = plt.subplots(1, len(perfiles), figsize=(12, 4))
    for ax, (a, p) in zip(np.atleast_1d(axes), perfiles.items()):
        ax.plot(p["v"], p["score"], color=C_HS)
        ax.axvspan(*p["intervalo"], color=C_N, alpha=0.2, label="[L_(k), L_(k+1))")
        ax.axvline(p["v_hs"], color=C_N, ls="--", lw=1, label=f"VaR HS = {p['v_hs']:.3f}")
        ax.set_xlabel("v (VaR reportado), con e óptimo para cada v")
        ax.set_ylabel("FZ0 promedio en la ventana")
        ax.set_title(f"{ve.nivel(a)}: ES óptimo = ES HS = {p['e_hs']:.4f}", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def delta_acumulado(scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, a in zip(axes, ve.LEVELS):
        d = scores[a]["HS"] - scores[a]["Normal"]
        ax.plot(d.cumsum(), color="k", lw=1.2)
        ax.axhline(0, color=C_L, lw=0.8)
        ax.set_title(f"Suma acumulada de ΔS_t = S(HS) - S(Normal), {ve.nivel(a)}")
        ax.set_ylabel("< 0: HS acumula mejor score")
    fig.tight_layout()
    return fig


def descomposicion_score(dec):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lab in zip(axes, [ve.nivel(a) for a in ve.LEVELS]):
        d = dec[dec["nivel"] == lab].set_index("modelo")
        x = np.arange(len(d))
        ax.bar(x, d["sin excepción"], color=C_L, label="días sin excepción")
        ax.bar(x, d["con excepción"], bottom=d["sin excepción"], color="crimson", label="días con excepción")
        ax.set_xticks(x, d.index)
        ax.set_ylabel("aporte al FZ0 promedio")
        ax.set_title(f"Descomposición del FZ0 promedio, {lab}")
        ax.set_ylim(0, 1.25 * (d["sin excepción"] + d["con excepción"]).max())
        ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def delta_anual(scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, a in zip(axes, ve.LEVELS):
        d = (scores[a]["HS"] - scores[a]["Normal"]).groupby(scores[a].index.year).mean()
        ax.bar(d.index.astype(str), d, color=np.where(d < 0, C_HS, C_N))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"ΔS medio por año, {ve.nivel(a)} (azul: gana HS)")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def robustez(rob):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, lab in zip(axes, [ve.nivel(a) for a in ve.LEVELS]):
        d = rob[rob["nivel"] == lab]
        y = np.arange(len(d))[::-1]
        ax.barh(y, d["DM"], color=np.where(d["p bilateral"] < 0.05, C_HS, C_L))
        ax.axvline(-1.96, color="k", ls=":", lw=1)
        ax.set_yticks(y, d["caso"])
        ax.set_xlabel("estadístico DM (< -1.96: HS mejor al 5 %)")
        ax.set_title(f"Robustez del ranking, {lab}")
    fig.tight_layout()
    return fig


def backtesting(bt):
    """Tasa de excepciones frente a alpha (Kupiec) y pi01 frente a pi11 (independencia)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    labels = [f"{m}\n{n}" for m, n in bt.index]
    x = np.arange(len(bt))
    colors = [COLOR[m] for m, _ in bt.index]
    ax[0].bar(x, 100 * bt["tasa"].astype(float), color=colors)
    exp = [100 * (1 - float(n.rstrip("%")) / 100) for _, n in bt.index]
    ax[0].scatter(x, exp, marker="_", s=900, color="k", zorder=3, label="α esperado")
    for xi, (t, p) in enumerate(zip(bt["tasa"], bt["p Kupiec"])):
        ax[0].text(xi, 100 * t + 0.1, f"p = {p:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("tasa de excepciones (%)")
    ax[0].set_title("Cobertura incondicional (Kupiec)")
    ax[0].set_ylim(0, 1.3 * 100 * bt["tasa"].astype(float).max())
    ax[0].legend(fontsize=8, loc="upper center")
    w = 0.38
    ax[1].bar(x - w / 2, 100 * bt["pi01"].astype(float), width=w, color=C_L, label="π01: tras un día sin excepción")
    ax[1].bar(x + w / 2, 100 * bt["pi11"].astype(float), width=w, color="crimson", label="π11: tras una excepción")
    for xi, (v, p) in enumerate(zip(bt["pi11"], bt["p independencia"])):
        ax[1].text(xi + w / 2, 100 * v + 0.3, f"p = {p:.4f}", ha="center", fontsize=8)
    ax[1].set_xticks(x, labels)
    ax[1].set_ylabel("probabilidad de excepción (%)")
    ax[1].set_title("Independencia (Christoffersen)")
    ax[1].set_ylim(0, 1.35 * 100 * bt["pi11"].astype(float).max())
    ax[1].legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def ranking(score_tab):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(score_tab))
    w = 0.38
    ax.bar(x - w / 2, score_tab["FZ0 HS"], width=w, color=C_HS, label="HS")
    ax.bar(x + w / 2, score_tab["FZ0 Normal"], width=w, color=C_N, label="Normal")
    for xi, (_, r) in enumerate(score_tab.iterrows()):
        top = max(r["FZ0 HS"], r["FZ0 Normal"])
        ax.text(xi, top + 0.05, f"DM = {r['DM']:.2f}\np = {r['p bilateral']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, score_tab.index)
    ax.set_ylabel("FZ0 promedio (menor es mejor)")
    ax.set_ylim(0, 1.25 * score_tab[["FZ0 HS", "FZ0 Normal"]].to_numpy().max())
    ax.set_title("Score FZ0 promedio fuera de muestra")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def severidad(er):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(er))
    w = 0.38
    ax.bar(x - w / 2, er["Pérdida media en excepción"], width=w, color=C_L, label="pérdida media en excepción")
    ax.bar(x + w / 2, er["ES medio reportado"], width=w, color=[COLOR[m] for m, _ in er.index],
           label="ES reportado esos días (color del modelo)")
    for xi, (_, r) in enumerate(er.iterrows()):
        ax.text(xi, r["Pérdida media en excepción"] + 0.08, f"×{r['Razón']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{m}\n{n}" for m, n in er.index])
    ax.set_ylabel("% del valor")
    ax.set_title("Severidad de la cola: pérdida realizada frente al ES")
    ax.set_ylim(0, 1.3 * er["Pérdida media en excepción"].max())
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig
