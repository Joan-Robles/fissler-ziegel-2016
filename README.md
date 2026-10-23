# fissler-ziegel-2016

Proyecto sobre el artículo:

> Fissler, T. y Ziegel, J. F. (2016). *Higher order elicitability and Osband's principle*. The Annals of Statistics, 44(4), 1680–1707.

El artículo muestra que, aunque el Expected Shortfall (ES) no es elicitable por sí solo, el par (VaR, ES) sí es conjuntamente elicitable, y caracteriza la familia de funciones de scoring consistentes para ese par.

## Contenido

| Archivo | Descripción |
|---|---|
| `var_es_fissler_ziegel.ipynb` | Notebook principal, ejecutado y con todas las salidas. |
| `requirements.txt` | Paquetes y versiones con los que se ejecutó el notebook. |

Los datos (precios diarios ajustados de SPY, 2010-01-04 a 2025-12-31) no se incluyen en el repositorio. El notebook los descarga de Yahoo Finance en la primera ejecución.

El notebook estima VaR y ES diarios de SPY con $\alpha = 0{,}05$ y $0{,}01$ (95 % y 99 %) mediante simulación histórica y un modelo normal, ambos con ventana móvil de 500 días. Evalúa los pronósticos fuera de muestra (2016–2025) de dos maneras:

1. Calibración absoluta del VaR: pruebas de Kupiec y Christoffersen, con p-valores asintóticos, exactos y de Monte Carlo.
2. Ranking de modelos con el score FZ0, un miembro estrictamente consistente de la familia del Corolario 5.5, y la prueba de Diebold-Mariano.

La sección 4 desarrolla la teoría con verificaciones numéricas: los conjuntos de nivel no convexos del ES, la descomposición de la prueba del Corolario 5.5, el principio de Osband y la simulación histórica como M-estimador del score FZ0.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace var_es_fissler_ziegel.ipynb
```

En la primera ejecución el notebook descarga los precios con `yfinance` y los guarda en `data/spy_2010_2025.csv`; las ejecuciones siguientes leen ese archivo. Para forzar una descarga nueva, cambie `FORCE_DOWNLOAD = True` en la celda de configuración.

Las salidas guardadas en el notebook corresponden a datos descargados el 22 de octubre de 2026. Una descarga posterior reproduce esos precios mientras Yahoo no recalcule los precios ajustados; si lo hace, las cifras pueden variar ligeramente.

## Licencia y cita

El contenido de este repositorio (código, texto y figuras) se distribuye bajo la licencia [Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](LICENSE). Puede usarlo, modificarlo y redistribuirlo, incluso con fines comerciales, siempre que dé crédito al autor, enlace la licencia e indique si hizo cambios.

Cita sugerida:

> Robles, J. (2026). *VaR y ES del S&P 500: simulación histórica, backtesting y elicitabilidad conjunta (Fissler y Ziegel, 2016)* [Repositorio de GitHub]. https://github.com/Joan-Robles/fissler-ziegel-2016

Los datos que descarga el notebook pertenecen a Yahoo Finance y no están cubiertos por esta licencia.
