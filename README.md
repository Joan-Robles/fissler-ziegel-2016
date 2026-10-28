# fissler-ziegel-2016

Proyecto sobre el artículo:

> Fissler, T. y Ziegel, J. F. (2016). *Higher order elicitability and Osband's principle*. The Annals of Statistics, 44(4), 1680–1707.

El artículo muestra que, aunque el Expected Shortfall (ES) no es elicitable por sí solo, el par (VaR, ES) sí es conjuntamente elicitable, y caracteriza la familia de funciones de scoring consistentes para ese par.

## Contenido

| Archivo | Descripción |
|---|---|
| `var_es_fissler_ziegel.ipynb` | Notebook ejecutado, con todas las salidas. Solo llama a los módulos. |
| `var_es.py` | Cálculos: datos, VaR/ES, backtesting, score FZ0, Diebold-Mariano y verificaciones. |
| `graficos.py` | Figuras del notebook. |
| `tests/test_var_es.py` | Pruebas automáticas (pytest). |
| `requirements.txt` | Paquetes y versiones con los que se ejecutó el notebook. |

El notebook sigue el orden del artículo:

1. **Introducción y datos.** SPY diario 2010–2025, con 2016–2025 como muestra de evaluación.
2. **Elicitabilidad de orden superior.** Definiciones; el VaR es 1-elicitable y el ES no (conjuntos de nivel no convexos).
3. **Principio de Osband.** Verificación numérica de $
abla ar S = h(x)ar V$.
4. **Componentes elicitables: el VaR.** Simulación histórica y normal con ventana de 500 días; backtesting de Kupiec y Christoffersen.
5. **Medidas de riesgo espectrales: el par (VaR, ES).** Corolario 5.5, score FZ0, M-estimación y ranking con Diebold-Mariano.
6. **Discusión.** Tabla final, conclusiones y verificaciones independientes.

Los datos no se incluyen en el repositorio: el notebook los descarga de Yahoo Finance en la primera ejecución.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace var_es_fissler_ziegel.ipynb
python -m pytest tests
```

En la primera ejecución se descargan los precios con `yfinance` y se guardan en `data/spy_2010_2025.csv`; las ejecuciones siguientes leen ese archivo (`ve.load_prices(force=True)` fuerza una descarga nueva). Las salidas guardadas corresponden a datos descargados el 22 de octubre de 2026; si Yahoo recalcula los precios ajustados, las cifras pueden variar ligeramente.

## Licencia y cita

El contenido de este repositorio (código, texto y figuras) se distribuye bajo la licencia [Creative Commons Atribución 4.0 Internacional (CC BY 4.0)](LICENSE). Puede usarlo, modificarlo y redistribuirlo, incluso con fines comerciales, siempre que dé crédito al autor, enlace la licencia e indique si hizo cambios.

Cita sugerida:

> Robles, J. (2026). *VaR y ES del S&P 500: simulación histórica, backtesting y elicitabilidad conjunta (Fissler y Ziegel, 2016)* [Repositorio de GitHub]. https://github.com/Joan-Robles/fissler-ziegel-2016

Los datos que descarga el notebook pertenecen a Yahoo Finance y no están cubiertos por esta licencia.
