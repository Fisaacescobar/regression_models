Regresiones (lineal simple, múltiple y polinomial)

Este módulo muestra cómo entrenar y evaluar:
- **Lineal simple** (y ~ X1)
- **Lineal múltiple** (y ~ X1 + X2 + X3)
- **Polinomial** sobre X1 (grado 3: X1, X1^2, X1^3)

Incluye dataset sintético reproducible, `train/test split`, validación cruzada y gráficos.

## Requisitos
- Python 3.10+
- Librerías: `numpy`, `pandas`, `matplotlib`, `scikit-learn`  
  (están en `requirements.txt` de la raíz).

## Archivo principal
`linear_multiple_poly.py` (CLI con `--demo` y parámetros opcionales).

## Cómo correr

Desde `regression_models/`:

```bash
# Ejecución completa con figuras y métricas
python linear_multiple_poly.py --demo

# Opcionalmente ajusta tamaño/ruido/semilla
python linear_multiple_poly.py --demo --n 800 --noise 1.5 --seed 7
```

Se crean en `regression_models/figs/`:
- `simple_fit.png` — dispersión (train) + línea de predicción para lineal simple.
- `multiple_pred_vs_true.png` — y real vs y predicho (línea ideal y=x).
- `poly_fit.png` — curva polinomial (grado 3) sobre X1.
- `metrics_YYYYMMDD_HHMMSS.csv` — tabla con métricas.

## Métricas reportadas
Para cada modelo:
- `r2_train`, `r2_test`
- `mae_test` (error absoluto medio)
- `rmse_test` (raíz del error cuadrático medio)
- `cv_r2_mean`, `cv_r2_std` (validación cruzada k=5)

Ejemplo rápido para ver el CSV:
```bash
# En bash/mac/linux
column -s, -t regression_models/figs/metrics_*.csv | less -S
```

## Notas de implementación
- Dataset: `make_dataset(n, noise, seed)` genera X1, X2, X3 ~ N(0,1) y un `y`
  con mezcla lineal + términos no lineales en X1, de modo que:
  - Lineal simple subajusta.
  - Lineal múltiple mejora.
  - Polinomial captura la no linealidad en X1.
- Los gráficos son intencionalmente simples (sin temas/estilos).

## Problemas comunes
- Si tu editor introduce comillas “curvas” o símbolos raros, usa el archivo ASCII-safe
  que te pasé o verifica que el archivo está guardado en **UTF-8** con saltos **LF**.
- Selecciona **Python 3.10+** en VS Code (Ctrl+Shift+P → Python: Select Interpreter).

## Uso desde Python (opcional)

```python
import pandas as pd
from linear_multiple_poly import make_dataset, fit_linear_multiple

df = make_dataset(n=500, noise=1.2, seed=123)
res, y_true, y_pred = fit_linear_multiple(df)
print(res)
```
