# Predicción de Carga de Servidor — Mínimos Cuadrados

Modelo de regresión lineal para predecir el uso futuro de CPU, RAM y Almacenamiento en un servidor, usando el método de Mínimos Cuadrados.

## Archivos
-`Minimos_Cuadrados_Servidor.ipynb` — notebook completo con código, entrenamiento y resultados
- `generate_dataset.py` — genera el dataset simulado de 24 meses
- `least_squares.py` — implementación del método de mínimos cuadrados
- `main.py` — ejecuta el análisis completo y genera la gráfica
- `requirements.txt` — dependencias del proyecto

## Cómo ejecutar en Google Colab

1. Abre [Google Colab](https://colab.research.google.com)
2. Sube el notebook `Minimos_Cuadrados_Servidor.ipynb`
3. Ejecuta todas las celdas en orden con ▶️

## Cómo ejecutar en tu computadora

```bash
pip install -r requirements.txt
python main.py
```

## Dependencias

- numpy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7

## Resultado

El modelo entrena con 24 meses de datos históricos simulados y predice el uso de CPU, RAM y Almacenamiento para los próximos 12 meses. También detecta en qué mes cada recurso supera el umbral crítico del 85%, lo que permite planificar ampliaciones de infraestructura con anticipación.

## Método matemático

El método de Mínimos Cuadrados busca la recta `y = mx + b` que minimiza la suma de errores al cuadrado entre los valores reales y los estimados.

Fórmulas utilizadas:

- `m = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`
- `b = (Σy − m·Σx) / n`
- `R² = 1 − (SS_res / SS_tot)`
