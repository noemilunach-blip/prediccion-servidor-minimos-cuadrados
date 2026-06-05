# Predicción de Carga de Servidor — Mínimos Cuadrados

Modelo de regresión lineal para predecir el uso futuro de CPU, RAM y Almacenamiento en un servidor, usando el método de Mínimos Cuadrados.

## Archivos

- `generate_dataset.py` — genera el dataset simulado de 24 meses
- `least_squares.py` — implementación del método de mínimos cuadrados
- `main.py` — ejecuta el análisis completo y genera la gráfica
- `requirements.txt` — dependencias del proyecto

## Cómo ejecutar en Google Colab

1. Abrir [Google Colab](https://colab.research.google.com)
2. Crea un nuevo cuaderno
3. Ejecutar primero esta celda para instalar dependencias:

\`\`\`
!pip install numpy pandas matplotlib -q
\`\`\`

4. Copiar el contenido de cada archivo en celdas en este orden y ejecuta :
   - `generate_dataset.py`
   - `least_squares.py`
   - `main.py`

## Cómo ejecutar

1. Instala las dependencias:

\`\`\`
pip install -r requirements.txt
\`\`\`

2. Ejecuta el programa:

\`\`\`
python main.py
\`\`\`

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
