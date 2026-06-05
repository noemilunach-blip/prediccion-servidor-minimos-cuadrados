import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate_dataset import generate_server_dataset
from least_squares import MinimoCuadrados

df = generate_server_dataset()
x  = df["mes"].values
ultimo_mes = int(df["mes"].max())

metricas = {
    "CPU (%)":            df["cpu_uso"].values,
    "RAM (%)":            df["ram_uso"].values,
    "Almacenamiento (%)": df["almacenamiento_uso"].values,
}

modelos = {}
for nombre, y in metricas.items():
    modelos[nombre] = MinimoCuadrados().ajustar(x, y)
    m = modelos[nombre]
    print(f"\n{nombre}")
    print(f"  Ecuacion : y = {m.pendiente:.4f}*x + {m.intercepto:.4f}")
    print(f"  R2       : {m.r_cuadrado:.4f}")

x_futuro = np.arange(ultimo_mes + 1, ultimo_mes + 13)
filas = []
for mes in x_futuro:
    fila = {"Mes": mes}
    for nombre, modelo in modelos.items():
        col = nombre.split(" ")[0]
        fila[f"{col} (%)"] = round(float(modelo.predecir(mes)), 2)
    filas.append(fila)

df_pred = pd.DataFrame(filas)
print("\nPredicciones proximos 12 meses:")
print(df_pred.to_string(index=False))

colores = {"CPU (%)": "#3b82f6", "RAM (%)": "#8b5cf6", "Almacenamiento (%)": "#10b981"}
cols_df = {"CPU (%)": "cpu_uso", "RAM (%)": "ram_uso", "Almacenamiento (%)": "almacenamiento_uso"}

fig = plt.figure(figsize=(16, 5), facecolor="#0f172a")
axes = [fig.add_subplot(1, 3, i+1) for i in range(3)]

for ax, (nombre, modelo) in zip(axes, modelos.items()):
    color  = colores[nombre]
    y_hist = df[cols_df[nombre]].values
    x_hist = df["mes"].values
    ax.set_facecolor("#1e293b")
    ax.axvspan(ultimo_mes + 0.5, ultimo_mes + 13, alpha=0.07, color=color)
    ax.axhline(85, color="#ef4444", lw=1.2, ls=":", alpha=0.8, label="Limite 85%")
    ax.scatter(x_hist, y_hist, color=color, s=35, zorder=5, edgecolors="white", linewidths=0.4, label="Real")
    ax.plot(np.linspace(1, ultimo_mes, 200), modelo.predecir(np.linspace(1, ultimo_mes, 200)), color=color, lw=2, ls="--", label="Ajuste MC")
    ax.plot(np.linspace(ultimo_mes, ultimo_mes+12, 100), modelo.predecir(np.linspace(ultimo_mes, ultimo_mes+12, 100)), color=color, lw=2.5, label="Prediccion")
    ax.set_title(nombre, color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mes", color="#94a3b8")
    ax.set_ylabel("Uso (%)", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, ultimo_mes + 13)
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#0f172a", edgecolor="#334155", framealpha=0.4)
    eq = f"y={modelo.pendiente:.3f}x+{modelo.intercepto:.2f}  R2={modelo.r_cuadrado:.3f}"
    ax.text(0.03, 0.05, eq, transform=ax.transAxes, fontsize=7.5, color="#e2e8f0",
            bbox=dict(boxstyle="round", facecolor="#0f172a", alpha=0.6))

plt.suptitle("Prediccion de Carga del Servidor - Minimos Cuadrados", color="white", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("prediccion_servidor.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.show()
