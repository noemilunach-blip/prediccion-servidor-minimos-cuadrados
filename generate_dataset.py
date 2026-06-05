import numpy as np
import pandas as pd

def generate_server_dataset(n=13, seed=42):
    np.random.seed(seed)
    meses   = np.arange(1, n + 1)
    cpu     = np.clip(20 + 2.3 * meses + np.random.normal(0, 3.5, n), 10, 98)
    ram     = np.clip(30 + 2.1 * meses + np.random.normal(0, 4.0, n), 15, 99)
    storage = np.clip(10 + 2.5 * meses + np.random.normal(0, 2.0, n),  5, 99)
    return pd.DataFrame({
        "mes": meses,
        "cpu_uso": np.round(cpu, 2),
        "ram_uso": np.round(ram, 2),
        "almacenamiento_uso": np.round(storage, 2)
    })

if __name__ == "__main__":
    df = generate_server_dataset()
    df.to_csv("server_metrics.csv", index=False)
    print(df.to_string(index=False))
