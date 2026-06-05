import numpy as np

class MinimoCuadrados:
    def __init__(self):
        self.pendiente  = None
        self.intercepto = None
        self.r_cuadrado = None
        self.n          = 0

    def ajustar(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n    = len(x)
        sx, sy   = x.sum(), y.sum()
        sxy, sx2 = (x * y).sum(), (x ** 2).sum()
        denom = n * sx2 - sx ** 2
        self.pendiente  = (n * sxy - sx * sy) / denom
        self.intercepto = (sy - self.pendiente * sx) / n
        yhat   = self.predecir(x)
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        self.r_cuadrado = 1 - ss_res / ss_tot
        self.n = n
        return self

    def predecir(self, x):
        return self.pendiente * np.asarray(x, float) + self.intercepto
