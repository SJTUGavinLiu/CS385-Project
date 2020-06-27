import numpy as np 


class KN:
    @staticmethod
    def poly(x, y, p=4):
        return (x.dot(y.T) + 1) ** p
    @staticmethod
    def rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))