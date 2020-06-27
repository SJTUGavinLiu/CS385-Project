import numpy as np 
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from utils.PCA import PCA




class GMM():
    def __init__(self, K, epoch):
        self._K = K
        self._epoch = epoch
        self._X = None
        self._mu = None
        self._cov = None
    def phi(self, k):
        norm = multivariate_normal(mean=self._mu[k], cov=self._cov[k])
        return norm.pdf(self._X)
    def _E(self):
        prob = np.zeros((self._X.shape[0], self._K))
        for k in range(self._K):
            prob[:, k] = self.phi(k)
        prob = np.mat(prob)
        for k in range(self._K):
            self._gamma[:, k] = self._alpha[k] * prob[:, k]
        for i in range(self._X.shape[0]):
            self._gamma[i, :] /= np.sum(self._gamma[i, :])
    def _M(self):
        for k in range(self._K):
            Nk = np.sum(self._gamma[:, k])
            self._mu[k, :] = np.sum(np.multiply(self._X, self._gamma[:, k]), axis=0) / Nk
            # 更新 cov
            self._cov[k, :] = (self._X - self._mu[k]).T * np.multiply((self._X - self._mu[k]), self._gamma[:, k]) / Nk
            # 更新 alpha
            self._alpha[k] = Nk / self._X.shape[0]

    def randomcolor(self):
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,14)]
        return "#"+color
    def PCA(self,X, K):
        cov_matrix = np.cov(X.T)    
        eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
        sortIdx = eigenValues.argsort()[::-1]
        eigenValues_Topk = eigenValues[sortIdx][:K]
        eigenVectors_Topk = eigenVectors[:, sortIdx][:, :K]

        X_ = X.dot(eigenVectors_Topk)
        return X_
    def fit(self, X, y):
        self._X = X
        self._y = y
        self._mu = np.random.rand(self._K, X.shape[1])
        self._cov = np.array([np.eye(X.shape[1])] * self._K)
        self._alpha = np.array([1.0 / self._K] * self._K)
        self._gamma = np.mat(np.zeros((X.shape[0], self._K)))
        for _ in range(self._epoch):
            self._E()
            self._M()
        self._category = self._gamma.argmax(axis=1).flatten().tolist()[0]
        #print(self._category)
        #print("alpha: ", self._alpha)

    def plot2d(self):
        class_EM = []
        class_GT = []
        X_2D = PCA(self._X, 2)
        for k in range(self._K):
            class_EM.append(np.array([X_2D[i] for i in range(self._X.shape[0]) if self._category[i] == k]))
            class_GT.append(np.array([X_2D[i] for i in range(self._X.shape[0]) if self._y[i] == k]))
        #print(class_EM[1])
        for i in range(self._K):
            plt.plot(class_EM[i][:, 0], class_EM[i][:, 1], color = self.randomcolor(), linestyle = 'None', marker = "o", label = str(i))
        plt.legend(loc="best")
        plt.title("GMM Clustering By EM Algorithm")
        plt.show()
        #plt.savefig('test1.png')

    def plot3d(self):
        class_EM = []
        class_GT = []
        data = [(self._X[i][0].real, self._X[i][1].real, self._X[i][2].real, self._category[i] * 10) for i in range(len(self._X))]
        return data



