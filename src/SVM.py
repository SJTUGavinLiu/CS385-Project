
import numpy as np 
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import random

from utils.kernels import KN
SIZE = 500

'''
    Based on Gradient Descent
    Single Classifier
'''
class SVM_GD():
    def __init__(self, kernel = "rbf", p = 4, gamma = 0.001, C = 1, rate = 0.0001, batch_size = 128, epoch = 10000):
        
        self._rate = rate
        self._C = C
        self._batch_size = batch_size
        self._epoch = epoch
        if kernel == 'linear':
            self._kernel = lambda a, b: a.dot(b.T)
        elif kernel == 'poly':
            self._kernel = lambda a, b: KN.poly(a,b,p)
        elif kernel == 'rbf':
            self._kernel = lambda a, b: KN.rbf(a, b, gamma)
        else:
            raise NotImplementedError("Kernel {} is not supported!".format(kernel))
    def fit(self, X, y):
        self._X = np.array(X)
        self._y = np.array(y)
        self._b = 0
        self._alpha = np.zeros(X.shape[0])
        K = self._kernel(X, X)
        K_diag = np.diag(K)

        for _ in range(self._epoch):
            self._alpha -= -self._rate * (np.sum(self._alpha * K, axis = 1) + self._alpha * K_diag) * 0.5
            batch = np.random.permutation(X.shape[0])[:self._batch_size]
            K_batch = K[batch]
            y_batch = y[batch]
            error = 1 - y_batch * (K_batch.dot(self._alpha) + self._b)

            mask = error > 0

            delta = self._C * self._rate * y_batch[mask]
            self._alpha += np.sum(delta[..., None] * K_batch[mask], axis=0)
            self._b += np.sum(delta)        
        
    def predict(self, X):
        #x = np.atleast_2d(x).astype(np.float32)
        K = self._kernel(self._X, X)
        y_pred = self._alpha.dot(K) + self._b
        #print(self._alpha)
        #print(self._b)
        return np.sign(y_pred)

          
'''
    Based on SMO algorithm
    Single Classifier
'''
class SVM_SMO:
    def __init__(self,C,tol,epoch,kernel = 'rbf', epsilon = 0.0001, **kernelargs):
        self._X = None
        self._y = None
        self._m = None
        self._alpha = None
        self._K = None
        self._err = None
        self._C = C
        self._tol = tol
        self._epoch = epoch

        self._b = 0.0
        self._epsilon = epsilon

        if kernel == 'linear':
            self._kernel = lambda a, b: a.dot(b.T)
        elif kernel == 'poly':
            self._kernel = lambda a, b: KN.poly(a,b,d)
        elif kernel == 'rbf':
            self._kernel = lambda a, b: KN.rbf(a, b, 0.0025)

    def _cal_err(self,k):
        return np.dot(self._alpha*self._y,self._K[:,k])+self._b - float(self._y[k])
    def _update_err(self,k):
        self._err[k] = [1 ,self._cal_err(k)]
    def _cal_idx(self,i,Ei):
        maxE = 0.0
        j = 0
        Ej = 0.0
        validECacheList = np.nonzero(self._err[:,0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i:continue
                Ek = self._cal_err(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxE:
                    j = k
                    maxE = deltaE
                    Ej = Ek
            return j,Ej
        else:
            j = i
            while j == i:
                j = int(random.uniform(0,self._m))
            Ej = self._cal_err(j)
            return j,Ej

    def _take_step(self,i):
        #print(self._alpha)
        Ei = self._cal_err(i)
        if (self._y[i] * Ei < -self._tol and self._alpha[i] < self._C) or \
                (self._y[i] * Ei > self._tol and self._alpha[i] > 0):
            self._update_err(i)
            j,Ej = self._cal_idx(i,Ei)
            ai = self._alpha[i]
            aj = self._alpha[j]
            if self._y[i] != self._y[j]:
                L = max(0,self._alpha[j]-self._alpha[i])
                H = min(self._C,self._C + self._alpha[j]-self._alpha[i])
            else:
                L = max(0,self._alpha[j]+self._alpha[i] - self._C)
                H = min(self._C,self._alpha[i]+self._alpha[j])
            #print(L, H)
            if L == H:
                return 0
            eta = 2*self._K[i,j] - self._K[i,i] - self._K[j,j]
            if eta >= 0:
                return 0
            self._alpha[j] -= self._y[j]*(Ei-Ej)/eta
            if self._alpha[j] < L:
                self._alpha[j] = L
            elif self._alpha[j] > H:
                self._alpha[j] = H
            self._update_err(j)
            if abs(aj-self._alpha[j]) < self._epsilon * (aj + self._alpha[j] + self._epsilon):
                return 0
            self._alpha[i] +=  self._y[i]*self._y[j]*(aj-self._alpha[j])
            self._update_err(i)
            b1 = self._b - Ei - self._y[i] * self._K[i, i] * (self._alpha[i] - ai) - \
                 self._y[j] * self._K[i, j] * (self._alpha[j] - aj)
            b2 = self._b - Ej - self._y[i] * self._K[i, j] * (self._alpha[i] - ai) - \
                 self._y[j] * self._K[j, j] * (self._alpha[j] - aj)
            if 0 <self._alpha[i] and self._alpha[i] < self._C:
                self._b = b1
            elif 0 < self._alpha[j] and self._alpha[j] < self._C:
                self._b = b2
            else:
                self._b = (b1 + b2) /2.0
            return 1
        else:
            return 0

    def fit(self, X, y):
        self._X = np.array(X)
        self._y = np.array(y)
        self._m = len(X)
        self._alpha = np.zeros(self._m)
        self._err = np.array(np.zeros((self._m,2)))
        self._K = self._kernel(X,X)
        numChanged = 0
        examineAll = True
        for _ in range(self._epoch):
            #print(numChanged, examineAll)
            if numChanged == 0 and (not examineAll):
                return 
            numChanged = 0
            if examineAll:
                for i in range(len(self._X)):
                    numChanged += self._take_step(i)
            else:
                nonZeroC = np.nonzero((self._alpha > 0)*(self._alpha < self._C))[0]
                for i in nonZeroC:
                    numChanged += self._take_step(i)
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

    def predict(self,X,raw=False):
        X = np.array(X)
        result = []
        m = np.shape(X)[0]
        for i in range(m):
            tmp = self._b
            #for j in range(len(self.SVIndex)):
                #tmp += self.SVAlpha[j] * self.SVLabel[j] * self._kernel(np.np.array([self.SV[j]]), np.np.array([test[i,:]]))[0]
            for j in range(len(self._X)):
                tmp += self._alpha[j] * self._y[j] * self._kernel(np.array([self._X[j]]), np.array([X[i]]))[0]
            while tmp == 0:
                tmp = random.uniform(-1,1)
            
            if raw:
                result.append(tmp)
            else:
                if tmp > 0:
                    tmp = 1
                else:
                    tmp = -1
            result.append(tmp)
        return np.array(result)




'''
    SVM
    Multiple Classifier
'''
class SVM():
    def __init__(self, SIZE = 500):
        self._SIZE = SIZE


    def predict(self, x_train, y_train, x_test, y_test):    
        x = [[] for i in range(10)]
        cnt = [0 for i in range(10)]
        for i in range(len(x_train)):
            if cnt[y_train[i]] >= self._SIZE:
                continue
            else:
                cnt[y_train[i]] += 1
                x[y_train[i]].append(x_train[i])

        #svm = [SVM_SMO(200, 0.0001, 10000, name='rbf', theta=2) for i in range(55)]
        counts = [[0 for i in range(10)] for j in range(len(y_test))]
        svm = SVM_SMO(200, 0.0001, 10000, name='rbf', theta=2)

        beg = time.time()

        id = 0
        for i in range(10):
            for j in range(i+1,10):
                x_ = x[i] + x[j]
                y_ = [1 for k in range(self._SIZE)] + [-1 for k in range(self._SIZE)]
                svm.fit(np.array(x_), np.array(y_))
                pred = svm.predict(x_test)
                for a in range(len(y_test)):
                    if pred[a] == 1:
                        counts[a][i] += 1
                    else:
                        counts[a][j] += 1
                id+=1
        pred = []
        for item in counts:
            pred.append(np.argmax(item))
        pred = np.array(pred)
        print(confusion_matrix(y_test, list(pred)))
        #f1-score,precision,recall
        print(classification_report(y_test, pred))
        #Accuracy
        print('accuracy=', accuracy_score(y_test, pred))
        #Time
        print('time=', time.time() - beg)
        return pred

if __name__ == '__main__':
    print(os.getcwd())
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from keras.datasets import mnist

    # Import Mnist dataset from keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train =x_train.reshape(60000, 784).astype('float32')
    x_test = x_test.reshape(10000, 784).astype('float32')

    # Normalization
    x_train = x_train / 255
    x_test = x_test / 255
    SIZE = 100
    model = SVM(SIZE)
    model.predict(x_train, y_train, x_test, y_test)