import numpy as np
import random

class LinearRegression():
    def __init__(self, epoch = 10000, rate = 0.001, batch_size = 128):
        self._theta = None
        self._b = None
        self._X = None
        self._y = None
        self._epoch = epoch
        self._rate = rate
        self._batch_size = batch_size

    def fit(X, y):
        self._theta = np.zeros(X.shape[1])
        self._b = 0
        for i in range(self._epoch):
            j = random.randint(0, len(X) - 1)
            self._theta += self._rate * (y[j] - X[j].dot(self._theta) - self._b) * X[j]
            self._b += self._rate * (y[j] - X[j].dot(self._theta) - self._b)
        

    def predict(X):
        return X.dot(self._theta) + self._b
    



'''
    Logistic Regression 
    Based on Mini-batch
'''

class LogisticRegression():
    def __init__(self, category_size, epoch = 10000, rate = 0.001, batch_size = 32):
        self._theta = None
        self._b = None
        self._epoch = epoch
        self._rate = rate
        self._batch_size = batch_size
        self._category_size = category_size

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))


    def fit(self, X, y, mode = None):
        self._theta = np.ones((self._category_size, X.shape[1]))
        self._b = np.ones(self._category_size)
        list = [i for i in range(len(X))]
        #print(list)
        lbd = 0.01

        if mode != None and mode != 'Ridge' and mode != 'Lasso':
            raise NotImplementedError("Mode {} is not implemented!".format(mode))



        for i in range(self._epoch):
            #j = random.randint(0, len(X) - 1)
            #self._theta += self._rate * (y[j] - self.sigmoid(X[j].dot(self._theta) - self._b)) * X[j]
            #self._b += self._rate * (y[j] - self.sigmoid(X[j].dot(self._theta) - self._b))
            random.shuffle(list)
            batch = list[:self._batch_size]
            

            for i in range(self._category_size):
                theta = self._theta[i]
                b = self._b[i]
                for j in batch:
                    if mode == 'Lasso':
                        #print(np.array([lbd if theta[k] > 0 else -lbd for k in range(X.shape[1])]))
                        self._theta[i] += self._rate * (((1 if i == y[j] else 0) - self.sigmoid(X[j].dot(theta) - b)) * X[j] - \
                                                        lbd * np.sign(theta))
                        self._b[i] += self._rate * ((1 if i == y[j] else 0) - self.sigmoid(X[j].dot(theta) - b) - \
                                                        lbd * (1 if b > 0 else -1))
                    elif mode == 'Ridge':
                        self._theta[i] += self._rate * (((1 if i == y[j] else 0) - self.sigmoid(X[j].dot(theta) - b)) * X[j] - \
                                                         2 * lbd * theta )
                        self._b[i] += self._rate * ((1 if i == y[j] else 0) - self.sigmoid(X[j].dot(theta) - b) - \
                                                         2 * lbd * b)
                    else:
                        self._theta[i] += self._rate * ((1 if i == y[j] else 0) - self.sigmoid(X[j].dot(theta) - b)) * X[j]
                        self._b[i] += self._rate * ((1 if i == y[j] else 0) - self.sigmoid(X[j].dot(theta) - b))

    def predict(self, X):
        pred_raw = X.dot(self._theta.transpose()) + self._b
        pred = np.zeros(len(X))
        
        for i in range(len(X)):
            pred[i] = np.argmax(pred_raw[i])
        
        return pred




if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = np.array(x_train)
    x_train =x_train.reshape(60000, 784).astype('float32')
    x_test = x_test.reshape(10000, 784).astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255
    print(x_train)

    clf = LogisticRegression(10)
    clf.fit(x_train[:1000], y_train[:1000], mode = 'Lasso')
    pred = clf.predict(x_test)
    print(pred)
    print("准确率：{:8.6} %".format((pred  == y_test).mean() * 100))

    #混淆矩阵
    #print(confusion_matrix(y_test, pred))
    #f1-score,precision,recall
    print(classification_report(y_test, np.array(pred)))
    #计算准确度
    print('accuracy=', accuracy_score(y_test, pred))