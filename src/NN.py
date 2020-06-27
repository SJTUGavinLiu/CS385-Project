import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from keras.datasets import mnist


def transferY(y):
    Y = []
    for i in range(len(y)):
        tmp = []
        for j in range(10):
            if y[i] == j:
                tmp.append(1)
            else:
                tmp.append(0)
        Y.append(tmp)
    return np.array(Y)



class SimpleNN():
    def __init__(self, loss_function = 'categorical_crossentropy'):
        self._model = Sequential()
        self._model.add(Dense(512, input_shape=(784,), activation='relu'))
        self._model.add(Dropout(0.4))
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(0.4))
        self._model.add(Dense(128, activation='relu'))
        self._model.add(Dropout(0.4))
        self._model.add(Dense(10, activation='softmax'))
        self._model.summary()
        self._model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])



    def fit(self, X, y):
        self._model.fit(X, y, batch_size=500, epochs=20, verbose=1)
        
    def predict(self, X_test, y_test):
        loss, accuracy = self._model.evaluate(X_test, y_test, verbose=0)
        result = self._model.predict(X_test)
        #result = np.argmax(result, axis = 1)
        #print(result)
        print('Loss:', loss)
        print('Accuracy:', accuracy)
        return result


class CovNN():
    def __init__(self, loss_function = 'categorical_crossentropy'):
        self._model = Sequential()
        self._model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Conv2D(15, (3, 3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.4))
        self._model.add(Flatten())
        self._model.add(Dense(128, activation='relu'))
        self._model.add(Dense(64, activation='relu'))
        self._model.add(Dense(10, activation='softmax'))
        self._model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
    def fit(self, X, y):
        self._model.fit(X, Y, batch_size=500, epochs=20, verbose=1)
        
    def predict(self, X_test, y_test):
        loss, accuracy = self._model.evaluate(X_test, y_test, verbose=0)
        result = self._model.predict(X_test)
        #result = np.argmax(result, axis = 1)
        #print(result)
        print('Loss:', loss)
        print('Accuracy:', accuracy)
        return result




if __name__ == '__main__':
    # Import Mnist dataset from keras
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Transfer Mnist dataset
    Y_train = transferY(Y_train)
    Y_test = transferY(Y_test)
    #X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    #X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_train =X_train.reshape(60000, 784).astype('float32')
    X_test = X_test.reshape(10000, 784).astype('float32')

    # Normalization
    X_train = X_train / 255
    X_test = X_test / 255
    print(X_train)
    print(X_train.shape)
    
    
    
    model = SimpleNN('kullback_leibler_divergence')
    model.fit(X_train, Y_train)
    model.predict(X_test, Y_test)
    

    #print(X_train.shape, X_train[0])
    #model = CovNN('kullback_leibler_divergence')
    #model.fit(X_train, Y_train)
    #model.predict(X_test, Y_test)