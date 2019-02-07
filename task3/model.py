from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from task3.dataset import createDataset

def createModel(units=50):
    model = Sequential()
    model.add(Dense(units, activation="relu", input_shape=(10,)))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
    return model

def classifyData():
    model = createModel()
    x, y = createDataset()
    length = len(y)
    x_train = x[0 : 15000, :]
    y_train = y[0 : 15000]
    x_test =  x[15000 : length, :]
    y_test = y[15000 : length]
    model.fit(x_train, y_train, epochs=5, batch_size=100)
    print(model.evaluate(x_test, y_test))