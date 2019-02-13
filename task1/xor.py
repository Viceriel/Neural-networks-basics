from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def model_creation(neurons):
    model = Sequential()
    model.add(Dense(neurons, activation="sigmoid", input_shape=(2,)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='sgd', loss='mse', metrics=['acc'])
    return model

def xor_problem_basic(neurons=2, epochs=30000, batch=1):
    model = model_creation(neurons)
    model.fit(x, y, epochs=epochs, batch_size=batch)
    print(model.predict(x))

def xor_problem_evaulation(neurons=2, epochs=30000, batch=1):
    model = model_creation(neurons)
    model.compile(optimizer='sgd', loss='mse', metrics=['acc'])
    acc = []
    for i in range(epochs):
        model.fit(x, y, epochs=1, batch_size=batch, verbose=0)
        evaluation = model.evaluate(x, y, verbose=0)
        acc.append(evaluation[1])
    acc = np.asarray(acc)
    plt.plot(acc)
    plt.title("Accuracy during training")
    plt.xlabel("Epochs [-]")
    plt.ylabel("Accuracy [-]")
    plt.grid()
    plt.show()
    print(model.predict(x))

def xor_problem_board(neurons=2, epochs=30000, batch=1):
    model= model_creation(neurons)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True)
    model.fit(x, y, epochs=epochs, batch_size=batch, verbose=1, callbacks=[tensorboard])
    print(model.predict(x))
    print(model.summary())