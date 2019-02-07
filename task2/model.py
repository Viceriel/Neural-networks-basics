from task2.dataset import dataPreprocessing
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def createModel(units=50):
    model = Sequential()
    model.add(Dense(units, activation="sigmoid", input_shape=(5,)))
    model.add(Dense(1))
    model.compile(optimizer='sgd', loss='mse', metrics=['acc'])
    return model

def fundPrediction():
    data_processor = dataPreprocessing("task2/fund.xls")
    data_processor.extractDataFromSheet()
    y = data_processor.getScaledData()
    plt.plot(y, color="g")
    plt.xlabel("Time")
    plt.ylabel("Fund value [EUR]")
    plt.title("Fund value over time")
    plt.grid()
    plt.show()
    x, y = data_processor.createDataset()
    train = 0.8
    test = 1 - train
    length = x.shape[0]
    x_train = x[0: int(train * length)]
    y_train = y[0: int(train * length)]
    x_test = x[int(train * length): length]
    y_test = y[int(train * length): length]
    model = createModel()
    model.fit(x_train, y_train, shuffle=False, epochs=150, batch_size=100)
    predictions = model.predict(x_test)
    plt.plot(y_test, "g", predictions, "r")
    plt.grid()
    plt.xlabel("Time [Days]")
    plt.ylabel("Fund value")
    plt.legend(["Price", "Predicted"])
    plt.show()
