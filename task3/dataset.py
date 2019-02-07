import numpy as np
from sklearn.preprocessing import MinMaxScaler

def randomize(arr1):
    indexes = np.arange(len(arr1))
    for counter in range(20000):
        rand_ind = np.random.randint(0, len(arr1))
        rand_ind2 = np.random.randint(0, len(arr1))
        temp = arr1[rand_ind]
        arr1[rand_ind] = arr1[rand_ind2]
        arr1[rand_ind2] = temp

def createDataset():
    file = open("task3\data.txt", "r")
    lines = file.readlines()
    randomize(lines)
    counter = 0
    data = np.ndarray((len(lines), 10))
    length = len(lines)
    y = np.ndarray(length)
    for line in lines:
        arr = line.split(",")
        for i in range(10):
            data[counter, i] = float(arr[i])
        y[counter] = float(arr[10].find("g") != -1)
        counter = counter + 1

    x = np.ndarray((length, 10))
    for i in range(10):
        column = data[:, i]
        column = np.reshape(column, (-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(column)
        column = scaler.transform(column)
        column = np.reshape(column, (length))
        x[:, i] = column

    return x, y