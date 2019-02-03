from datetime import timedelta, datetime
import xlrd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class dataPreprocessing:

    def __init__(self, path):
        self.path = path
        self.scaler = {}

    def loadSheet(self, path):
        book = xlrd.open_workbook(path)
        sheet = book.sheet_by_index(0)

        arr = []
        counter = 1
        while True:
            try:
                val = xlrd.xldate_as_tuple(sheet.cell_value(counter, 0), book.datemode)
                date = datetime(val[0], val[1], val[2])
                arr.append([date, sheet.cell_value(counter, 2)])
                counter = counter + 1
            except:
                break

        return list(reversed(arr))

    def fillGap(self, arr):
        length = len(arr)
        x = []
        y = []
        all = []
        for i in range(length):
            if i != 0:
                delta = arr[i][0] - arr[i - 1][0]
                diff = delta.days
                if diff > 1:
                    date = arr[i - 1][0]
                    diff = diff - 1
                    for j in range(diff):
                        date += timedelta(days=1)
                        x.append(date)
                        y.append(arr[i - 1][1])
                        all.append([date, arr[i - 1][1]])
            x.append(arr[i][0])
            y.append(arr[i][1])
            all.append([arr[i][0], arr[i][1]])
        return [x, y, all]

    def extractDataFromSheet(self):
        arr = self.loadSheet(self.path)
        x, y, all = self.fillGap(arr)
        self.y = y
        return self.y

    def getScaledData(self):
        scaled_y = np.asarray(self.y)
        scaled_y = scaled_y.reshape(-1, 1)
        self.scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        self.scaler.fit(scaled_y)
        return self.scaler.transform(scaled_y)

    def createDataset(self, window=5):
        self.extractDataFromSheet()
        scaled = self.getScaledData()
        actual = 0
        target = window
        length = scaled.shape[0]
        x = []
        y = []

        while target < length:
            x.append(scaled[actual : actual + window].tolist())
            y.append(scaled[target])
            target = target + 1
            actual = actual + 1
        x = np.asarray(x)
        x = x.reshape(x.shape[0], window)
        y = np.asarray(y)
        return x, y



