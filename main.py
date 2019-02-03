import tensorflow as tf
from task1.xor import xor_problem_basic, xor_problem_evaulation, xor_problem_board
from task2.dataset import dataPreprocessing
from matplotlib import pyplot as plt

data_processor = dataPreprocessing("task2/fund.xls")
data_processor.extractDataFromSheet()
y = data_processor.getScaledData()
plt.plot(y, color="g")
plt.xlabel("Time")
plt.ylabel("Fund value [EUR]")
plt.title("Fund value over time")
plt.grid()
plt.show()
data_processor.createDataset()