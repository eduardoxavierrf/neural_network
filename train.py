import pandas as pd
import numpy as np
from neural_network import NeuralNetwork

df = pd.read_csv("train.csv")

data = df.to_numpy()

np.random.shuffle(data)

y_train = data[:6400, [0]]
x_train = data[:6400, 1:] / 255
y_test = data[6400:9600, [0]]
x_test = data[6400:9600, 1:] / 255

nn = NeuralNetwork()

y_train = np.array([nn.one_hot(y, 10) for y in y_train])
y_test = np.array([nn.one_hot(y, 10) for y in y_test])

nn.train(x_train, y_train, 40, 32)

print(f"Train: Loss {nn.avarage_loss(x_train, y_train)}, Accuracy {nn.accuracy(x_train, y_train)}")
print(f"Test: Loss {nn.avarage_loss(x_test, y_test)}, Accuracy {nn.accuracy(x_test, y_test)}")

nn.save("model.pkl")