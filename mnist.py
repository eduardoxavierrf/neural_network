import pandas as pd
import numpy as np
from neural_network import NeuralNetwork

df = pd.read_csv("train.csv")

data = df.to_numpy()

np.random.shuffle(data)

y_train = data[:250, [0]]
x_train = data[:250, 1:] / 255

nn = NeuralNetwork()

y_train = np.array([nn.one_hot(y, 10) for y in y_train])

nn.train(x_train, y_train, 10)

nn.save("model.pkl")