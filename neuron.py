import numpy as np

class Neuron:
    def __init__(self, learning_rate=0.2) -> None:
        self.bias = np.random.randn()
        self.weights = np.random.rand(2)
        self.learning_rate = learning_rate

    def relu(self, weighted_sum):
        return np.maximum(weighted_sum, 0)

    def forward(self, input):
        weighted_sum = np.dot(input, self.weights) + self.bias

        return self.relu(weighted_sum)
    
    def train(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                prediction = self.forward(x)

                error = y - prediction

                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error
            print(f"Epoch {epoch+1}/{epochs} - Weights: {self.weights}, Bias: {self.bias}")

neuron = Neuron()

training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([0, 0, 0, 1])

neuron.train(training_inputs, targets)

print(neuron.weights)
print(neuron.bias)

for inputs in training_inputs:
    output = neuron.forward(inputs)
    print(f"Inputs: {inputs}, Output: {output}")