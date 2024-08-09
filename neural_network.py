import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10) -> None:
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))
            
    def relu(self, vector):
        return np.maximum(vector, 0)

    def relu_deriv(self, vector):
        return np.where(vector > 0, 1, 0)
    
    def softmax(self, vector):
        exps = np.exp(vector - np.max(vector))
        return exps / np.sum(exps)
    
    def one_hot(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1

        return one_hot.flatten()

    def forward(self, inputs):
        weighted_sums = []
        activations = [inputs]

        for i in range(len(self.layer_sizes) - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(z)
            activations.append(self.relu(z))

        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        weighted_sums.append(z)
        activations.append(self.softmax(z))

        return weighted_sums, activations

    def backward(self, y, weighted_sums, activations):
        dW = []
        db = []

        delta = activations[-1] - y
        dW.append(np.dot(activations[-2].T, delta))
        db.append(np.sum(delta))
        
        for i in range(len(self.layer_sizes) - 2):
            dA = np.dot(delta, self.weights[-(i+1)].T)
            delta = dA * self.relu_deriv(weighted_sums[-(i+2)])
            dW.insert(0, np.dot(activations[-(i+3)].T, delta))
            db.insert(0, np.sum(delta))

        return dW, db
    
    def update_params(self, dW, db, learning_rate=0.01):
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def predict(self, inputs):
        _, activations = self.forward(inputs)
        return np.argmax(activations[-1])
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                x = x.reshape(1, -1)
                weighted_sums, activations = self.forward(x)
                dW, db = self.backward(y, weighted_sums, activations)

                self.update_params(dW, db)

            print(f"Epoch {epoch}")