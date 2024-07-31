import numpy as np
import pandas as pd
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

        return one_hot
    
    def categorical_cross_entropy(self, y_true, y_pred):

        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping predicted values
        loss = - np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)

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

    def backward(self, X, y, weighted_sums, activations, learning_rate=0.1):
        m = X.shape[0]

        # Gradient of loss with respect to Z3 (output layer logits)
        dZ3 = activations[-1] - y
        dW3 = np.dot(activations[-2].T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Gradient of loss with respect to A2 (second hidden layer activations)
        dA2 = np.dot(dZ3, self.weights[-1].T)
        dZ2 = dA2 * (weighted_sums[-2] > 0)  # Derivative of ReLU
        dW2 = np.dot(activations[-3].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Gradient of loss with respect to A1 (first hidden layer activations)
        dA1 = np.dot(dZ2, self.weights[-2].T)
        dZ1 = dA1 * (weighted_sums[-3] > 0)  # Derivative of ReLU
        dW1 = np.dot(X.reshape(784,1), dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        dW = [dW1, dW2, dW3]
        db = [db1, db2, db3]
        return dW, db
    
    def update_params(self, dW, db):
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] -= 0.1 * dW[i]
            self.biases[i] -= 0.1 * db[i]

    def predict(self, inputs):
        _, activations = self.forward(inputs)
        return np.argmax(activations[-1])
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                weighted_sums, activations = self.forward(x)
                dW, db = self.backward(x, y, weighted_sums, activations)

                self.update_params(dW, db)

            # loss = np.mean(np.square(y - activations[-1])) * 100
            loss = self.categorical_cross_entropy(y, activations[-1]) * 100
            print(f"Epoch {epoch}, Loss:{loss}")


if __name__ == "__main__":
    nn = NeuralNetwork()

    data = pd.read_csv("train.csv")

    data = np.array(data)
    m, n = data.shape

    np.random.shuffle(data)

    data = data.T

    y_train = data[0]
    x_train = data[1:n].T

    y_train = nn.one_hot(y_train, 10)

    nn.train(x_train[:2000], y_train[:2000], 10, 0.1)

    nn.save("modelo.pkl")