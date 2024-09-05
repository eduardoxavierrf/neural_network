import numpy as np
import pickle
import layers

class NeuralNetwork:
  def __init__(self) -> None:
    self.layers = [
      layers.Linear(28 * 28, 512),
      layers.ReLU(),
      layers.Linear(512, 512),
      layers.ReLU(),
      layers.Linear(512, 10),
      layers.Softmax()
    ]

  def forward(self, activations):
    for layer in self.layers:
      activations = layer.forward(activations)
    return activations
  
  def backward(self, delta):
    for layer in reversed(self.layers):
      delta = layer.backward(delta)

    return delta
  
  def update_params(self, learning_rate):
    for layer in self.layers:
      layer.update_params(learning_rate)

  def cross_entropy_loss(self, y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    
    return np.mean(loss)
  
  def predict(self, inputs):
    output = self.forward(inputs)
    return np.argmax(output)
  
  def accuracy(self, x_data, y_data):
    results = [(self.predict(x),np.argmax(y)) for x, y in zip(x_data, y_data)]

    return (sum(int(x == y) for (x, y) in results)/len(y_data)) * 100

  def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, learning_rate:float = 0.01):
    num_batches = x_train.shape[0] // batch_size
    for epoch in range(epochs):
      total_loss = 0
      for i in range(num_batches):
        start_idx = batch_size * i
        end_idx = start_idx + 32

        output = self.forward(x_train[start_idx:end_idx])

        delta = output - y_train[start_idx:end_idx]
        self.backward(delta)

        self.update_params(learning_rate)
        total_loss += self.cross_entropy_loss(y_train[start_idx:end_idx], output)

      print(f"Epoch {epoch}, Loss {total_loss/num_batches}, Accuracy {self.accuracy(x_train, y_train)}")

  def one_hot(self, y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1

    return one_hot.flatten()
  
  def save(self, filename):
    with open(filename, 'wb') as file:
      pickle.dump(self, file)