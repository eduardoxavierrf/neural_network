import numpy as np

class Linear:
  def __init__(self, input_size: int, output_size: int) -> None:
    self.weight = np.random.randn(input_size, output_size) * 0.01
    self.bias = np.zeros((1, output_size))

  def forward(self, inputs: np.ndarray):
    self.inputs = inputs
    self.z  = np.dot(self.inputs, self.weight) + self.bias

    return self.z

  def backward(self, delta):
    m = self.inputs.shape[0]
    
    self.dW = np.dot(self.inputs.T, delta) / m
    self.db = np.sum(delta, axis=0, keepdims=True) / m

    return np.dot(delta, self.weight.T)
  
  def update_params(self, learning_rate):
    self.weight -= learning_rate * self.dW
    self.bias -= learning_rate * self.db