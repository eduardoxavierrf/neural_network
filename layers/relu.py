import numpy as np

class ReLU:
  def forward(self, inputs):
    self.inputs = inputs
    return np.maximum(inputs, 0)
  
  def backward(self, delta):
    return delta * (self.inputs > 0)
  
  def update_params(self, learning_rate):
    pass