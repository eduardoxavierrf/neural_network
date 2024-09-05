import numpy as np

class Softmax:
  def forward(self, inputs):
    exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    
    return exps / np.sum(exps, axis=1, keepdims=True)
  
  def backward(self, delta):
    return delta
  
  def update_params(self, learning_rate):
    pass