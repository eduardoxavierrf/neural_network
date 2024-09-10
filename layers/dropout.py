import numpy as np

class Dropout:
  def __init__(self, drop_rate: float) -> None:
    self.drop_rate = drop_rate
    self.mask = None

  def forward(self, inputs: np.ndarray, is_training: bool = False):
    if is_training:
      self.mask = np.random.binomial(1, 1 - self.drop_rate, size=inputs.shape)

      return (inputs * self.mask) / (1 - self.drop_rate)
    else:
      return inputs
    
  def backward(self, delta):
    if self.mask is not None:
      return (delta * self.mask) / (1 - self.drop_rate)
    else:
      return delta
    
  def update_params(self, learning_rate):
    pass