import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

class LinearRepeat(nn.Module):

  def __init__(self, in_features, out_features, repeats):
    super(LinearRepeat, self).__init__()
    self.repeats = repeats
    self.out_features = out_features
    self.linear = nn.Linear(in_features, repeats*out_features)

  def forward(self, x):
    y = self.linear(x)
    return torch.reshape(y, (*(y.size()[:-1]), self.repeats, self.out_features))


# texts, 3 tokens, 4 features
original_input = Tensor(
  [
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ],
    [
      [11, 22, 33, 44],
      [55, 66, 77, 88],
      [99, 1010, 1111, 1212],
    ]
  ])
print(original_input.size())
linear_repeat = LinearRepeat(4, 2, 5)
print(linear_repeat(original_input).size())