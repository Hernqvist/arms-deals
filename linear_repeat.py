import torch.nn as nn
import torch

LABELS = ("Weapon", "Buyer", "Buyer Country", "Seller", "Seller Country", "Quantity", "Price", "Date")

class LinearRepeat(nn.Module):

    def __init__(self, in_features, out_features, repeats):
        super(LinearRepeat, self).__init__()
        self.repeats = repeats
        self.out_features = out_features
        self.linear = nn.Linear(in_features, repeats*out_features)

    def forward(self, x):
        y = self.linear(x)
        return torch.reshape(y, (*(y.size()[:-1]), self.repeats, self.out_features))