import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class SimpleLinear(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleLinear, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.main(x)
        return feature
