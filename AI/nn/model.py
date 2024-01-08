import torch.nn as nn


class OsuModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(80 * 60, 100)
        self.linear2 = nn.Linear(100, 2)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.flat(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        return x
