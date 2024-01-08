import torch.nn as nn
import torch.nn.functional as F


class OsuModel(nn.Module):
    def __init__(self):
        super(OsuModel, self).__init__()

        # self.fc1 = nn.Linear(80 * 60, 100)
        # self.dropout = nn.Dropout(0.5)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(80 * 60, 100)
        self.linear2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.5)

        self.act = nn.ReLU()

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)

        x = self.flat(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x
