import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function


# class Net(nn.Module):
#     def __init__(self, input_size):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_size, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return torch.sigmoid(x)

class Net_CENSUS(nn.Module):

    def __init__(self, input_shape):
        super(Net_CENSUS, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, 0.1, training=self.training)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)
    
        y = self.fc4(hidden)
        # y = F.dropout(y, 0.1)

        return torch.sigmoid(y)

