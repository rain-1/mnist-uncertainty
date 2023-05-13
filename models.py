import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.name = "MNISTClassifier"
        self.fc1 = nn.Linear(28*28, 128)   # input layer
        self.fc2 = nn.Linear(128, 64)      # hidden layer
        self.fc3 = nn.Linear(64, 10)       # output layer

    def forward(self, x):
        x = x.view(-1, 28*28)             # flatten the input tensor
        x = nn.functional.relu(self.fc1(x))   # apply ReLU activation to the output of the input layer
        x = nn.functional.relu(self.fc2(x))   # apply ReLU activation to the output of the hidden layer
        x = self.fc3(x)                   # apply the output layer
        return x


