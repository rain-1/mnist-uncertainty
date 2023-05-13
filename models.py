import torch
import torch.nn as nn
import torch.nn.functional as F

# got this from ChatGPT, interesting that it seems to be using the same numbers as https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
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
        # TODO: add log softmax here?
        #output = F.log_softmax(x, dim=1)
        #print("output shape is {}".format(x.shape))
        return x

# https://nextjournal.com/gkoehler/pytorch-mnist
class NextJournalNet(nn.Module):
    def __init__(self):
        super(NextJournalNet, self).__init__()
        self.name = "NextJournalNet"
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# https://flower.dev/docs/example-walkthrough-pytorch-mnist.html
# UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result
# in an error in a future release. To retain the behavior and silence this warning, please use
# dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2
# spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).

class MNISTNet(nn.Module):
    """Simple CNN adapted from Pytorch's 'Basic MNIST Example'."""

    def __init__(self) -> None:
        super(MNISTNet, self).__init__()
        self.name = "MNISTNet"
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Compute forward pass.

        Parameters
        ----------
        x: Tensor
            Mini-batch of shape (N,28,28) containing images from MNIST dataset.


        Returns
        -------
        output: Tensor
            The probability density of the output being from a specific class given the input.

        """

        # we need to reshape the input from [-, 784] to [-, 28, 28]
        #print(x.shape)
        x = x.view(-1, 28, 28)
        #print(x.shape)
        x = x.unsqueeze(1)
        #print(x.shape)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        #print("output shape is {}".format(output.shape))
        return output

#latest_model = MNISTClassifier
#latest_model = NextJournalNet
latest_model = MNISTNet

# output shape is torch.Size([32, 10])
# output shape is torch.Size([32, 10])
