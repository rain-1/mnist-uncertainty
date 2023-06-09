import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import data
import models
import utilities
import hypers


X, Y = data.load_data_all_folders("data/mnist_png/testing")
#print("{} {}".format(X.shape, Y.shape))
dataset = TensorDataset(X, Y)
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=hypers.batch_size, shuffle=True)

model = models.latest_model()
utilities.load_model(model)

# no grad
successful_predictions = 0
total_predictions = 0
for i, (inputs, labels) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs)
    successful_predictions += torch.sum(torch.eq(torch.argmax(outputs, dim=1, keepdim=True), torch.argmax(labels, dim=1, keepdim=True)))
    total_predictions += labels.shape[0]

print("{} out of {} successful. {:.2f}% accuracy".format(successful_predictions, total_predictions, 100.0*successful_predictions/total_predictions))
