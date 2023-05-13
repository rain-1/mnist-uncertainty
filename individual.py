import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import sys

import data
import models
import utilities
import hypers

image_path = sys.argv[1]

X = data.load_individual(image_path)

model = models.MNISTClassifier()
utilities.load_model(model)

outputs = model(X)
print(outputs)
print(torch.argmax(outputs))
