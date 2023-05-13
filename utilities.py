import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import data
import models
import utilities

model_path = "models"
def save_model(model):
    torch.save(model.state_dict(), "{}/{}.pt".format(model_path, model.name))

def load_model(model):
    state_dict = torch.load("{}/{}.pt".format(model_path, model.name))
    #print(state_dict)
    model.load_state_dict(state_dict)
