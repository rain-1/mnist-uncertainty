import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import data
import models
import utilities
import hypers

X, Y = data.load_data_all_folders("data/mnist_png/training")
print("{} {}".format(X.shape, Y.shape))
dataset = TensorDataset(X, Y)
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=hypers.batch_size, shuffle=True)

# Define the neural network and optimizer
model = models.latest_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function (cross-entropy)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

train_loss = []

# Train the neural network
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        #loss = criterion(outputs, labels)
        loss = F.nll_loss(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        train_loss.append(loss.item())
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss/100:.6f}")
            running_loss = 0.0

# Save the model
#torch.save(model, "{}/{}.pt".format(model_path, model.name))
utilities.save_model(model)

# Plot the loss values
plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig("plots/loss_{}_{}.png".format(model.name, datetime.datetime.now()))
plt.show()
