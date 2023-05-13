import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # input layer
        self.fc2 = nn.Linear(128, 64)      # hidden layer
        self.fc3 = nn.Linear(64, 10)       # output layer

    def forward(self, x):
        x = x.view(-1, 28*28)             # flatten the input tensor
        x = nn.functional.relu(self.fc1(x))   # apply ReLU activation to the output of the input layer
        x = nn.functional.relu(self.fc2(x))   # apply ReLU activation to the output of the hidden layer
        x = self.fc3(x)                   # apply the output layer
        return x



# Define the neural network and optimizer
model = MNISTClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function (cross-entropy)
criterion = nn.CrossEntropyLoss()

# Train the neural network
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

