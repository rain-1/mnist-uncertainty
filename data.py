# Load a PNG up as a pytorch tensor

import torch
import torchvision.transforms as transforms
from PIL import Image

example_image_path = "data/mnist_png/training/0/1.png"

# Load the image using PIL
image = Image.open(example_image_path)

# Define a transformation to convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor() # Convert the PIL image to a PyTorch tensor
])

# Apply the transformation to the image
tensor = transform(image)

# Print the shape and data type of the tensor
print(tensor.shape, tensor.dtype)
reshaped_tensor = tensor.reshape(1, -1)
print(reshaped_tensor.shape, reshaped_tensor.dtype)
#print(tensor)

# torch.Size([1, 28, 28]) torch.float32
# torch.Size([1, 784]) torch.float32
