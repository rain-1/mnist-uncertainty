# Load a PNG up as a pytorch tensor

import torch
import torchvision.transforms as transforms
from PIL import Image
import os

example_image_path = "data/mnist_png/training/0/"

# Define a transformation to convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor() # Convert the PIL image to a PyTorch tensor
])

# X will be [N, 784]
# Y will be [N, 10] where the values are 0,0,1,0,0,0,0,0 for 2

def load_data_folder(path, label):
    images = []
    labels = []

    label_tensor = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=10)

    # Loop over all files in the folder
    for filename in os.listdir(path):
        # Check if the file ends with '.png'
        if filename.endswith('.png'):
            # Load the image using PIL
            image = Image.open(path + filename)

            # Apply the transformation to the image
            tensor = transform(image)

            reshaped_tensor = tensor.reshape(1, -1)

            images.append(reshaped_tensor)
            labels.append(label_tensor)
    
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


X, Y = load_data_folder(example_image_path, 0)
print("{} {}".format(X.shape, Y.shape))
