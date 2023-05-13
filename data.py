# Load a PNG up as a pytorch tensor

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

# Define a transformation to convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor() # Convert the PIL image to a PyTorch tensor
])

# X will be [N, 784]
# Y will be [N, 10] where the values are 0,0,1,0,0,0,0,0 for 2

# Only load 200 images for quick testing/development
fast_mode_for_dev = True
fast_mode_for_dev = False

## Flatten the 2D grid down to a 1D tensor
reshape_tensors = True
#reshape_tensors = False

def load_data_folder(path, label):
    images = []
    labels = []

    label_tensor = torch.nn.functional.one_hot(torch.tensor([label]), num_classes=10).float()

    dir_listing = os.listdir(path)
    if fast_mode_for_dev:
        dir_listing = dir_listing[:200]

    # Loop over all files in the folder
    for filename in tqdm(dir_listing):
        # Check if the file ends with '.png'
        if filename.endswith('.png'):
            # Load the image using PIL
            image = Image.open(path + filename)

            # Apply the transformation to the image
            tensor = transform(image)

            if reshape_tensors:
                reshaped_tensor = tensor.reshape(1, -1)
            else:
                reshaped_tensor = tensor

            images.append(reshaped_tensor)
            labels.append(label_tensor)
    
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)

def load_data_all_folders(base_path):
    print("Loading 10 sets of data")
    X, Y = [], []
    for i in range(10):
        X_append, Y_append = load_data_folder("{}/{}/".format(base_path,i), i)
        X.append(X_append)
        Y.append(Y_append)
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)

def load_individual(path):
    images = []

    # Load the image using PIL
    image = Image.open(path)

    # Apply the transformation to the image
    tensor = transform(image)

    if reshape_tensors:
        reshaped_tensor = tensor.reshape(1, -1)
    else:
        reshaped_tensor = tensor
    
    return reshaped_tensor

if __name__ == "__main__":
    example_image_path = "data/mnist_png/training/"

    X, Y = load_data_all_folders(example_image_path)
    print("{} {}".format(X.shape, Y.shape))
    # torch.Size([60000, 784]) torch.Size([60000, 10])
