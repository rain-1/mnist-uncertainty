import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import sys
import random

import data
import models
import utilities
import hypers

image_list = ['./data/mnist_png/testing/0/3.png',
 './data/mnist_png/testing/0/69.png',
 './data/mnist_png/testing/0/296.png',
 './data/mnist_png/testing/1/154.png',
 './data/mnist_png/testing/1/190.png',
 './data/mnist_png/testing/1/1097.png',
 './data/mnist_png/testing/2/174.png',
 './data/mnist_png/testing/2/225.png',
 './data/mnist_png/testing/2/1016.png',
 './data/mnist_png/testing/3/402.png',
 './data/mnist_png/testing/3/437.png',
 './data/mnist_png/testing/3/1205.png']

###

def get_color(value):
    red = int(255 * 0.5 * (1 - value))
    green = int(255 * value)
    blue = 0
    return red, green, blue

def generate_color(level):
    level = max(0, level)
    level = min(1, level)
    a,b,c = get_color(level)
    #random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)
    return "#{:02x}{:02x}{:02x}".format(a,b,c)

###

# Generate HTML code
html = '<html>\n'
html += '<body style="background-color:grey;">\n'

for image_path in image_list:

    X = data.load_individual(image_path)

    model = models.latest_model()
    utilities.load_model(model)

    outputs = model(X)

    # Add the image
    html += '<div style="margin:10px;text-align:center;border:1px solid black;display:inline-block;">\n'
    html += "<img src='../{}' style='width:96;'>\n".format(image_path)

    # Add the dots
    html += '<div style="text-align:center;">\n'
    html += "<p>{}</p>".format(torch.argmax(outputs))
    html += '<p>'
    for level in outputs.squeeze().tolist():
        # Generate a random color for each dot
        color = generate_color(level)
#        html += '<span style="display:inline-block;width:20px;height:20px;margin:2px;background-color:{};"></span>\n'.format("?")
        html += '<span style="color:{}">&#9679;</span>'.format(color)
    html += '</p>\n'
    html += '</div>\n'
    html += '</div>\n'

html += '</body>\n'
html += '</html>'

print(html)
