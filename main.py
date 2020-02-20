#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from skimage import io, transform

def load_image(filename):
    image = io.imread(os.path.join('images//', filename))
    return image

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def hflip(image):
    image = image[::-1]
    return image

def vflip(image):
    i = 0
    while i < len(image):
        image[i] = image[i][::-1]
        i+=1
    return image

def change_channel(image, channel_id, val2):
    indices = channel_id * np.ones((image.shape[0], image.shape[1], 1))
    np.put_along_axis(image, indices.astype(int), val2, axis=2)
    return image

if __name__ == '__main__':
    image = load_image('test.jpg')
    show_image(image)
    im2 = np.copy(image)
    im2 = change_channel(im2, 0, 0)
    print('No red')
    show_image(im2)
    im3 = np.copy(image)
    im3 = change_channel(im3, 1, 0)
    print('No green')
    show_image(im3)
    im4 = np.copy(image)
    im4 = change_channel(im4, 2, 0)
    print('No blue')
    show_image(im4)
