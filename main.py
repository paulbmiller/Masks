#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
from torchvision import transforms, utils
from skimage import io, transform


def load_image(filename):
    """Loads the image with the given name in the folder images."""
    image = io.imread(os.path.join('images//', filename))
    return image


def show_image(image):
    """Displays an image with matplotlib."""
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def hflip(image):
    """Flips the image on the x axis."""
    image = image[::-1]
    return image


def vflip(image):
    """Flips the image on the y axis."""
    i = 0
    while i < len(image):
        image[i] = image[i][::-1]
        i += 1

    return image


def change_channel(image, channel_id, percentage):
    """Modifies an output channel by multiplying it by a percentage."""
    mask = np.ones(image.shape, dtype=float)
    indices = channel_id * np.ones((image.shape[0], image.shape[1], 1),
                                   dtype=int)
    np.put_along_axis(mask, indices, percentage, axis=2)
    return np.clip((image * mask).astype(int), 0, 255)


def change_channels(image, red=1, green=1, blue=1):
    if red != 1:
        image = change_channel(image, 0, red)
    if blue != 1:
        image = change_channel(image, 1, green)
    if green != 1:
        image = change_channel(image, 2, blue)

    return image


class Display():
    def __init__(self, filename):
        self.image = load_image(filename)
        self.root = tk.Tk()
        self.root.title("GUI Image manipulation")
        
        f = Figure(figsize=(6, 6), dpi=100)
        self.a = f.add_subplot(111)
        self.a.imshow(self.image)
        self.a.axis('off')

        self.canvas = FigureCanvasTkAgg(f, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)
        
        self.button_red = tk.Button(self.root, command=self.reset_red, text='Reset', bg='red')
        self.slider_red = tk.Scale(self.root, from_=0, to=3, resolution=0.05, orient=tk.HORIZONTAL)
        self.slider_red.set(1)
        self.button_red.grid(row=1, column=0)
        self.slider_red.grid(row=1, column=1, sticky='EW')
        
        self.button_green = tk.Button(self.root, command=self.reset_green, text='Reset', bg='green')
        self.slider_green = tk.Scale(self.root, from_=0, to=3, resolution=0.05, orient=tk.HORIZONTAL)
        self.slider_green.set(1)
        self.button_green.grid(row=2, column=0)
        self.slider_green.grid(row=2, column=1, sticky='EW')
        
        self.button_blue = tk.Button(self.root, command=self.reset_blue, text='Reset', bg='blue')
        self.slider_blue = tk.Scale(self.root, from_=0, to=3, resolution=0.05, orient=tk.HORIZONTAL)
        self.slider_blue.set(1)
        self.button_blue.grid(row=3, column=0)
        self.slider_blue.grid(row=3, column=1, sticky='EW')
        self.slider_red.bind("<ButtonRelease-1>", self.update_colors)
        self.slider_green.bind("<ButtonRelease-1>", self.update_colors)
        self.slider_blue.bind("<ButtonRelease-1>", self.update_colors)

        self.slider_values = [1., 1., 1.]

        self.root.mainloop()

    def reset_red(self):
        self.slider_red.set(1)
        self.update_colors(None)

    def reset_green(self):
        self.slider_green.set(1)
        self.update_colors(None)
    
    def reset_blue(self):
        self.slider_blue.set(1)
        self.update_colors(None)

    def update_colors(self, event):
        self.image = load_image('test.jpg')
        self.slider_values[0] = self.slider_red.get()
        self.slider_values[1] = self.slider_green.get()
        self.slider_values[2] = self.slider_blue.get()
        self.image = change_channels(self.image, self.slider_values[0], self.slider_values[1], self.slider_values[2])
        self.redraw()

    def redraw(self):
        self.a.imshow(self.image)
        self.canvas.draw()


if __name__ == '__main__':
    display = Display('test.jpg')
