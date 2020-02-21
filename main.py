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
import torch.nn.functional as F


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


def apply_conv(image, kernel):
    """Applies a 3x3 filter on the image."""
    weights = kernel.return_weights()
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (3, 0, 1, 2))
    image = torch.DoubleTensor(image)
    weights = torch.DoubleTensor(weights)
    weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    output = F.conv2d(image, weights, padding=1).numpy()
    output = output.astype(int)
    output = np.transpose(output, (1, 2, 3, 0))[0]

    return np.clip(output, 0, 255)


def apply_blur(image):
    weights = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return apply_conv(image, weights)


def apply_gaussianblur(image):
    weights = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return apply_conv(image, weights)


class Kernel():
    def __init__(self, weights=np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]]),
                 coeff=1.):
        self.weights = weights
        self.coeff = coeff

    def return_weights(self):
        return self.coeff * self.weights


class Display():
    def __init__(self, filename):
        self.image = load_image(filename)
        self.img_nochannel_change = np.copy(self.image)
        self.root = tk.Tk()
        self.root.title("GUI Image manipulation")

        f = Figure(figsize=(6, 6), dpi=100)
        self.a = f.add_axes([0, 0, 1, 1])
        self.a.imshow(self.image)
        self.a.axis('off')

        self.canvas = FigureCanvasTkAgg(f, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=10,
                                         columnspan=2)

        self.button_blur = tk.Button(self.root, command=self.blur,
                                     text='Box blur')
        self.button_blur.grid(row=0, column=2)

        self.button_gaussianblur = tk.Button(self.root,
                                             command=self.gaussianblur,
                                             text='Gaussian blur')
        self.button_gaussianblur.grid(row=1, column=2)

        self.button_edgedetect = tk.Button(self.root, command=self.edgedetect,
                                           text='Edge')
        self.button_edgedetect.grid(row=0, column=3)

        self.button_sharpen = tk.Button(self.root, command=self.sharpen,
                                        text='Sharpen')
        self.button_sharpen.grid(row=1, column=3)

        self.entries = self.init_entries()

        self.main_mult = tk.Entry(self.root)
        self.main_mult.insert(0, '1')
        self.main_mult.grid(row=6, column=3)

        self.button_custom = tk.Button(self.root, command=self.custom_kernel,
                                       text='Apply custom kernel')
        self.button_custom.grid(row=7, column=3)
        
        self.button_reset_img = tk.Button(self.root, command=self.reset_image,
                                          text='Reset image')
        self.button_reset_img.grid(row=8, column=2)

        self.button_reset_k = tk.Button(self.root, command=self.reset_kernel,
                                        text='Reset kernel')
        self.button_reset_k.grid(row=8, column=4)

        self.button_red = tk.Button(self.root, command=self.reset_red,
                                    text='Reset', bg='red')
        self.slider_red = tk.Scale(self.root, from_=0, to=3, resolution=0.05,
                                   orient=tk.HORIZONTAL)
        self.slider_red.set(1)
        self.button_red.grid(row=10, column=0)
        self.slider_red.grid(row=10, column=1, sticky='EW')

        self.button_green = tk.Button(self.root, command=self.reset_green,
                                      text='Reset', bg='green')
        self.slider_green = tk.Scale(self.root, from_=0, to=3, resolution=0.05,
                                     orient=tk.HORIZONTAL)
        self.slider_green.set(1)
        self.button_green.grid(row=11, column=0)
        self.slider_green.grid(row=11, column=1, sticky='EW')

        self.button_blue = tk.Button(self.root, command=self.reset_blue,
                                     text='Reset', bg='blue')
        self.slider_blue = tk.Scale(self.root, from_=0, to=3, resolution=0.05,
                                    orient=tk.HORIZONTAL)
        self.slider_blue.set(1)
        self.button_blue.grid(row=12, column=0)
        self.slider_blue.grid(row=12, column=1, sticky='EW')
        self.slider_red.bind("<ButtonRelease-1>", self.update_colors)
        self.slider_green.bind("<ButtonRelease-1>", self.update_colors)
        self.slider_blue.bind("<ButtonRelease-1>", self.update_colors)

        self.slider_values = [1., 1., 1.]

        self.root.mainloop()

    def blur(self):
        kernel = Kernel(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), 1/9)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel)
        self.redraw()

    def custom_kernel(self):
        w = self.weights_from_entries()
        coeff = float(self.main_mult.get())
        kernel = Kernel(w, coeff)
        self.image = apply_conv(self.image, kernel)
        self.redraw()

    def edgedetect(self):
        kernel = Kernel(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                        1.)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel)
        self.redraw()

    def get_entry(self, index):
        return float(self.entries[index].get())

    def weights_from_entries(self):
        w = np.array([])
        for i in range(9):
            w = np.append(w, self.get_entry(i))
        w = w.reshape((3, 3))
        return w

    def gaussianblur(self):
        kernel = Kernel(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), 1/19)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel)
        self.redraw()

    def sharpen(self):
        kernel = Kernel(np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), 1.)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel)
        self.redraw()

    def update_kernel(self, kernel):
        w = kernel.weights
        coeff = kernel.coeff
        for i in range(3):
            for j in range(3):
                self.entries[i+j*3].delete(0, 'end')
                self.entries[i+j*3].insert(0, w[j, i])
        self.main_mult.delete(0, 'end')
        self.main_mult.insert(0, coeff)

    def init_entries(self):
        list_entries = []

        for i in range(3):
            for j in range(3):
                entry = tk.Entry(self.root)
                if i == 1 and j == 1:
                    entry.insert(0, '1')
                else:
                    entry.insert(0, '0')
                list_entries.append(entry)
                entry.grid(row=3+i, column=2+j)

        return list_entries

    def reset_image(self):
        self.image = load_image('test.jpg')
        self.redraw()

    def reset_kernel(self):
        kernel = Kernel()
        self.update_kernel(kernel)

    def reset_red(self):
        self.slider_red.set(1)
        self.slider_values[0] = 1.
        self.update_colors(None)

    def reset_green(self):
        self.slider_green.set(1)
        self.slider_values[1] = 1.
        self.update_colors(None)

    def reset_blue(self):
        self.slider_blue.set(1)
        self.slider_values[2] = 1.
        self.update_colors(None)

    def update_colors(self, event):
        self.image = load_image('test.jpg')
        self.slider_values[0] = self.slider_red.get()
        self.slider_values[1] = self.slider_green.get()
        self.slider_values[2] = self.slider_blue.get()
        self.image = change_channels(self.image,
                                     self.slider_values[0],
                                     self.slider_values[1],
                                     self.slider_values[2])
        self.redraw()

    def redraw(self):
        self.a.imshow(self.image)
        self.canvas.draw()


if __name__ == '__main__':
    display = Display('test.jpg')
