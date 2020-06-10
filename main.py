#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage import io
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


def apply_conv(image, kernel, grey=False):
    """Applies a convolutional filter on the image."""
    weights = kernel.return_weights()
    image = np.expand_dims(image, axis=0)
    if grey:
        image = np.expand_dims(image, axis=3)
    image = np.transpose(image, (3, 0, 1, 2))
    image = torch.DoubleTensor(image)
    weights = torch.DoubleTensor(weights)
    if kernel.size == 2:
        padx = image.shape[0] % 2
        pady = image.shape[1] % 2
    elif kernel.size == 3:
        padx = 1
        pady = 1
    elif kernel.size == 5:
        padx = 2
        pady = 2
    weights = weights.view(1, 1, kernel.size, kernel.size).repeat(1, 1, 1, 1)
    output = F.conv2d(image, weights, padding=(padx, pady)).numpy()
    output = output.astype(int)
    output = np.transpose(output, (1, 2, 3, 0))[0]
    output = np.clip(output, 0, 255)
    if grey:
        output = np.squeeze(output, axis=2)

    return output


def apply_blur(image):
    weights = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return apply_conv(image, weights)


def apply_gaussianblur(image):
    weights = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return apply_conv(image, weights)


def corrected_grey(pixel):
    return 0.2126*pixel[0] + 0.7152*pixel[1] + 0.0722*pixel[2]


class Kernel():
    def __init__(self, weights=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                 coeff=1.):
        self.weights = weights
        self.coeff = coeff
        self.size = self.weights.shape[0]

    def to_5x5(self):
        if self.size == 5:
            return self.weights
        elif self.size == 3:
            w = np.zeros((5, 5))
            w[1:4, 1:4] = self.weights
            return w
        elif self.size == 2:
            w = np.zeros((5, 5))
            w[2:4, 2:4] = self.weights
            return w
        elif self.size == 1:
            w = np.zeros((5, 5))
            w[2, 2] = self.weights
            return w

    def return_weights(self):
        return self.coeff * self.weights


class Display():
    def __init__(self, filename):
        self.filename = filename
        self.image = load_image(filename)
        self.img_nochannel_change = np.copy(self.image)
        self.root = tk.Tk()
        self.root.title("GUI Image manipulation")

        f = Figure(figsize=(6, 6), dpi=100)
        self.a = f.add_axes([0, 0, 1, 1])
        self.a.imshow(self.image)
        self.a.axis('off')
        self.grey = False

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
        self.button_edgedetect.grid(row=0, column=5)

        self.button_sharpen = tk.Button(self.root, command=self.sharpen,
                                        text='Sharpen')
        self.button_sharpen.grid(row=0, column=3)

        self.button_sobel = tk.Button(self.root, command=self.sobel,
                                      text='Sobel')
        self.button_sobel.grid(row=1, column=5)

        self.button_grey = tk.Button(self.root, command=self.to_grey,
                                     text='Grey')
        self.button_grey.grid(row=0, column=4)

        self.button_corr_grey = tk.Button(self.root,
                                          command=self.to_corrected_grey,
                                          text='Corr. grey')
        self.button_corr_grey.grid(row=1, column=4)

        self.button_canny = tk.Button(self.root, command=self.gaussian_filter,
                                      text='Gaussian filter')
        self.button_canny.grid(row=2, column=2)

        self.button_prewitt = tk.Button(self.root, command=self.prewitt,
                                        text='Prewitt')
        self.button_prewitt.grid(row=2, column=5)

        self.button_roberts = tk.Button(self.root, command=self.roberts,
                                        text='Roberts')
        self.button_roberts.grid(row=0, column=6)

        self.entries = self.init_entries()

        self.main_mult = tk.Entry(self.root)
        self.main_mult.insert(0, '1')
        self.main_mult.grid(row=8, column=4)

        self.button_custom = tk.Button(self.root, command=self.custom_kernel,
                                       text='Apply kernel')
        self.button_custom.grid(row=8, column=3)

        self.entry_file = tk.Entry(self.root)
        self.entry_file.grid(row=9, column=4)
        self.entry_file.insert(0, self.filename)

        self.button_load_img = tk.Button(self.root, command=self.load_img,
                                         text='Load image')
        self.button_load_img.grid(row=9, column=5)

        self.button_reset_img = tk.Button(self.root, command=self.reset_image,
                                          text='Reset image')
        self.button_reset_img.grid(row=9, column=3)

        self.button_reset_k = tk.Button(self.root, command=self.reset_kernel,
                                        text='Reset kernel')
        self.button_reset_k.grid(row=8, column=5)

        self.button_swap021 = tk.Button(self.root,
                                        command=lambda: self.swap('021'),
                                        text='RBG')
        self.button_swap021.grid(row=10, column=2)

        self.button_swap120 = tk.Button(self.root,
                                        command=lambda: self.swap('120'),
                                        text='GBR')
        self.button_swap120.grid(row=10, column=3)

        self.button_swap102 = tk.Button(self.root,
                                        command=lambda: self.swap('102'),
                                        text='GRB')
        self.button_swap102.grid(row=10, column=4)

        self.button_swap201 = tk.Button(self.root,
                                        command=lambda: self.swap('201'),
                                        text='BRG')
        self.button_swap201.grid(row=10, column=5)

        self.button_swap210 = tk.Button(self.root,
                                        command=lambda: self.swap('210'),
                                        text='BGR')
        self.button_swap210.grid(row=10, column=6)

        self.button_saveas = tk.Button(self.root,
                                       command=self.save_img,
                                       text='Save image as')
        self.button_saveas.grid(row=11, column=3)

        self.entry_x0 = tk.Entry(self.root)
        self.entry_x0.grid(row=12, column=2)
        self.entry_x0.insert(0, '0')

        self.entry_x1 = tk.Entry(self.root)
        self.entry_x1.grid(row=12, column=3)
        self.entry_x1.insert(0, self.image.shape[1])

        self.entry_y0 = tk.Entry(self.root)
        self.entry_y0.grid(row=12, column=4)
        self.entry_y0.insert(0, '0')

        self.entry_y1 = tk.Entry(self.root)
        self.entry_y1.grid(row=12, column=5)
        self.entry_y1.insert(0, self.image.shape[0])

        self.button_crop = tk.Button(self.root, command=self.crop,
                                     text='Crop')
        self.button_crop.grid(row=12, column=6)

        self.entry_saveas = tk.Entry(self.root)
        self.entry_saveas.grid(row=11, column=4)

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

    def crop(self):
        x0 = int(self.entry_x0.get())
        x1 = int(self.entry_x1.get())
        y0 = int(self.entry_y0.get())
        y1 = int(self.entry_y1.get())
        self.image = self.image[y0:y1, x0:x1]
        new_width = x1-x0
        new_height = y1-y0
        self.entry_x0.delete(0, 'end')
        self.entry_x0.insert(0, '0')
        self.entry_x1.delete(0, 'end')
        self.entry_x1.insert(0, new_width)
        self.entry_y0.delete(0, 'end')
        self.entry_y0.insert(0, '0')
        self.entry_y1.delete(0, 'end')
        self.entry_y1.insert(0, new_height)
        self.redraw(self.grey)

    def reset_crop(self):
        self.entry_x0.delete(0, 'end')
        self.entry_x0.insert(0, '0')
        self.entry_x1.delete(0, 'end')
        self.entry_x1.insert(0, self.image.shape[1])
        self.entry_y0.delete(0, 'end')
        self.entry_y0.insert(0, '0')
        self.entry_y1.delete(0, 'end')
        self.entry_y1.insert(0, self.image.shape[0])

    def swap(self, order):
        if self.grey:
            print("Can't swap channels on a greyscale image")
            return
        if order == '021':
            col1 = self.image[:, :, 1].copy()
            col2 = self.image[:, :, 2].copy()
            self.image[:, :, 1], self.image[:, :, 2] = col2, col1
        elif order == '120':
            col0 = self.image[:, :, 0].copy()
            col1 = self.image[:, :, 1].copy()
            col2 = self.image[:, :, 2].copy()
            self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2] =\
                col1, col2, col0
        elif order == '102':
            col0 = self.image[:, :, 0].copy()
            col1 = self.image[:, :, 1].copy()
            self.image[:, :, 0], self.image[:, :, 1] = col1, col0
        elif order == '201':
            col0 = self.image[:, :, 0].copy()
            col1 = self.image[:, :, 1].copy()
            col2 = self.image[:, :, 2].copy()
            self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2] =\
                col2, col0, col1
        elif order == '210':
            col0 = self.image[:, :, 0].copy()
            col2 = self.image[:, :, 2].copy()
            self.image[:, :, 0], self.image[:, :, 2] = col2, col0
        self.redraw(self.grey)

    def blur(self):
        kernel = Kernel(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), 1/9)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel, self.grey)
        self.redraw(self.grey)

    def gaussian_filter(self):
        x_kern_weights = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4],
                                  [5, 12, 15, 12, 5], [4, 9, 12, 9, 4],
                                  [2, 4, 5, 4, 2]])
        y_kern_weights = np.transpose(x_kern_weights)
        k1 = Kernel(x_kern_weights, 1/159)
        self.update_kernel(k1)
        k2 = Kernel(y_kern_weights, 1/159)
        Gx = apply_conv(self.image, k1, self.grey)
        Gy = apply_conv(self.image, k2, self.grey)
        G = np.sqrt(Gx*Gx + Gy*Gy)
        self.image = np.clip(G, 0, 255).astype(int)
        self.redraw(self.grey)

    def prewitt(self):
        if not self.grey:
            self.image = np.expand_dims(np.mean(self.image, axis=2), axis=2)
        x_kern_weights = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        y_kern_weights = np.transpose(x_kern_weights)
        k1 = Kernel(x_kern_weights)
        k2 = Kernel(y_kern_weights)
        self.update_kernel(k1)
        Gx = apply_conv(self.image, k1, self.grey)
        Gy = apply_conv(self.image, k2, self.grey)
        G = np.sqrt(Gx*Gx + Gy*Gy)
        if not self.grey:
            self.image = np.mean(self.image, axis=2)
        self.image = np.clip(G, 0, 255).astype(int)
        if not self.grey:
            self.image = np.squeeze(self.image, axis=2)
        self.redraw(grey=True)

    def custom_kernel(self):
        w = self.weights_from_entries()
        coeff = float(self.main_mult.get())
        kernel = Kernel(w, coeff)
        self.image = apply_conv(self.image, kernel, self.grey)
        self.redraw(self.grey)

    def edgedetect(self):
        if not self.grey:
            self.image = np.expand_dims(np.mean(self.image, axis=2), axis=2)
        kernel = Kernel(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
                        1.)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel, self.grey)
        if not self.grey:
            self.image = np.squeeze(self.image, axis=2)
        self.redraw(grey=True)

    def get_entry(self, index):
        return float(self.entries[index].get())

    def load_img(self):
        entry = self.entry_file.get()
        try:
            self.image = load_image(entry)
            self.filename = entry
            self.grey = False
            self.reset_crop()
            self.redraw()
        except ValueError as e:
            print(e)
            print('The entry for the path is probably empty.')
        except FileNotFoundError:
            print('Invalid filename for loading the image.')

    def save_img(self):
        filename = self.entry_saveas.get()
        try:
            path = os.path.join('images\\', filename)
            io.imsave(path, self.image.astype(np.uint8))
            print('File saved as {}'.format(path))
        except TypeError:
            print('Wrong filename for saving.')
        except ValueError:
            print('Wrong filename for saving')

    def to_grey(self):
        if self.grey:
            return
        self.grey = True
        self.image = np.mean(self.image, axis=2)
        self.redraw(self.grey)

    def to_corrected_grey(self):
        if self.grey:
            return
        self.grey = True
        self.image = np.apply_along_axis(corrected_grey, 2, self.image)
        self.redraw(self.grey)

    def weights_from_entries(self):
        w = np.array([])
        for i in range(25):
            w = np.append(w, self.get_entry(i))
        w = w.reshape((5, 5))
        return w

    def gaussianblur(self):
        kernel = Kernel(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), 1/19)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel, self.grey)
        self.redraw(self.grey)

    def roberts(self):
        if not self.grey:
            self.image = np.expand_dims(np.mean(self.image, axis=2), axis=2)
        k1 = Kernel(np.array([[1, 0], [0, -1]]), 1.)
        k2 = Kernel(np.array([[0, 1], [-1, 0]]), 1.)
        self.update_kernel(k1)
        Gx = apply_conv(self.image, k1, self.grey)
        Gy = apply_conv(self.image, k2, self.grey)
        G = np.sqrt(Gx*Gx + Gy*Gy)
        self.image = np.clip(G, 0, 255)
        if not self.grey:
            self.image = np.squeeze(self.image, axis=2).astype(int)
        self.redraw(grey=True)

    def sharpen(self):
        kernel = Kernel(np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), 1.)
        self.update_kernel(kernel)
        self.image = apply_conv(self.image, kernel, self.grey)
        self.redraw(self.grey)

    def sobel(self):
        if not self.grey:
            self.image = np.expand_dims(np.mean(self.image, axis=2), axis=2)
        k1 = Kernel(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), 1.)
        k2 = Kernel(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), 1.)
        self.update_kernel(k1)
        Gx = apply_conv(self.image, k1, self.grey)
        Gy = apply_conv(self.image, k2, self.grey)
        G = np.sqrt(Gx*Gx + Gy*Gy)
        self.image = np.clip(G, 0, 255)
        if not self.grey:
            self.image = np.squeeze(self.image, axis=2)
        self.redraw(grey=True)

    def update_kernel(self, kernel):
        w = kernel.to_5x5()
        coeff = kernel.coeff
        for i in range(5):
            for j in range(5):
                self.entries[i+j*5].delete(0, 'end')
                self.entries[i+j*5].insert(0, str(w[j, i]))
        self.main_mult.delete(0, 'end')
        self.main_mult.insert(0, coeff)

    def init_entries(self):
        list_entries = []

        for i in range(5):
            for j in range(5):
                entry = tk.Entry(self.root)
                if i == 2 and j == 2:
                    entry.insert(0, '1')
                else:
                    entry.insert(0, '0')
                list_entries.append(entry)
                entry.grid(row=3 + i, column=2 + j)

        return list_entries

    def reset_image(self):
        self.reset_crop()
        self.image = load_image(self.filename)
        self.slider_red.set(1)
        self.slider_values[0] = 1.
        self.slider_green.set(1)
        self.slider_values[1] = 1.
        self.slider_blue.set(1)
        self.slider_values[2] = 1.
        self.redraw(grey=False)

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
        self.image = load_image(self.filename)
        self.slider_values[0] = self.slider_red.get()
        self.slider_values[1] = self.slider_green.get()
        self.slider_values[2] = self.slider_blue.get()
        self.image = change_channels(self.image,
                                     self.slider_values[0],
                                     self.slider_values[1],
                                     self.slider_values[2])
        self.redraw()

    def redraw(self, grey=False):
        self.grey = grey
        if self.grey:
            self.a.imshow(self.image, cmap='gray')
        else:
            self.a.imshow(self.image)
        self.canvas.draw()


if __name__ == '__main__':
    display = Display('test2.jpg')
