

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

plt.figure(figsize=(9,6))

def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def equalize_this(image_file, with_plot=False, gray_scale=False, bins=256):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    if not gray_scale:
        r_image = image_src[:, :, 0]
        g_image = image_src[:, :, 1]
        b_image = image_src[:, :, 2]

        r_image_eq = enhance_contrast(image_matrix=g_image)
        g_image_eq = enhance_contrast(image_matrix=g_image)
        b_image_eq = enhance_contrast(image_matrix=g_image)

        image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
        cmap_val = 'Greens'
    else:
        image_eq = enhance_contrast(image_matrix=image_src)
        cmap_val = 'Greens'

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.axis("off")
    ax2.imshow(image_eq, cmap=cmap_val)
    fig.savefig('out\\enh\\'+image_file.split('\\')[1].split('.')[0] + '_enh.png')
    plt.close()

for filename in os.listdir('out'):
    if (filename != 'enh'):
        equalize_this('out\\'+filename)