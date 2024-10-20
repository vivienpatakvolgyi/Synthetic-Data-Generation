"""This script apply a chosen model on a given image.

One needs to choose a network architecture and provide the corresponding
state dictionary.

Example:

    $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg

The script also allows to visualize the results by drawing a resulting
density map on the input image.

Example:

    $ $ python infer.py -n UNet -c mall_UNet.pth -i seq_000001.jpg --visualize

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib.figure import figaspect

from model import UNet, FCRN_A


def infer(
        image_path='C:\\Users\\Tomi\\PycharmProjects\\DCGAN\\mosaic\\mosaic_001\\img\\001_00039_01906_img.png', #801
        network_architecture='UNet',
        checkpoint='cell_UNet.pth',
        convolutions=2,
        one_channel=False,
        pad=False,
        unet_filters=16,
        visualize=True):
    """Run inference for a single image."""
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if one_channel else 3

    # initialize a model based on chosen network_architecture
    network = {
        'UNet': UNet,
        'FCRN_A': FCRN_A
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)

    # load provided state dictionary
    # note: by default train.py saves the model in data parallel mode
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(checkpoint))
    network.eval()

    img = Image.open(image_path)

    # padding was applied for ucsd images to allow down and upsampling
    if pad:
        img = Image.fromarray(np.pad(img, 1, 'constant', constant_values=0))

    # network's output represents a density map
    density_map = network(TF.to_tensor(img).unsqueeze_(0))

    # note: density maps were normalized to 100 * no. of objects
    n_objects = torch.sum(density_map).item() / 100


    if visualize:
        _visualize(img, density_map.squeeze().cpu().detach().numpy())
    return n_objects

def _visualize(img, dmap):
    """Draw a density map onto the image."""
    # keep the same aspect ratio as an input image
    fig, ax = plt.subplots(figsize=figaspect(1.0 * img.size[1] / img.size[0]))
    fig.subplots_adjust(0, 0, 1, 1)

    # plot a density map without axis
    ax.imshow(dmap, cmap="hot")
    plt.axis('off')
    fig.canvas.draw()

    # create a PIL image from a matplotlib figure
    dmap = Image.frombytes('RGB',
                           fig.canvas.get_width_height(),
                           fig.canvas.tostring_rgb())

    # add a alpha channel proportional to a density map value
    dmap.putalpha(dmap.convert('L'))

    # display an image with density map put on top of it
    Image.alpha_composite(img.convert('RGBA'), dmap.resize(img.size)).show()


if __name__ == "__main__":
    dir = 'irl'
    for filename in os.listdir(dir):
        if 'img' in filename:
            print(f'{filename} -> {infer(image_path=dir+ "/" + filename, visualize=False)}')
