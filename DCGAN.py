# -*- coding: utf-8 -*-
# %% pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import os
import shutil

import _model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import misc

# %%
#print('CUDA enabled: ' + str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_IMG = 3
Z_DIM = 100
FEATURES_GEN = 92
LEARNING_RATE_G = 2e-4  # could also use two lrs, one for gen and one for disc
LEARNING_RATE_D = 2e-5  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 512
IMAGE_SIZE = 64
NUM_EPOCHS = 31
FEATURES_DISC = 92
LOSS_THRESHOLD = 0.7

# %%
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder('./lum_crop_rot', transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
real_batch = next(iter(dataloader))
plt.figure(figsize=(3, 3))
plt.axis("off")
plt.imshow(
    np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=2, normalize=True, nrow=12).cpu(), (1, 2, 0)))


# >> uncomment if img preparation needed
# imgt.dcgan_prep()

# ----------------------------------------------------------------------------------------------------------------------

def deleteOutFolder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# %%
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


# %%
def show_tensor_images(type, img, epoch, image_tensor, num_images=6, size=(1, 64, 64)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], padding=2, normalize=True, nrow=6)
    plt.title("Epoch: " + str(epoch) + ' Image: ' + str(img) + ' - (' + type + ')')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    plt.savefig('out\img' + str(img) + '_epoch' + str(epoch) + '_' + type + '.png')


def trainModels():
    fixed_noise = torch.randn(512, Z_DIM, 1, 1).to(device)
    deleteOutFolder('out')
    gen.train()
    disc.train()
    epoch = 0
    loss_diff = []
    log = ""
    while epoch < 5 or misc.avg(loss_diff[-5:]) > LOSS_THRESHOLD:
        dirname = 'out\\result'
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            ### create noise tensor
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            ### Print losses occasionally and fake images occasionally
            if batch_idx % len(dataloader) == 0:
                loss_diff.append(abs(loss_gen - loss_disc))
                stat = f"Epoch [{epoch}] \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, loss diff: {loss_diff[epoch]:.4f}, loss diff avg: {misc.avg(loss_diff[-5:]):.4f}"
                print(stat)
                log = log + stat + '\n'

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    if epoch >= 5 and misc.avg(loss_diff[-5:]) < LOSS_THRESHOLD:
                        os.mkdir(dirname)
                        for imx in range(len(fake)):
                            torchvision.utils.save_image(fake[imx],
                                                         dirname + '\\' + str(epoch) + '_' + str(imx) + '.png')

                    for x in range(5):
                        y = (x + 1) * 12
                        img_grid_real = torchvision.utils.make_grid(real[y], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[y], normalize=True)
                        show_tensor_images('real', x, 'x', img_grid_real)
                        show_tensor_images('fake', x, epoch, img_grid_fake)
        epoch += 1
    with open("out\log.txt", "w") as text_file:
        text_file.write(log)
    torch.save(gen.state_dict(), 'out\gen.model')
    torch.save(disc.state_dict(), 'out\disc.model')


def useModels(count):
    # dirname = 'tmp'
    # deleteOutFolder(dirname)
    arr = []
    if not os.path.exists('out\gen.model'):
        trainModels()

    gen.load_state_dict(torch.load('out\gen.model'))
    gen.train(False)

    fixed_noise = torch.randn(max(1, min(512, count)), Z_DIM, 1, 1).to(device)
    with torch.no_grad():
        fake = gen(fixed_noise)

        for imx in range(len(fake)):
            # len(os.listdir(dirname))
            # torchvision.utils.save_image(fake[imx],
            #                             dirname + '\\' + str(
            #                                 len(os.listdir(dirname)) + 1) + '.png')

            tensor = fake[imx]
            grid = torchvision.utils.make_grid(fake[imx], nrow=1, padding=0, normalize=True)
            grid = torchvision.transforms.ToPILImage()(grid)
            arr.append(grid)
    return arr


# ----------------------------------------------------------------------------------------------------------------------


# %%
gen = _model.Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = _model.Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

# %%
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
criterion = nn.BCELoss()
