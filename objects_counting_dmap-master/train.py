"""Main script used to train networks."""
import os

import numpy as np
import torch
from matplotlib import pyplot

from data_loader import H5Dataset
from looper import Looper
from model import UNet, FCRN_A
import get_data
import misc

RESULT_THRESHOLD = 0.001
PLATEAU_THRESHOLD = 50
TRAIN_FOLDER = '../mosaic/mosaic_002/img'

def train(network_architecture='UNet',
          learning_rate=1e-3,
          batch_size=32,
          horizontal_flip=0.5,
          vertical_flip=0.5,
          convolutions=3,
          plot=True,
          dataset_name='cell',
          unet_filters= 16):

    print(f'Training on: {TRAIN_FOLDER}')
    get_data.generate_cell_data(TRAIN_FOLDER)

    """Train chosen model on selected dataset."""
    # use GPU if avilable
    print('CUDA enabled: ' + str(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = {}  # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        # expected HDF5 files in dataset_name/(train | valid).h5
        data_path = os.path.join(dataset_name, f"{mode}.h5")
        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(data_path,
                                  horizontal_flip if mode == 'train' else 0,
                                  vertical_flip if mode == 'train' else 0)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=batch_size)

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == 'ucsd' else 3

    # initialize a model based on chosen network_architecture
    network = {
        'UNet': UNet,
        'FCRN_A': FCRN_A
    }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)
    network = torch.nn.DataParallel(network)

    # initialize loss, optimized and learning rate scheduler
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)

    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0])
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True)

    # current best results (lowest mean absolute error on validation set)
    current_best = np.inf
    res_diff = []
    best_epoch = 0
    epoch = 0
    print("\n", "-" * 80, "\n", sep='')  
    while epoch < 20 or (epoch+1-best_epoch < PLATEAU_THRESHOLD and misc.abs(misc.avg(res_diff[-10:])-misc.avg(res_diff[-11:-1])) > RESULT_THRESHOLD):
        print(f"Epoch {epoch + 1}\n")
        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()
            res_diff.append(result)

        
        # update checkpoint if new best is reached
        if misc.abs(result) < current_best:
            current_best = misc.abs(result)

            print(f"\n!!!New best result!!!")
            best_epoch = epoch + 1

        print(f'\nCurrent result: {result} - Best result: {current_best} - Best epoch: {best_epoch}')
        print(f'Result deviation from avg(10): {misc.abs(misc.avg(res_diff[-10:])-misc.avg(res_diff[-11:-1]))} - limit: {RESULT_THRESHOLD}') 
        print("\n", "-" * 80, "\n", sep='')        
        epoch += 1

    
    pthname = TRAIN_FOLDER.split('/')[len(TRAIN_FOLDER.split('/'))-2]
    pthpath = TRAIN_FOLDER.split(pthname)[0]
    torch.save(network.state_dict(), f'{pthpath}/{pthname}_cntrmodel_{current_best}_{best_epoch}.pth')
    print(f'{pthpath}/{pthname}_cntrmodel_{current_best}_{best_epoch}.pth')
    if epoch+1-best_epoch >= PLATEAU_THRESHOLD:
        print(f"[Training done - No new best result in {PLATEAU_THRESHOLD} epochs] Best result: {current_best}")
    else:
        print(f"[Training done - Plateau reached] Best result: {current_best}")


if __name__ == '__main__':
    for x in range(5):
        train()