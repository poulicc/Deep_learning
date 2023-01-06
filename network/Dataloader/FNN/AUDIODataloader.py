"""
    Data loader.
    Combines a dataset and a sampler, and provides an iterable over the given
    dataset.
"""

import sys
import os
import numpy as np
import torch
from torch import manual_seed
import pytorch_lightning as pl
from torchvision.transforms import Compose
from MyDataset import MyDataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

class DataModuleClass(pl.LightningDataModule):
    def __init__(self, nbSamplesTrain,nbSamplesVal, in_db):
        super().__init__()
        self.nbSamplesTrain=nbSamplesTrain
        self.nbSamplesVal=nbSamplesVal
        self.train_size = self.nbSamplesTrain*251
        self.val_size = self.nbSamplesVal*251
        self.in_db=in_db

    def prepare_data(self):
        self.column_train=MyDataset("Train",self.nbSamplesTrain, self.in_db)
        self.column_val=MyDataset("Val",self.nbSamplesVal, self.in_db)

    def setup(self):
        manual_seed(42)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data

VERIF=False
CROISE=False

if VERIF:
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    #H-Params
    nbSamplesTrain=1098
    nbSamplesVal=343
    data_CNN=DataModuleClass(nbSamplesTrain,nbSamplesVal, in_db=True)
    data_CNN.prepare_data()
    train_dataloader = DataLoader(data_CNN.column_train, batch_size=251, shuffle=False,num_workers=0)

    k=3
    num=251*k
    num_1=251*(k+1)

    for batch_ndx, sample in enumerate(train_dataloader):
        if batch_ndx==k:
            noisy_spectro_num, clean_spectro_num=sample[0], sample[1]
            print("Dans le test : la taille du spectro apres convertion", noisy_spectro_num.shape)
            #in numpy
            noisy_spectro_num, clean_spectro_num=noisy_spectro_num.numpy(), clean_spectro_num.numpy()
            print("Dans le test : la taille du spectro apres convertion", noisy_spectro_num.shape)
            #transposition
            noisy_spectro_num, clean_spectro_num = np.transpose(noisy_spectro_num), np.transpose(clean_spectro_num)
            #Affichage des spectro
            plt.figure()
            plt.subplot(121)
            plt.pcolormesh(noisy_spectro_num)
            plt.colorbar()
            plt.title("Spectrogram of the noisy signal")
            plt.subplot(122)
            plt.pcolormesh(clean_spectro_num)
            plt.colorbar()
            plt.title("Spectrogram of the clean one")
            plt.show()
            break

    if CROISE:
        TRAINNOISEPATH=os.path.join("..", "..", "Dataset","Train","Noisy","X_all.npy") #noisy spectro train
        TRAINCLEANPATH=os.path.join("..", "..", "Dataset", "Train","Clean","X_clean_all.npy") #clean spectro train

        train_noise = np.load(TRAINNOISEPATH)
        train_clean = np.load(TRAINCLEANPATH)

        noisy_spectro= train_noise[num:num_1]
        clean_spectro= train_clean[num:num_1]

        plt.figure()
        plt.subplot(121)
        plt.pcolormesh(np.transpose(20*np.log10(noisy_spectro)))
        plt.colorbar()
        plt.title("Spectrogram of new noisy signal CROISE")

        plt.subplot(122)
        plt.pcolormesh(np.transpose(20*np.log10(clean_spectro)))
        plt.colorbar()
        plt.title("Spectrogram of the clean signal CROISE")
        plt.show()
