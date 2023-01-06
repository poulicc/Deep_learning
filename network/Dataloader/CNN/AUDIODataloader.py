"""
    Ce dataloader permet de préparer les données pour un réseau de type CNN.
    Un élément est donc un spectrogramme.
"""

import os
from torch import manual_seed
import pytorch_lightning as pl
from MyDataset import MyDataset

class DataModuleClass(pl.LightningDataModule):
    def __init__(self, nbSamplesTrain,nbSamplesVal):
        super().__init__()
        self.nbSamplesTrain=nbSamplesTrain
        self.nbSamplesVal=nbSamplesVal

    def prepare_data(self):
        self.spectro_train=MyDataset("Train",self.nbSamplesTrain)
        self.spectro_val=MyDataset("Val",self.nbSamplesVal)

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
    nbSamplesTrain=1000
    nbSamplesVal=200
    data_CNN=DataModuleClass(nbSamplesTrain,nbSamplesVal)
    data_CNN.prepare_data()

    num=10
    noisy_spectro_num, clean_spectro_num=data_CNN.spectro_train[num]

    #in numpy
    noisy_spectro_num, clean_spectro_num=noisy_spectro_num.numpy(), clean_spectro_num.numpy()
    print("Dans le test : la taille du spectro apres convertion", noisy_spectro_num.shape)

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

    if CROISE:
        TRAINNOISEPATH=os.path.join("..", "..", "Dataset_spectro","Train","Noisy","*.npy") #noisy spectro train
        TRAINCLEANPATH=os.path.join("..", "..", "Dataset_spectro", "Train","Clean","*.npy") #clean spectro train

        train_noise_paths = sorted(glob.glob(TRAINNOISEPATH))
        train_clean_paths = sorted(glob.glob(TRAINCLEANPATH))

        noisy_spectro=np.load(train_noise_paths[num])
        clean_spectro=np.load(train_clean_paths[num])

        plt.figure()
        plt.subplot(121)
        plt.pcolormesh(noisy_spectro)
        plt.colorbar()
        plt.title("Spectrogram of new noisy signal CROISE")

        plt.subplot(122)
        plt.pcolormesh(clean_spectro)
        plt.colorbar()
        plt.title("Spectrogram of the clean signal CROISE")
        plt.show()
