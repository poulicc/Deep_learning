import os
import sys
import glob
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(__file__))
CURDIRPATH=os.path.dirname(__file__)
DATADIRPATH=os.path.join(CURDIRPATH,"..","..", "Dataset_spectro")

DATAPATH="Noisy/*.npy"
CLEANPATH="Clean/*.npy"

class MyDataset(Dataset):
    def __init__(self, data_type, nbSamples, transform=None):
        data_path=os.path.join(DATADIRPATH,data_type,DATAPATH)
        target_path=os.path.join(DATADIRPATH,data_type,CLEANPATH)
        self.data=sorted(glob.glob(data_path))
        self.target=sorted(glob.glob(target_path))
        self.nbSamples=nbSamples
        self.tensor=ToTensor()
        self.transform = transform

    def __getitem__(self, index):
        #Loading of spectro noisy and targets
        noisy_spectro, clean_spectro = np.load(self.data[index]), np.load(self.target[index])
        #Convert it into tensor
        noisy_spectro, clean_spectro = self.tensor(noisy_spectro).float(), self.tensor(clean_spectro).float()

        if self.transform:
            noisy_spectro = self.transform(noisy_spectro)

        return noisy_spectro, clean_spectro

    def __len__(self):
        return self.nbSamples
