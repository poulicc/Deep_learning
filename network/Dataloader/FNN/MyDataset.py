import os
import sys
import glob
import numpy as np
import torch as th
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(__file__))
CURDIRPATH=os.path.dirname(__file__)
DATADIRPATH=os.path.join(CURDIRPATH,"..","..", "Dataset")

DATAPATH="Noisy/X_all.npy"
CLEANPATH="Clean/X_clean_all.npy"

DATAPATHDB="Noisy/X_all_dB.npy"
CLEANPATHDB="Clean/X_clean_all_dB.npy"

class MyDataset(Dataset):
    def __init__(self, data_type, nbSamples, in_db):
        if in_db:
            data_path=os.path.join(DATADIRPATH,data_type,DATAPATHDB)
            target_path=os.path.join(DATADIRPATH,data_type,CLEANPATHDB)
        else:
            data_path=os.path.join(DATADIRPATH,data_type,DATAPATH)
            target_path=os.path.join(DATADIRPATH,data_type,CLEANPATH)
        self.data=np.load(data_path)
        self.target=np.load(target_path)
        self.nbSamples=nbSamples
        self.tensor=ToTensor()
        self.in_db=in_db

    def __getitem__(self, index):
        #Loading spectro noisy and target
        noisy_column, clean_column = self.data[index], self.target[index]
        #Convert it into tensor
        noisy_column, clean_column = self.tensor(noisy_column.reshape(noisy_column.shape[0], 1)), self.tensor(clean_column.reshape(clean_column.shape[0], 1))
        return noisy_column.squeeze(0).squeeze(-1).float(), clean_column.squeeze(0).squeeze(-1).float()

    def __len__(self):
        return self.nbSamples*251
