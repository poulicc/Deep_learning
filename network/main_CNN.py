import time
import os
import sys

sys.path.append(os.path.dirname(__file__))
CURDIRPATH = os.path.dirname(__file__)
DATALOADERPATH = os.path.join(CURDIRPATH, "Dataloader", "CNN")
sys.path.append(DATALOADERPATH)

import torch as th
import matplotlib.pyplot as plt
import torch.nn as nn
from AUDIODataloader import DataModuleClass
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 6, 5)
        self.conv4 = nn.Conv2d(6, 1, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = th.flatten(x, 1) # flatten all dimensions except batch
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        return x

#Hyper parameters
BATCH_SIZE = 30#int(sys.argv[1])#251
NBSAMPLESTRAIN= 753
NBSAMPLESVAL= 343
MAX_EPOCH = 4#int(sys.argv[2])#2
LR= 0.001 #float(sys.argv[3])#0.001
nb_ligne=251

#import des data
column_dataloader = DataModuleClass(NBSAMPLESTRAIN, NBSAMPLESVAL)
column_dataloader.prepare_data()

train_dataloader = DataLoader(column_dataloader.spectro_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
val_dataloader = DataLoader(column_dataloader.spectro_val, batch_size=BATCH_SIZE, shuffle=False,num_workers=0,pin_memory=True)

net = Net()
my_loss = nn.MSELoss()
val_my_loss = nn.MSELoss()

optimizer = th.optim.Adam(net.parameters(), lr=LR)

# Décris la boucle d'apprentissage pendant l'entrainement avec MAX_EPOCH le nombre max d'epoch à réaliser, et nb_batch le nombre de batch de taille BATCH_SIZE
tab_loss=[]
tab_val=[]
for id_epoch in range(MAX_EPOCH):
    for batch_ndx, sample in enumerate(train_dataloader):
        out = net(sample[0])
        #Compare le resultat du réseau aux données clean
        loss = my_loss(out, clean)

        #Gradiant
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tab_loss.append(loss.data)

        # out_val=net(val_dataset[:][0])#noisy
        # loss_val=val_my_loss(out_val, val_dataset[:][1])#clean
        # tab_val.append(loss_val.data)

        print("epoch : "+str(id_epoch)+"/"+str(MAX_EPOCH))
        print("Chargement : "+str(int(batch_ndx/nb_batch*100.))+"%")

    for batch_ndx, sample in enumerate(val_dataloader):
        # Forward Pass
        out_val = net(sample[0])
        # Find the Loss
        loss_val = val_my_loss(out_val, sample[1])
        # Calculate Loss
        tab_val.append(loss_val.data)

        print("Epoch"+str(id_epoch)+ "\t\t Validation Loss:"+
        str(loss_val.data / len(val_dataloader)))


th.save(net, "test_modeleShuffle3.pt")

interval=time.time()-start


print("Il a fallu : ",interval, "secondes")

plt.figure()
plt.plot(tab_loss,label='Training')
plt.plot(tab_val,label='Validation')
plt.legend()
plt.xlabel('Batch itération')
plt.ylabel('MSE')
# plt.savefig("plot/modele_"+sys.argv[5])
plt.show()
