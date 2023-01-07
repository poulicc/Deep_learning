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
        self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), output_size=th.Size([20, 16, 61, 59]))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 6, 5)
        self.conv4 = nn.Conv2d(6, 1, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        # x = th.flatten(x, 1) # flatten all dimensions except batch
        x = self.unpool(F.relu(self.conv3(x)), output_size=th.Size([20, 16, 61, 59]))
        print(x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        return x

#Hyper parameters
BATCH_SIZE = 20#int(sys.argv[1])#251
NBSAMPLESTRAIN= 753
NBSAMPLESVAL= 343
MAX_EPOCH = 4#int(sys.argv[2])#2
LR= 0.001 #float(sys.argv[3])#0.001
nb_ligne=251
name_file = "test_1CNN.pt"
name_file_plot= "test_1CNN"
nb_sample=NBSAMPLESTRAIN+NBSAMPLESVAL
nb_batch=int(nb_ligne*nb_sample//BATCH_SIZE*0.80)

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
loss_epoch_train=[]
loss_epoch_val=[]
for id_epoch in range(MAX_EPOCH):
    tab_loss=[]
    tab_val=[]
    for batch_ndx, sample in enumerate(train_dataloader):
        out = net(sample[0])
        print(sample[0].shape, out.shape, sample[1].shape)
        exit()
        #Compare le resultat du réseau aux données clean
        loss = my_loss(out, sample[1])

        #Gradiant
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tab_loss.append(loss.data)

        print("epoch : "+str(id_epoch)+"/"+str(MAX_EPOCH))
        print("Chargement : "+str(int(batch_ndx/nb_batch*100.))+"%")
    loss_epoch_train.append(sum(tab_loss)/len(tab_loss))

    for batch_ndx, sample in enumerate(val_dataloader):
        # Forward Pass
        out_val = net(sample[0])
        # Find the Loss
        loss_val = val_my_loss(out_val, sample[1])
        # Calculate Loss
        tab_val.append(loss_val.data)

        print("Epoch"+str(id_epoch)+ "\t\t Validation Loss:"+
        str(loss_val.data / len(val_dataloader)))
    loss_epoch_val.append(sum(tab_val)/len(tab_val))

SAVEDIR=os.path.join("model_CNN", name_file)
PLOTDIR=os.path.join("plot_CNN", name_file_plot)
th.save(net, SAVEDIR)

interval=time.time()-start


print("Il a fallu : ",interval, "secondes")

#Plot loss
plt.figure()
plt.plot(loss_epoch_train,label='Training')
plt.plot(loss_epoch_val,label='Validation')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig(PLOTDIR+".png")
