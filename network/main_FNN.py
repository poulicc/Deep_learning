"""
    This code is the one that allows the training of the FNN network.
    There are hyper-parameters to set which are as follows:
        - BATCH_SIZE
        - NBSAMPLESTRAIN
        - NBSAMPLESVAL
        - MAX_EPOCH
        - LR
        - SPECTRO_DB
    There is also the name of the backup file to be filled in.
"""

import time
import os
import sys

sys.path.append(os.path.dirname(__file__))
CURDIRPATH = os.path.dirname(__file__)
DATALOADERPATH = os.path.join(CURDIRPATH, "Dataloader", "FNN")
sys.path.append(DATALOADERPATH)

import torch as th
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch.nn as nn
from AUDIODataloader import DataModuleClass
import numpy as np
from torch.utils.data import DataLoader

# Create a dataloader from the subset as usual
start=time.time()

#Hyper parameters
BATCH_SIZE = 25#int(sys.argv[1])#251
NBSAMPLESTRAIN= 1098
NBSAMPLESVAL= 343
MAX_EPOCH =60#int(sys.argv[2])#2
LR= 0.001 #float(sys.argv[3])#0.001
in_db_=False
name_file = "test_60epoch.pt" #sys.argv[4]
nb_ligne=251
nb_sample=NBSAMPLESTRAIN+NBSAMPLESVAL
nb_batch=int(nb_ligne*nb_sample//BATCH_SIZE*0.80)

#import des data
column_dataloader = DataModuleClass(NBSAMPLESTRAIN, NBSAMPLESVAL, in_db=in_db_)
column_dataloader.prepare_data()

#formatage des data
train_dataloader = DataLoader(column_dataloader.column_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
val_dataloader = DataLoader(column_dataloader.column_val, batch_size=BATCH_SIZE, shuffle=False,num_workers=0,pin_memory=True)

#Déclaration des fonctions loss et optimizer
my_loss = nn.MSELoss()
val_my_loss = nn.MSELoss()

#Définition du réseau
net = nn.Sequential(
nn.Linear(257, 257),
nn.ReLU(),
nn.Linear(257, 257),
nn.ReLU(),
nn.Linear(257, 257),
)

optimizer = th.optim.Adam(net.parameters(), lr=LR)

# Décris la boucle d'apprentissage pendant l'entrainement avec MAX_EPOCH le nombre max d'epoch à réaliser, et nb_batch le nombre de batch de taille BATCH_SIZE
loss_epoch_train=[]
loss_epoch_val=[]
for id_epoch in range(MAX_EPOCH):
    tab_loss=[]
    tab_val=[]
    for batch_ndx, sample in enumerate(train_dataloader):
        out = net(sample[0])
        #Compare le resultat du réseau aux données clean
        # noisy=np.transpose(sample[0].detach().numpy())
        # clean=np.transpose(sample[1].detach().numpy())
        # plt.figure()
        #
        # plt.subplot(121)
        # plt.pcolormesh(noisy)
        # plt.title("input")
        #
        # plt.subplot(122)
        # plt.pcolormesh(clean)
        # plt.title("clean")
        # plt.show()
        #
        # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        # D = librosa.amplitude_to_db(noisy, ref=np.max)
        # img=librosa.display.specshow(D, y_axis='linear', x_axis='time',
        #                                sr=16e3, ax=ax[0])
        # ax[0].set(title='Linear-frequency power spectrogram')
        # ax[0].label_outer()
        #
        # hop_length = 1024
        # D = librosa.amplitude_to_db(clean,ref=np.max)
        # img=librosa.display.specshow(D, y_axis='linear', x_axis='time',
        #                                sr=16e3, ax=ax[1])
        # ax[1].set(title='Linear-frequency power spectrogram')
        # ax[1].label_outer()
        # fig.colorbar(img, ax=ax, format="%+2.f dB")
        # plt.show()
        # exit()
        # plt.subplot(133)
        # plt.pcolormesh(out.detach().numpy())
        # plt.title("out")

        loss = my_loss(out, sample[1])

        #Gradiant
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tab_loss.append(loss.data)

        print("epoch : "+str(id_epoch+1)+"/"+str(MAX_EPOCH))
        print("Chargement : "+str(int(batch_ndx/nb_batch*100.))+"%")
    loss_epoch_train.append(sum(tab_loss)/len(tab_loss))

    for batch_ndx, sample in enumerate(val_dataloader):
        # Forward Pass
        out_val = net(sample[0])
        # Find the Loss
        loss_val = val_my_loss(out_val, sample[1])
        # Calculate Loss
        tab_val.append(loss_val.data)

        print("Epoch"+str(id_epoch+1)+ "\t\t Validation Loss:"+
        str(loss_val.data / len(val_dataloader)))
    loss_epoch_val.append(sum(tab_val)/len(tab_val))

# Save weights
SAVEDIR=os.path.join(CURDIRPATH, "model_FNN", name_file)
th.save(net, name_file)

interval=time.time()-start
print("Il a fallu : ",interval, "secondes")

#Plot loss
plt.figure()
plt.plot(loss_epoch_train,label='Training')
plt.plot(loss_epoch_val,label='Validation')
plt.legend()
plt.xlabel('Batch itération')
plt.ylabel('MSE')
# plt.savefig("plot/modele_"+sys.argv[5])
plt.show()
