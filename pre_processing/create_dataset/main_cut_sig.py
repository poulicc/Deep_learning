"""
    This code allows to cut the signals of the LIBRISPEECH library
    and save them in the TRAIN, VAL and TEST folders in a given distribution
    (preservation of the male/female equity in each of the folders).

    It is possible to check that the dynamics are preserved in the
    test part.

    It is possible to check the length of the cuted version in the
    test part.
"""

import os
import glob
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from cut_sig import tab_sig

SAVE_SIGNALS = False #FLAG save clean signals of T seconds in the folders TRAIN/TEST/VAL
TEST_DYNAM = False #FLAG plot the signals to check the dynamics
TEST_LENGTH = False #FLAG check the signals length

T = 4 #duration in seconds
"""
    If you want to create the clean signals of duration T (seconds),
    put SAVE_SIGNALS=True and run the code below
"""
if SAVE_SIGNALS:
    sample = tab_sig(4,fech=16000)

"""
    If you want to check that only the length of the signal has changed,
    put TEST_DYNAM = True and run this part of the code

    !! note that one signal is normalized but not the other !!

"""
if TEST_DYNAM:
    #Ref
    data = torchaudio.datasets.LIBRISPEECH(root='.', url = 'dev-clean')
    #Charge
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    #Données
    dataset=data_loader.dataset

    num_tested = 0 #cela correspond aux audios de la personne 1272, dossier 128104, [0]

    #Import the same audio but cut by the previous step
    path = os.path.join("..", "..", "network", "Dataset", "Train", "Clean")
    sig0 = os.path.join(path, "0.flac")
    cut0, _= sf.read(sig0)

    #Plot Signal 0
    plt.figure()
    plt.subplot(211)
    plt.plot(dataset[num_tested][0].numpy()[0])
    plt.title("Signal 0 not cut, length" + str(dataset[num_tested][0].numpy()[0].shape))
    plt.subplot(212)
    plt.plot(cut0)
    plt.title("Signal 0 cut, length" + str(cut0.shape))
    plt.show()

"""
    If you want to check that all the cut signals in the dataset are
    of the right size (here 4s), put TEST_LENGTH=True
"""
if TEST_LENGTH:
    #path of the data
    path_train = os.path.join("..", "..", "network", "Dataset", "Train", "Clean", "*.flac")
    path_val = os.path.join("..", "..", "network", "Dataset", "Val", "Clean", "*.flac")
    path_test = os.path.join("..", "..", "network", "Dataset", "Test", "Clean", "*.flac")

    #create a list of the path for all the data .flac in the given (TRAIN/VAL/TEST) folder
    train_names = glob.glob(path_train)
    val_names = glob.glob(path_train)
    test_names = glob.glob(path_train)

    #Check the length first for the training set
    for i in range(0, len(train_names)):
        cut, samplerate = sf.read(train_names[i]) #open file
        print(i/len(train_names)) #print the % of the achievement of the task
        if cut.shape[0] !=int(samplerate*T):
            print("Error, I don't have the right length !!!!! I measure " + str(cut.shape[0]))
            exit()

    #Check the length first for the validation set
    for i in range(0, len(val_names)):
        cut, samplerate= sf.read(val_names[i])#open file
        print(i/len(val_names)) #print the % of the achievement of the task
        if cut.shape[0] !=int(samplerate*T):
            print("Error, I don't have the right length !!!!! I measure " + str(cut.shape[0]))
            exit()

    #Check the length first for the test set
    for i in range(0, len(test_names)):
        cut, samplerate= sf.read(test_names[i])#open file
        print(i/len(test_names)) #print the % of the achievement of the task
        if cut.shape[0] !=int(samplerate*T):
            print("Error, I don't have the right length !!!!! I measure " + str(cut.shape[0]))
            exit()
