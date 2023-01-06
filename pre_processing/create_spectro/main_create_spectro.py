"""
    This file allows to create the dataset (TRAIN/VAL) for the two types of networks (FNN/CNN)

    For the CNN network as for the FNN network :
        - the phases of the noisy and non-noisy signals are stored in dimensional tables
                                                                    (number of TRAIN/VAL files, 257,251)
        There are thus at the end of this code four .npy files of the phases which are saved in the
        info/Train" and "info/Val" files under the name of X_phase_all and X_clean_phase_all.

        - the indexes of the signals are saved in tables of dimensions
                                                                    (number of TRAIN/VAL files, name file)
        At the end of this code there are two .npy files of the indices which are saved in the
        info/Train" and "info/Val" files under the name of index_all.

        - the min/max values of the normalizations are saved in a list: there are two lists, one
            for train and one for val.
        At the end of this code there are two .npy files of the normalizations which are saved in the
        folders "info/Train" and "info/Val" under the name of normalize_info_all.

    For the CNN network:
        The spectrograms are saved one by one in two folders and 4 subfolders in .npy:
                - Dataset_spectro/TRAIN -> CLEAN and NOISY
                - Dataset_spectro/VAL -> CLEAN and NOISY
    For the FNN :
        Spectrograms are stored column by column in the same .npy file:
                - Dataset/TRAIN -> CLEAN and NOISY
                - Dataset/VAL -> CLEAN and NOISY
"""
import sys
import os
import glob
import re
import numpy as np
import soundfile as sf
import librosa.display
import torch as th
from spectro_norm import *
from column_spectro import fill_tab_column

# Global variables : choose what you want to do (set to TRUE)
SAVE_INDEX=False # save the index (correspondence between file name -number- and its number in the complete list)
SAVE_VAL=True #save the validation informations (NOISY and CLEAN)
SAVE_TRAIN=True #save the training informations (NOISY and CLEAN)

#Definitions of .flac data paths
sys.path.append(os.path.dirname(__file__))
CURDIRPATH=os.path.dirname(__file__)

TRAINNOISEPATH=os.path.join("..", "..", "network", "Dataset","Train","Noisy","*.flac") #noisy signals train
TRAINCLEANPATH=os.path.join("..", "..", "network", "Dataset", "Train","Clean","*.flac") #clean signals train

VALNOISEPATH=os.path.join("..", "..", "network", "Dataset","Val","Noisy","*.flac")#noisy signals val
VALCLEANPATH=os.path.join("..", "..", "network", "Dataset", "Val","Clean","*.flac")#clean signals val

#Number of signals in the study
NB_SAMPLES = "all" #if you want only a part, change it but don't forget to change the saving names of the files
NETWORK_TYPE = "FNN" #"CNN" or "FNN"

# Create a list of the signals paths (train noise / train clean / val noise / val clean)
if NB_SAMPLES == "all" : # we take all the signals
    train_noise_paths = sorted(glob.glob(TRAINNOISEPATH))
    train_clean_paths = sorted(glob.glob(TRAINCLEANPATH))

    val_noise_paths = sorted(glob.glob(VALNOISEPATH))
    val_clean_paths = sorted(glob.glob(VALCLEANPATH))

else: # we take onlys the first NB_SAMPLES signals
    train_noise_paths = sorted(glob.glob(TRAINNOISEPATH))[:NB_SAMPLES]
    train_clean_paths = sorted(glob.glob(TRAINCLEANPATH))[:NB_SAMPLES]

    val_noise_paths = sorted(glob.glob(VALNOISEPATH))[:NB_SAMPLES]
    val_clean_paths = sorted(glob.glob(VALCLEANPATH))[:NB_SAMPLES]

#Creation of data tables as explained at the beginning of the file
train_noise_tab_phase = np.zeros((len(train_noise_paths), 257, 251)) #phase train noisy
train_clean_tab_phase = np.zeros((len(train_noise_paths), 257, 251)) #phase train clean
train_tab_normalize_info=[] #normalization factors
train_index=np.zeros((len(train_noise_paths), 2))

val_noise_tab_phase = np.zeros((len(val_noise_paths), 257, 251))#phase val noisy
val_clean_tab_phase = np.zeros((len(val_noise_paths), 257, 251))#phase val clean
val_tab_normalize_info=[]#normalization factors
val_index=np.zeros((len(val_noise_paths), 2))

#If the network is FNN we have to create 4 news tables so that the spectrograms
#can be saved into columns
if NETWORK_TYPE=="FNN":
    train_noise_tab = np.zeros((len(train_noise_paths), 257, 251))
    train_clean_tab = np.zeros((len(train_noise_paths), 257, 251))

    val_noise_tab = np.zeros((len(val_noise_paths), 257, 251))
    val_clean_tab = np.zeros((len(val_noise_paths), 257, 251))

#Calculations of spectrograms and filling of data tables TRAIN
for index in range (0, len(train_noise_paths)):
    #open and read the signals
    noisy, sr= sf.read(train_noise_paths[index])
    clean, sr = sf.read(train_clean_paths[index])

    #extraction of the modulus spectrogram, the phase, and the normalization info for the noisy and the clean
    noise_spectro, clean_spectro, phase_noise, phase_clean,normalize_info = sig_to_spectro(noisy, clean)

    #fill the correct tab
    train_noise_tab_phase[index, :, :] = phase_noise
    train_clean_tab_phase[index, :, :] = phase_clean
    train_tab_normalize_info.append(normalize_info)

    #associate the index with the real name of the file
    list = [int(s) for s in re.findall(r'-?\d+?', train_noise_paths[index])] # take all numbers before .flac in the path in a list
    list = ''.join(map(str, list)) #extract the number of the list

    if SAVE_INDEX:
        train_index[index,0] = index
        train_index[index,1] = list

    if SAVE_TRAIN:
        SAVENOISEPATH=os.path.join("..", "..", "network", "Dataset_spectro","Train","Noisy")
        SAVECLEANPATH=os.path.join("..", "..", "network", "Dataset_spectro", "Train","Clean")
        namefile=str(list)

        #If the network is a CNN, we save each spectrogram
        if NETWORK_TYPE=="CNN":
            np.save(os.path.join(SAVENOISEPATH, namefile), noise_spectro)
            np.save(os.path.join(SAVECLEANPATH, namefile), clean_spectro)
        #If the network is a FNN, we stock each column spectrogram
        if NETWORK_TYPE=="FNN":
            train_noise_tab[index, :, :] = noise_spectro
            train_clean_tab[index, :, :] = clean_spectro


#Calculations of spectrograms and filling of data tables VAL
for index in range (0, len(val_noise_paths)):
    #open and read the signals
    noisy, _= sf.read(val_noise_paths[index])
    clean, _ = sf.read(val_clean_paths[index])

    #extraction of the modulus spectrogram, the phase, and the normalization info for the noisy and the clean
    noise_spectro, clean_spectro, phase_noise, phase_clean,normalize_info = sig_to_spectro(noisy, clean)

    #fill the correct tab
    val_noise_tab_phase[index, :, :] = phase_noise
    val_clean_tab_phase[index, :, :] = phase_clean
    val_tab_normalize_info.append(normalize_info)

    #associate the index with the real name of the file
    list = [int(s) for s in re.findall(r'-?\d+?', train_noise_paths[index])] # take all numbers before .flac in the path in a list
    list = ''.join(map(str, list)) #extract the number of the list

    if SAVE_INDEX:
        val_index[index,0] = index
        val_index[index,1] = list

    if SAVE_VAL:
        SAVENOISEPATH=os.path.join("..", "..", "network", "Dataset_spectro","Val","Noisy")
        SAVECLEANPATH=os.path.join("..", "..", "network", "Dataset_spectro", "Val","Clean")
        namefile=str(list)

        if NETWORK_TYPE=="CNN":
            np.save(os.path.join(SAVENOISEPATH, namefile), noise_spectro)
            np.save(os.path.join(SAVECLEANPATH, namefile), clean_spectro)
        if NETWORK_TYPE=="FNN":
            val_noise_tab[index, :, :] = noise_spectro
            val_clean_tab[index, :, :] = clean_spectro

print("Start to save files")
if NB_SAMPLES == "all" :
    #paths definitions for each tab
    DATADIRPATH=os.path.join(CURDIRPATH,"..","..","network", "info")

    TRAIN_NOISEPHASE=os.path.join(DATADIRPATH,"Train","X_phase_all")
    TRAIN_CLEANPHASE=os.path.join(DATADIRPATH,"Train","X_clean_phase_all")
    TRAIN_NORMA=os.path.join(DATADIRPATH,"Train","normalize_info_all")
    TRAIN_INDEX=os.path.join(DATADIRPATH,"Train","index_all")

    VAL_NOISEPHASE=os.path.join(DATADIRPATH,"Val","X_phase_all")
    VAL_CLEANPHASE=os.path.join(DATADIRPATH,"Val","X_clean_phase_all")
    VAL_NORMA=os.path.join(DATADIRPATH,"Val","normalize_info_all")
    VAL_INDEX=os.path.join(DATADIRPATH,"Val","index_all")

    if NETWORK_TYPE=="FNN":
        DATADIRPATH2=os.path.join(CURDIRPATH,"..","..","network", "Dataset")
        TRAIN_NOISEPATH=os.path.join(DATADIRPATH2,"Train", "Noisy","X_all")
        TRAIN_CLEANPATH=os.path.join(DATADIRPATH2,"Train","Clean","X_clean_all")
        VAL_NOISEPATH=os.path.join(DATADIRPATH2,"Val","Noisy","X_all")
        VAL_CLEANPATH=os.path.join(DATADIRPATH2,"Val","Clean","X_clean_all")

if SAVE_TRAIN:
    np.save(TRAIN_NOISEPHASE, train_noise_tab_phase)
    np.save(TRAIN_CLEANPHASE, train_clean_tab_phase)
    np.save(TRAIN_NORMA, train_tab_normalize_info)

if SAVE_VAL:
    np.save(VAL_NOISEPHASE, val_noise_tab_phase)
    np.save(VAL_CLEANPHASE, val_clean_tab_phase)
    np.save(VAL_NORMA, val_tab_normalize_info)

if SAVE_INDEX:
    np.save(TRAIN_INDEX, train_index)
    np.save(VAL_INDEX, val_index)
    np.savetxt(TRAIN_INDEX+'.txt', train_index)
    np.savetxt(VAL_INDEX+'.txt', val_index)

if NETWORK_TYPE=="FNN":
    X_train, X_clean_train = fill_tab_column(len(train_noise_paths), train_noise_tab, train_clean_tab)
    np.save(TRAIN_NOISEPATH, X_train)
    np.save(TRAIN_CLEANPATH, X_clean_train)

    X_val, X_clean_val = fill_tab_column(len(val_noise_paths), val_noise_tab, val_clean_tab)
    np.save(VAL_NOISEPATH, X_val)
    np.save(VAL_CLEANPATH, X_clean_val)
