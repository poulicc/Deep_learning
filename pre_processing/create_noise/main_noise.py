"""
    This module allows you to create a file of several
    duration extracted from a larger signal.
   
"""
import os
from scipy.io import wavfile
from noise import cut_noise_sig, new_additiv_noise
import numpy as np

duration = 4 # 4s
new_freq = 16000 # new frequency of the signal
_, noise = wavfile.read("babble_ech.wav")

len_noise = len(noise)
len_sig_ech = new_freq*duration

cut_noise, fac = cut_noise_sig(noise, len_noise)
tab_noises = new_additiv_noise(cut_noise, fac, len_sig_ech)

save = True
if save:
    path = os.path.join("..", "create_dataset", 'noise_cut_4s.npy')
    np.save(path, tab_noises)
