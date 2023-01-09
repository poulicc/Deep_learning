"""
    This module contains the functions for creating noise signals

    The functions are the following:
        - cut_noise_sig : it cuts the initial signal into several signals of shorter length
        - new_additiv_noise : it adds the noises together to create more noises and diversity.
"""


import numpy as np

def cut_noise_sig(noise_sig, len_ech_reel_sig):
    """
    INPUT :
        - noise_sig : array original cafeteria noise
        - len_ech_reel_sig : INT length of this array

    OUTPUT :
        - cute_noise : array cropped noise_sig
        - fac : INT number of segment cute_noise we can create with noise_sig

    This function splits the original signal into small pieces of noise signals
    and stores them in an output array.

    """
    size_noise = len(noise_sig)
    fac = size_noise//len_ech_reel_sig # number of section

    cut_noise = np.zeros((fac, len_ech_reel_sig)) # output tab

    for k in range(0, fac):
        cut_noise[k, :] = noise_sig[len_ech_reel_sig*k:len_ech_reel_sig*(k+1)] # crop the signal
    return cut_noise, fac


def new_additiv_noise(cut_noise_init, fac, taille_sig_ech):
    """
    INPUT :
        - cut_noise_init : array cropped noise_sig
        - fac : INT number of segment cute_noise we can create with noise_sig
        - taille_sig_ech : INT size of the new signal

    OUTPUT :
        - new_noise : array cropped noise_sig which are add with noise
        to create more diversity of noise.

    This function creates noises, from the noises cut from the initial cafeteria
    noise signal to create more different noises.

    """
    sig_ref = cut_noise_init[0]
    cut_noise_crop = np.delete(cut_noise_init, (0), axis=0)
    add_new_element= np.zeros((fac-1, taille_sig_ech))
    for k in range(0, fac-1):
        add_new_element[k] = cut_noise_init[0]+cut_noise_crop[k]
    new_noise = np.concatenate((cut_noise_init, add_new_element))
    for i in range(1, fac-1):
        sig_ref = cut_noise_init[i]
        cut_noise_crop = np.delete(cut_noise_crop, (0), axis=0)
        add_new_element = np.zeros((fac-(i+1), taille_sig_ech))
        for k in range(0, cut_noise_crop.shape[0]):
            add_new_element[k] = cut_noise_init[i]+cut_noise_crop[k]
        new_noise = np.concatenate((new_noise, add_new_element))
    return new_noise
