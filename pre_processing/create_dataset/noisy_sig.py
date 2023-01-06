"""
    This module contains functions which allow to generate datasets
    train, val, test of noisy signals.

    This file contains the following functions:
        - create_noisy_sig :
                    For a given noise and a given clean signal, creation of a
                    noisy signal with a chosen SNR in dB
        - single_noisy_sig :
                    Creation of a noisy signal by the create_noisy_sig function
                    with a choice of noise randomly set by a
                    random function

"""

import numpy as np
import soundfile as sf

def create_noisy_sig(clean, noise, RSB_db):
    """
        Inputs
            - clean : tensor whose shape is of type 'numpy([number of points])
            - noise : tensor whose shape is of type 'numpy([number of points])'
            NOTE the size of the two signals is the same
            - RSB_db : float signal to noise ratio that the final signal must have

        Output
            -signal : array whose shape is of type 'numpy([number of points])
                            which is the noisy signal
    """

    if clean.shape != noise.shape :
        raise Exception("The two signals are not the same size")
        exit()

    #signal normalization (clean already done but you never know)
    clean = clean/np.max(clean)
    noise = noise/np.max(noise)

    #calculation of signal strengths
    Pc = np.sum(clean**2)
    Pn = np.sum(noise**2)

    #calculation of the linear SNR
    RSB_signal = 10**(RSB_db/10)

    #calculation of the factor of s = signal + alpha noise
    frac_power = Pc/Pn
    alpha = np.sqrt(frac_power*(1/RSB_signal))

    #create the noise signal
    signal = clean + alpha*noise

    return signal

def single_noisy_sig(namefile, RSB_db, tab_noise, random_function='unif'):
    """
        Inputs
            - namefile : STR path of the clean signal to be modify with noise
            - random_function : STR as the noise is chosen in a list of 1711 noise
              it's the function to choose one of them in the list
            - RSB_db : float signal to noise ratio that the final signal must have

        Output
            - signal : array whose shape is of type 'numpy([number of points])
                            which is the noisy signal
    """
    #import clean signal
    clean, _ = sf.read(namefile)

    #import noise
    if random_function=='unif':
        n_noise = tab_noise.shape[0]
        randm = np.random.randint(0, n_noise)
        noise = tab_noise[randm, :]
    else :
        raise Exception("The function to randomly select a noise does not exist.")
        exit()

    #Create the new noisy signal
    signal = create_noisy_sig(clean, noise, RSB_db)

    return signal
