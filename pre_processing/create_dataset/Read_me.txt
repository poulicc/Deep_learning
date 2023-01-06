Un readme rien que pour nous, pour savoir comment générer quoi.

Normalement, l'ensemble des chemins sont écrits de manière à ce que nos
deux ordinateurs soient capables de les lires.

Il faut que dans ce dossier deux choses soient impérativement présentes:
  - 'noise_cut_4s.npy' : le tableau de bruits de cafet de 4s
  - LIBRISPEECH : le dossier avec l'ensemble des signaux

De plus, il faut que sur le même "étage" (dans les dossiers) que preprocessing
il y ait un dossier network. Ce dossier network est constitué d'un dossier
Dataset, dans lequel on trouve 3 dossiers TRAIN/VAL/TEST chacun constitués
de 2 dossiers : Noisy et Clean.

Pour créer les fichiers clean de 4s chacun, il faut lancer :
      main_cut_sig.py avec la variable globale SAVE_SIGNALS = True
Ainsi, les dossiers CLEAN sont remplis de signaux de 4s NORMALISÉS


Pour créer les fichiers noisy de 4s chacun, il faut lancer :
        main_noisy_sig.py avec la variable globale SAVE_TRAIN = True,
        SAVE_VAL = True et SAVE_TEST=True
Ainsi, les dossiers NOISY sont remplis de signaux de 4s.
Il est possible de visualiser les spectro en dB ou en linéaire et de
vérifier le RSB.
