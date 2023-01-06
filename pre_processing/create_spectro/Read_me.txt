Un readme rien que pour nous, pour savoir comment générer quoi.

Normalement, l'ensemble des chemins sont écrits de manière à ce que nos
deux ordinateurs soient capables de les lires.

Il faut que dans ce dossier trois choses soient impérativement présentes:
	- la structure précédente
	- Dataset_spectro qui contient Test, Train, Val, contenant Noisy, Clean chacun
	- un dossier info qui contient deux sous dossier TRAIN et VAL sur le même
	  niveau d'arborescence que Dataset et Dataset_spectro

MAIN CREATE SPECTRO
Il permet de générer les spectro à partir des audio, et de les sauvegarder ainsi que
Les phases, et les normalisations.
ATTENTION bien lire les indications en début de main pour savoir ce que tu souhaites enregistrer et donc quels paramètres mettre à TRUE avant de lancer le code.

MAIN_VERIF
Ne sert à rien d'autre que reconstruire le signal, retrouver le SNR, plot les spectrogrammes.
Cela permet de tester les sauvegardes, les ouvertures et les fonctions.
Tout est ok on retrouve bien un SNR=10dB YUPI.
Dans test on retrouve notamment des essais, tout est bien rangé par catégorie.

MAIN_ORACLE
Permet de générer les signaux appelés oracles.