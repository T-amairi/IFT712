# -*- coding: utf-8 -*-

# CHAIMAE TOUBALI 
# CIP : 22 142 450
# Corentin POMMELEC
# CIP : pomc0601
# Tahar AMAIRI
# CIP : amat0601

"""
Execution dans un terminal

Exemple:
   python non_lineaire_classification.py rbf 100 200 0 0
"""

import numpy as np
import sys
from map_noyau import MAPnoyau
import gestion_donnees as gd


def analyse_erreur(err_train, err_test):
    """
    Fonction qui affiche un WARNING lorsqu'il y a apparence de sur ou de sous
    apprentissage
    """
    difference = err_test - err_train
    seuil = 15
    if difference > seuil:
        print('WARNING : le modèle souffre de sur apprentissage')
    elif err_train > seuil and err_test > seuil:
        print('WARNING : le modèle souffre de sous apprentissage')

def main():

    if len(sys.argv) < 6:
        usage = "\n Usage: python non_lineaire_classification.py type_noyau nb_train nb_test lin validation\
        \n\n\t type_noyau: rbf, lineaire, polynomial, sigmoidal\
        \n\t nb_train, nb_test: nb de donnees d'entrainement et de test\
        \n\t lin : 0: donnees non lineairement separables, 1: donnees lineairement separable\
        \n\t validation: 0: pas de validation croisee,  1: validation croisee\n"
        print(usage)
        return

    type_noyau = sys.argv[1]
    nb_train = int(sys.argv[2])
    nb_test = int(sys.argv[3])
    lin_sep = int(sys.argv[4])
    vc = bool(int(sys.argv[5]))

    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees(nb_train, nb_test, lin_sep)
    [x_train, t_train, x_test, t_test] = generateur_donnees.generer_donnees()

    # On entraine le modèle
    mp = MAPnoyau(noyau=type_noyau)

    if vc is False:
        mp.entrainement(x_train, t_train)
    else:
        mp.validation_croisee(x_train, t_train)

    predTrain = np.array([mp.prediction(x_train[i,]) for i in range(x_train.shape[0])])
    predTest = np.array([mp.prediction(x_test[i,]) for i in range(x_test.shape[0])])

    err_train = np.mean(mp.erreur(predTrain, t_train)) * 100
    err_test = np.mean(mp.erreur(predTest, t_test)) * 100

    print('Erreur train = ', err_train, '%')
    print('Erreur test = ', err_test, '%')
    analyse_erreur(err_train, err_test)

    # Affichage
    mp.affichage(x_test, t_test)

if __name__ == "__main__":
    main()
