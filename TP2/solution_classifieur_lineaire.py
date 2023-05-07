# -*- coding: utf-8 -*-

#####

# CHAIMAE TOUBALI 
# CIP : 22 142 450
# Corentin POMMELEC
# CIP : pomc0601
# Tahar AMAIRI
# CIP : amat0601

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.])  # paramètre aléatoire
        self.w_0 = -5.  # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            # AJOUTER CODE ICI
            N = len(t_train)
            N1 = np.sum(t_train)
            N2 = N - N1
            p = N1 / N
            C1_indexs = np.where(t_train == 1)[0]
            C2_indexs = np.where(t_train == 0)[0]
            mu_1 = np.array([np.mean(x_train[C1_indexs, 0]), np.mean(x_train[C1_indexs, 1])])
            mu_2 = np.array([np.mean(x_train[C2_indexs, 0]), np.mean(x_train[C2_indexs, 1])])
            S1 = np.cov(x_train[C1_indexs].T)
            S2 = np.cov(x_train[C2_indexs].T)
            S1 = S1 * (N1 - 1) / N1
            S2 = S2 * (N2 - 1) / N2
            sigma = (N1 * S1 + N2 * S2) / N
            sigma = sigma + self.lamb * np.identity(2)
            self.w = np.linalg.solve(sigma, (mu_1 - mu_2))
            self.w_0 = -np.dot(mu_1, np.linalg.solve(sigma, mu_1)) / 2 + np.dot(mu_2, np.linalg.solve(sigma, mu_2)) / 2 + np.log( p / (1 - p))


        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            # AJOUTER CODE ICI
            nb_iterations_max = 1000
            ifError = False
            eta = 0.001

            T = np.select([t_train == 0.0], [-1], t_train)

            for k in range(0,nb_iterations_max):
                for n in range(len(x_train)):
                    if(self.erreur(t_train[n], self.prediction(x_train[n]))):
                        self.w += eta * (x_train[n] * T[n] + self.lamb * self.w)
                        self.w_0 += eta * (T[n] + self.lamb * self.w_0)
                        ifError = True

                if not ifError:
                    break
                else:
                    ifError = False
                    
        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            # AJOUTER CODE ICI
            clf = Perceptron(penalty='l2', alpha=self.lamb, eta0=0.001, max_iter=1000)
            clf.fit(x_train, t_train)
            self.w = clf.coef_[0]
            self.w_0 = clf.intercept_[0]

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        # AJOUTER CODE ICI
        y = self.w_0 + np.vdot(self.w, x[:len(self.w)])
        if y >= 0:
            return 1
        else:
            return 0

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        # AJOUTER CODE ICI
        if t == prediction:
            return 0
        else:
            return 1

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
        