# -*- coding: utf-8 -*-

# Chaimae TOUBALI 
# CIP : touc1402

# Tahar AMAIRI
# CIP : amat0601

# Corentin POMMELEC
# CIP : pomc0601

from operator import length_hint
import numpy as np
import random
from sklearn import linear_model, model_selection

class Regression:
    def __init__(self, lamb, m=1, using_sklearn=False):
        self.lamb = lamb
        self.w = None
        self.M = m
        self.usingSkl = using_sklearn

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        isVector = isinstance(x,np.ndarray)

        # Cas de la régression linéaire
        if self.M == 1:
            return x.reshape(len(x),1) if isVector else np.array([[x]])
        
        # Cas de la régression non-linéaire
        powerArray = np.arange(self.M) + 1
        
        # Si x est un vecteur
        if isVector:
            length = len(x)
            phi = np.zeros((length, self.M))

            for i in range(length):
                phi[i,] = np.power(x[i], powerArray)

            return phi

        # Si x est un scalaire
        return np.power(x, powerArray).reshape(1,self.M)

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        k = 10 if X.shape[0] >= 10 else X.shape[0]
        # permet d'arrêter la recherche lorsque la différence d'erreur 
        # entre deux runs est trop grande
        maxErrorDifference = 0.1
        minError = [1,np.max(t)]
        self.M = 1
        tmp = -1
        
        while(True):
            kFold = model_selection.KFold(n_splits=k,shuffle=True,random_state=self.M)
            errorList = list()

            for trainIdx, testIdx in kFold.split(X):
                trainX = X[trainIdx]
                trainT = t[trainIdx]
                testX = X[testIdx]
                testT = t[testIdx]

                self.resolution(trainX, trainT)
                prediction = self.prediction(testX)
                err = self.erreur(testT,prediction)
                errorList.append(np.mean(err))
            
            mean = np.mean(errorList)
            print("M = {}, erreur après cross-validation : {:.4f}".format(self.M,mean))

            toContinue = True if tmp == -1 else maxErrorDifference > (mean - tmp)

            if toContinue :
                if minError[1] > mean:
                    minError[0] = self.M
                    minError[1] = mean 
                tmp = mean
                self.M += 1
            
            else:
                print("Dépassement de maxErrorDifference entre deux runs")
                break
        
        self.M = minError[0]
        print("=> M optimal : {}, erreur de cross-validation : {:.4f}".format(minError[0],minError[1]))

    def resolution(self, X, t):
        phiX = self.fonction_base_polynomiale(X)
        
        # En utilisant le module scikit-learn
        if self.usingSkl:
            rdg = linear_model.Ridge(self.lamb)
            rdg.fit(phiX,t.reshape(-1, 1))
            self.w = np.insert(rdg.coef_, 0, rdg.intercept_)
            return

        # En utilisant la formule
        phiX = np.insert(phiX, 0, 1, axis=1)
        toInvert = (np.identity(self.M + 1) * self.lamb) + (phiX.transpose() @ phiX)
        self.w = np.linalg.solve(toInvert, phiX.transpose() @ t)

    def entrainement(self, X, t):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M
        """
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)
        
        self.resolution(X,t)

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        return self.w[0] + self.fonction_base_polynomiale(x) @ self.w[1:]

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        return np.power(t - prediction, 2)
