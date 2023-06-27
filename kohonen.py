# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Carte de Kohonen
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------
# Implémentation de l'algorithme des cartes auto-organisatrices de Kohonen
# ------------------------------------------------------------------------
# Pour que les divisions soient toutes réelles (pas de division entière)
from __future__ import division

import math

# Librairie d'affichage
import matplotlib.pyplot as plt
# Librairie de calcul matriciel
import numpy


class Neuron:
    ''' Classe représentant un neurone '''

    def __init__(self, w, posx, posy):
        '''
        @summary: Création d'un neurone
        @param w: poids du neurone
        @type w: numpy array
        @param posx: position en x du neurone dans la carte
        @type posx: int
        @param posy: position en y du neurone dans la carte
        @type posy: int
        '''
        # Initialisation des poids
        self.weights = w.flatten()
        # Initialisation de la position
        self.posx = posx
        self.posy = posy
        # Initialisation de la sortie du neurone
        self.y = 0.

    def compute(self, x):
        '''
        @summary: Affecte à y la valeur de sortie du neurone (i.e. la distance entre son poids et l'entrée)
        @param x: entrée du neurone
        @type x: numpy array
        '''
        # TODO
        self.y = numpy.linalg.norm(self.weights - x)

    def learn(self, eta, sigma, posxbmu, posybmu, x):
        '''
        @summary: Modifie les poids selon la règle de Kohonen
        @param eta: taux d'apprentissage
        @type eta: float
        @param sigma: largeur du voisinage
        @type sigma: float
        @param posxbmu: position en x du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
        @type posxbmu: int
        @param posybmu: position en y du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
        @type posybmu: int
        @param x: entrée du neurone
        @type x: numpy array
        '''
        # TODO (attention à ne pas changer la partie à gauche du =)
        self.weights[:] = self.weights[:] + eta * numpy.exp(-((
                    numpy.linalg.norm((numpy.array([self.posx, self.posy]) - numpy.array([posxbmu, posybmu])) ** 2) / (
                        2 * sigma ** 2)))) * (x - self.weights)


class SOM:
    ''' Classe implémentant une carte de Kohonen. '''

    def __init__(self, inputsize, gridsize):
        '''
        @summary: Création du réseau
        @param inputsize: taille de l'entrée
        @type inputsize: tuple
        @param gridsize: taille de la carte
        @type gridsize: tuple
        '''
        # Initialisation de la taille de l'entrée
        self.inputsize = inputsize
        # Initialisation de la taille de la carte
        self.gridsize = gridsize
        # Création de la carte
        # Carte de neurones
        self.map = []
        # Carte des poids
        self.weightsmap = []
        # Carte des activités
        self.activitymap = []
        for posx in range(gridsize[0]):
            mline = []
            wmline = []
            amline = []
            for posy in range(gridsize[1]):
                neuron = Neuron(numpy.random.random(self.inputsize), posx, posy)
                mline.append(neuron)
                wmline.append(neuron.weights)
                amline.append(neuron.y)
            self.map.append(mline)
            self.weightsmap.append(wmline)
            self.activitymap.append(amline)
        self.activitymap = numpy.array(self.activitymap)

    def compute(self, x):
        '''
        @summary: calcule de l'activité des neurones de la carte
        @param x: entrée de la carte (identique pour chaque neurone)
        @type x: numpy array
        '''
        # On demande à chaque neurone de calculer son activité et on met à jour la carte d'activité de la carte
        for posx in range(self.gridsize[0]):
            for posy in range(self.gridsize[1]):
                self.map[posx][posy].compute(x)
                self.activitymap[posx][posy] = self.map[posx][posy].y

    def predictWith1Neuron(self, theta1, theta2):
        '''
        Prédiction du neurone le plus proche de [theta1, theta2]
        @param theta1: Poids dans theta1 du neurone que l'on cherche à approcher
        @param theta2: Poids dans theta2 du neurone que l'on cherche à approcher
        @return: le neurone le plus proche de [theta1, theta2]
        '''
        pred = None
        dist = 100000
        for posx in range(self.gridsize[0]):
            for posy in range(self.gridsize[1]):
                tmp = math.dist(self.weightsmap[posx][posy][:2], [theta1, theta2])
                if tmp < dist:
                    dist = tmp
                    pred = [posx, posy]
        return pred
    
    def predictWith4Neurons(self, theta1, theta2):
        '''
        Prédiction du neurone le plus proche de [theta1, theta2]
        @param theta1: Poids dans theta1 du neurone que l'on cherche à approcher
        @param theta2: Poids dans theta2 du neurone que l'on cherche à approcher
        @return: Les poids [x1,x2] pondérés des 4 neurone les plus proche de [theta1, theta2]
        '''
        pred1 = None
        dist1 = 100000
        pred2 = None
        dist2 = 100000
        pred3 = None
        dist3 = 100000
        pred4 = None
        dist4 = 100000
        for posx in range(self.gridsize[0]):
            for posy in range(self.gridsize[1]):
                tmp = math.dist(self.weightsmap[posx][posy][:2], [theta1, theta2])
                if tmp < dist1:
                    dist1 = tmp
                    pred1 = self.weightsmap[posx][posy][2:]
                elif tmp < dist2:
                    dist2 = tmp
                    pred2 = self.weightsmap[posx][posy][2:]
                elif tmp < dist3:
                    dist3 = tmp
                    pred3 = self.weightsmap[posx][posy][2:]
                elif tmp < dist4:
                    dist4 = tmp
                    pred4 = self.weightsmap[posx][posy][2:]

        weights_sum = (1 / dist1) + (1 / dist2) + (1 / dist3) + (1 / dist4)
        pred = (pred1 * (1 / dist1) + pred2 * (1 / dist2) + pred3 * (1 / dist3) + pred4 * (1 / dist4)) / weights_sum
        return pred

    def predict_path(self, theta1D, theta2D, theta1A, theta2A):
        '''
        Prédiction des points de passage de la mains en fonction d'un point de départ et d'un point d'arrivée dans theta
        @param theta1D: Point de départ dans theta1
        @param theta2D: Point de départ dans theta2
        @param theta1A: Point d'arrivée dans theta1
        @param theta2A: Point d'arrivée dans theta2
        @return: La liste de [x1,x2] où la main passe
        '''
        [x_tmp, y_tmp] = self.predictWith1Neuron(theta1D, theta2D)
        [x_arr, y_arr] = self.predictWith1Neuron(theta1A, theta2A)
        position = list()
        position.append([x_tmp, y_tmp])
        while x_tmp != x_arr or y_tmp != y_arr:
            tmp = [1000, 1000, 1000, 1000]
            if x_tmp < len(self.weightsmap) - 1:
                tmp[0] = math.dist(self.weightsmap[x_tmp + 1][y_tmp][:2], [theta1A, theta2A])
            if x_tmp > 0:
                tmp[1] = math.dist(self.weightsmap[x_tmp - 1][y_tmp][:2], [theta1A, theta2A])
            if y_tmp < len(self.weightsmap[0]) - 1:
                tmp[2] = math.dist(self.weightsmap[x_tmp][y_tmp + 1][:2], [theta1A, theta2A])
            if y_tmp > 0:
                tmp[3] = math.dist(self.weightsmap[x_tmp][y_tmp - 1][:2], [theta1A, theta2A])

            index = tmp.index(min(tmp))
            match index:
                case 0:
                    x_tmp, y_tmp = x_tmp + 1, y_tmp
                case 1:
                    x_tmp, y_tmp = x_tmp - 1, y_tmp
                case 2:
                    x_tmp, y_tmp = x_tmp, y_tmp + 1
                case 3:
                    x_tmp, y_tmp = x_tmp, y_tmp - 1
            position.append([x_tmp, y_tmp])
        path_theta=list()
        path_x=list()
        for i in range(len(position)):
          path_theta.append([self.weightsmap[position[i][0]][position[i][1]][0],self.weightsmap[position[i][0]][position[i][1]][1]])
          path_x.append([self.weightsmap[position[i][0]][position[i][1]][2],self.weightsmap[position[i][0]][position[i][1]][3]])
        print("Pour θ1=",theta1D," θ2=",theta2D, " θ1'=",theta1A, " θ2'=",theta2A)
        print("[θ1,θ2] des points de passage: \n\t",path_theta.__str__().replace('], [',']\n\t ['))
        print("[x1,x2] des points de passage: \n\t",path_x.__str__().replace('], [',']\n\t ['))
        print("Indices des neurones dans le passage: \n\t", position.__str__().replace('], [',']\n\t ['))
        t='θ1='+str(theta1D)+', θ2='+str(theta2D)+', θ\'1='+str(theta1A)+', θ\'2='+str(theta2A)
        self.scatter_plot_2_path(False, position,t)
        return path_x

    def scatter_plot_2_path(self, interactive, pos, titre):
        '''
        Affichage du réseau dans l'espace d'entrée en 2 fois 2d et coloration des neurones du chemin
        @param interactive: Indique si l'affichage se fait en mode interactif
        @param pos: Liste des indices dans weightsmap des neurones par lesquels les angles et la mains passe
        @return: Nothing
        '''
        # Création de la figure
        if not interactive:
            plt.figure()
        # Affichage des 2 premières dimensions dans le plan
        plt.subplot(1, 2, 1)
        # Récupération des poids
        w = numpy.array(self.weightsmap)
        # Affichage des poids
        for x in range(w.shape[0]):
            for y in range(w.shape[1]):
                if [x, y] in pos:
                    plt.scatter(w[x, y, 0].flatten(), w[x, y, 1].flatten(), c='red')
                else:
                    plt.scatter(w[x, y, 0].flatten(), w[x, y, 1].flatten(), c='k')
        # Affichage de la grille
        for i in range(w.shape[0]):
            plt.plot(w[i, :, 0], w[i, :, 1], 'k', linewidth=1.)
        for i in range(w.shape[1]):
            plt.plot(w[:, i, 0], w[:, i, 1], 'k', linewidth=1.)

        # Affichage des 2 dernières dimensions dans le plan
        plt.subplot(1, 2, 2)
        # Récupération des poids
        w = numpy.array(self.weightsmap)
        # Affichage des poids
        for x in range(w.shape[0]):
            for y in range(w.shape[1]):
                if [x, y] in pos:
                    plt.scatter(w[x, y, 2].flatten(), w[x, y, 3].flatten(), c='red')
                else:
                    plt.scatter(w[x, y, 2].flatten(), w[x, y, 3].flatten(), c='k')
        # Affichage de la grille
        for i in range(w.shape[0]):
            plt.plot(w[i, :, 2], w[i, :, 3], 'k', linewidth=1.)
        for i in range(w.shape[1]):
            plt.plot(w[:, i, 2], w[:, i, 3], 'k', linewidth=1.)
        # Affichage du titre de la figure
        plt.suptitle('Poids et chemin pour '+titre)
        # Affichage de la figure
        if not interactive:
            plt.show()


    def predict_theta(self, x1, x2):
        '''
        Prédiction du neurone le plus proche de [x1,x2]
        @param x1: Poids dans x1 du neurone que l'on cherche à approcher
        @param x2: Poids dans x2 du neurone que l'on cherche à approcher
        @return: Les poids [theta1, theta2] pondérés des 4 neurone les plus proche de [x1,x2]
        '''
        pred1 = None
        dist1 = 100000
        pred2 = None
        dist2 = 100000
        pred3 = None
        dist3 = 100000
        pred4 = None
        dist4 = 100000
        for posx in range(self.gridsize[0]):
            for posy in range(self.gridsize[1]):
                tmp = math.dist(self.weightsmap[posx][posy][2:], [x1, x2])
                if tmp < dist1:
                    dist1 = tmp
                    pred1 = self.weightsmap[posx][posy][:2]
                elif tmp < dist2:
                    dist2 = tmp
                    pred2 = self.weightsmap[posx][posy][:2]
                elif tmp < dist3:
                    dist3 = tmp
                    pred3 = self.weightsmap[posx][posy][:2]
                elif tmp < dist4:
                    dist4 = tmp
                    pred4 = self.weightsmap[posx][posy][2:]
        weights_sum = (1 / dist1) + (1 / dist2) + (1 / dist3) + (1 / dist4)
        pred = (pred1 * (1 / dist1) + pred2 * (1 / dist2) + pred3 * (1 / dist3) + pred4 * (1 / dist4)) / weights_sum
        return pred
        '''pred= (pred1*(1/dist1)+pred2*(1/dist2)+pred3*(1/dist3)+pred4*(1/dist4))/((1/dist1)+(1/dist2)+(1/dist3)+(1/dist4))
        return pred'''

    def learn(self, eta, sigma, x):
        '''
        @summary: Modifie les poids de la carte selon la règle de Kohonen
        @param eta: taux d'apprentissage
        @type eta: float
        @param sigma: largeur du voisinage
        @type sigma: float
        @param x: entrée de la carte
        @type x: numpy array
        '''
        # Calcul du neurone vainqueur
        bmux, bmuy = numpy.unravel_index(numpy.argmin(self.activitymap), self.gridsize)
        # Mise à jour des poids de chaque neurone
        for posx in range(self.gridsize[0]):
            for posy in range(self.gridsize[1]):
                self.map[posx][posy].learn(eta, sigma, bmux, bmuy, x)

    def scatter_plot(self, interactive=False):
        '''
        @summary: Affichage du réseau dans l'espace d'entrée (utilisable dans le cas d'entrée à deux dimensions et d'une carte avec une topologie de grille carrée)
        @param interactive: Indique si l'affichage se fait en mode interactif
        @type interactive: boolean
        '''
        # Création de la figure
        if not interactive:
            plt.figure()
        # Récupération des poids
        w = numpy.array(self.weightsmap)
        # Affichage des poids
        plt.scatter(w[:, :, 0].flatten(), w[:, :, 1].flatten(), c='k')
        # Affichage de la grille
        for i in range(w.shape[0]):
            plt.plot(w[i, :, 0], w[i, :, 1], 'k', linewidth=1.)
        for i in range(w.shape[1]):
            plt.plot(w[:, i, 0], w[:, i, 1], 'k', linewidth=1.)
        # Modification des limites de l'affichage
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        # Affichage du titre de la figure
        plt.suptitle('Poids dans l\'espace d\'entree')
        # Affichage de la figure
        if not interactive:
            plt.show()

    def scatter_plot_2(self, interactive=False):
        '''
        @summary: Affichage du réseau dans l'espace d'entrée en 2 fois 2d (utilisable dans le cas d'entrée à quatre dimensions et d'une carte avec une topologie de grille carrée)
        @param interactive: Indique si l'affichage se fait en mode interactif
        @type interactive: boolean
        '''
        # Création de la figure
        if not interactive:
            plt.figure()
        # Affichage des 2 premières dimensions dans le plan
        plt.subplot(1, 2, 1)
        # Récupération des poids
        w = numpy.array(self.weightsmap)
        # Affichage des poids
        plt.scatter(w[:, :, 0].flatten(), w[:, :, 1].flatten(), c='k')
        # Affichage de la grille
        for i in range(w.shape[0]):
            plt.plot(w[i, :, 0], w[i, :, 1], 'k', linewidth=1.)
        for i in range(w.shape[1]):
            plt.plot(w[:, i, 0], w[:, i, 1], 'k', linewidth=1.)
        # Affichage des 2 dernières dimensions dans le plan
        plt.subplot(1, 2, 2)
        # Récupération des poids
        w = numpy.array(self.weightsmap)
        # Affichage des poids
        plt.scatter(w[:, :, 2].flatten(), w[:, :, 3].flatten(), c='k')
        # Affichage de la grille
        for i in range(w.shape[0]):
            plt.plot(w[i, :, 2], w[i, :, 3], 'k', linewidth=1.)
        for i in range(w.shape[1]):
            plt.plot(w[:, i, 2], w[:, i, 3], 'k', linewidth=1.)
        # Affichage du titre de la figure
        plt.suptitle('Poids dans l\'espace d\'entree')
        # Affichage de la figure
        if not interactive:
            plt.show()

    def plot(self):
        '''
        @summary: Affichage des poids du réseau (matrice des poids)
        '''
        # Récupération des poids
        w = numpy.array(self.weightsmap)
        # Création de la figure
        f, a = plt.subplots(w.shape[0], w.shape[1])
        # Affichage des poids dans un sous graphique (suivant sa position de la SOM)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                plt.subplot(w.shape[0], w.shape[1], i * w.shape[1] + j + 1)
                im = plt.imshow(w[i, j].reshape(self.inputsize), interpolation='nearest', vmin=numpy.min(w),
                                vmax=numpy.max(w), cmap='binary')
                plt.xticks([])
                plt.yticks([])
        # Affichage de l'échelle
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)
        # Affichage du titre de la figure
        plt.suptitle('Poids dans l\'espace de la carte')
        # Affichage de la figure
        plt.show()

    def ecartVoisin(self):
        w = numpy.array(self.weightsmap)
        dist = 0.0
        nbarrete = 0
        for i in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                if i < self.gridsize[0] - 1:
                    dist += math.dist(w[i][y], w[i + 1][y])
                    nbarrete += 1
                if y < self.gridsize[1] - 1:
                    dist += math.dist(w[i][y], w[i][y + 1])
                    nbarrete += 1

        return dist / nbarrete

    def MSE(self, X):
        '''
        @summary: Calcul de l'erreur de quantification vectorielle moyenne du réseau sur le jeu de données
        @param X: le jeu de données
        @type X: numpy array
        '''
        # On récupère le nombre d'exemples
        nsamples = X.shape[0]
        # Somme des erreurs quadratiques
        s = 0
        # Pour tous les exemples du jeu de test
        for x in X:
            # On calcule la distance à chaque poids de neurone
            self.compute(x.flatten())
            # On rajoute la distance minimale au carré à la somme71480923
            s += numpy.min(self.activitymap) ** 2
        # On renvoie l'erreur de quantification vectorielle moyenne
        return s / nsamples


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Création d'un réseau avec une entrée (2,1) et une carte (10,10)
    # TODO mettre à jour la taille des données d'entrée pour les données robotiques
    network = SOM((4, 1), (10, 10))
    # PARAMÈTRES DU RÉSEAU
    # Taux d'apprentissage
    ETA = 0.05
    # Largeur du voisinage
    SIGMA = 1.4
    # Nombre de pas de temps d'apprentissage
    N = 15000
    # Affichage interactif de l'évolution du réseau
    # TODO à mettre à faux pour que les simulations aillent plus vite
    VERBOSE = False
    # Nombre de pas de temps avant rafraissichement de l'affichage
    NAFFICHAGE = 3000
    # DONNÉES D'APPRENTISSAGE
    # Nombre de données à générer pour les ensembles 1, 2 et 3
    # TODO décommenter les données souhaitées
    nsamples = 1200
    # Ensemble de données 1
    # samples = numpy.random.random((nsamples,2,1))*2-1
    # ensemble maison
    # samples1 = numpy.random.random((nsamples//10,2,1))
    # samples1[:,0,:] -= 1
    # samples2 = numpy.random.random((nsamples*9//10,2,1))
    # samples2[:,1,:] -= 1
    # samples = numpy.concatenate((samples1,samples2))
    # Ensemble de données 2
    # samples1 = -numpy.random.random((nsamples//3,2,1))
    # samples2 = numpy.random.random((nsamples//3,2,1))
    # samples2[:,0,:] -= 1
    # samples3 = numpy.random.random((nsamples//3,2,1))
    # samples3[:,1,:] -= 1
    # samples = numpy.concatenate((samples1,samples2,samples3))
    # Ensemble de données 3
    # samples1 = numpy.random.random((nsamples//2,2,1))
    # samples1[:,0,:] -= 1
    # samples2 = numpy.random.random((nsamples//2,2,1))
    # samples2[:,1,:] -= 1
    # samples = numpy.concatenate((samples1,samples2))
    # Ensemble de données robotiques
    samples = numpy.random.random((nsamples, 4, 1))
    samples[:, 0:2, :] *= numpy.pi
    l1 = 0.7
    l2 = 0.3
    samples[:, 2, :] = l1 * numpy.cos(samples[:, 0, :]) + l2 * numpy.cos(samples[:, 0, :] + samples[:, 1, :])
    samples[:, 3, :] = l1 * numpy.sin(samples[:, 0, :]) + l2 * numpy.sin(samples[:, 0, :] + samples[:, 1, :])
    # Affichage des données (pour les ensembles 1, 2 et 3)
    # plt.figure()
    # plt.scatter(samples[:,0,0], samples[:,1,0])
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    # plt.suptitle('Donnees apprentissage')
    # plt.show()
    # Affichage des données (pour l'ensemble robotique)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(samples[:, 0, 0].flatten(), samples[:, 1, 0].flatten(), c='k')
    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 2, 0].flatten(), samples[:, 3, 0].flatten(), c='k')
    plt.suptitle('Donnees apprentissage')
    plt.show()

    # SIMULATION
    # Affichage des poids du réseau
    network.plot()
    # Initialisation de l'affichage interactif
    if VERBOSE:
        # Création d'une figure
        plt.figure()
        # Mode interactif
        plt.ion()
        # Affichage de la figure
        plt.show()
    # Boucle d'apprentissage
    for i in range(N + 1):
        # Choix d'un exemple aléatoire pour l'entrée courante
        index = numpy.random.randint(nsamples)
        x = samples[index].flatten()
        # Calcul de l'activité du réseau
        network.compute(x)
        # Modification des poids du réseau
        network.learn(ETA, SIGMA, x)
        # Mise à jour de l'affichage
        if VERBOSE and i % NAFFICHAGE == 0:
            # Effacement du contenu de la figure
            plt.clf()
            # Remplissage de la figure
            # TODO à remplacer par scatter_plot_2 pour les données robotiques
            network.scatter_plot_2(True)
            # Affichage du contenu de la figure
            plt.pause(0.00001)
            plt.draw()
    # Fin de l'affichage interactif
    if VERBOSE:
        # Désactivation du mode interactif
        plt.ioff()
    # Affichage des poids du réseau
    network.plot()
    # Affichage de l'erreur de quantification vectorielle moyenne après apprentissage
    print("erreur de quantification vectorielle moyenne ", network.MSE(samples))
    print("distance de qualification vectorielle moyenne ", network.ecartVoisin())
    print("\n****** PREDICTION X ******")
    a1 = 2.55
    a2 = 0.57
    [pred_x1, pred_x2] = network.predictWith4Neurons(a1, a2)
    print("prediction de x1,x2 pour θ1=", a1, " et θ2=", a2, ": ", [pred_x1, pred_x2])
    res_x_esp = [l1 * math.cos(a1) + l2 * math.cos(a1 + a2), l1 * math.sin(a1) + l2 * math.sin(a1 + a2)]
    print("résultat de [x1,x2] espéré: ", res_x_esp)
    print("erreur de prediction : ", math.dist([pred_x1, pred_x2], res_x_esp))

    print("\n****** PREDICTION THETA ******")
    [x1, x2] = res_x_esp
    [pred_theta1, pred_theta2] = network.predict_theta(x1, x2)
    print("prediction de θ1,θ2 pour x1=", x1, " et x2=", x2, ": ", [pred_theta1, pred_theta2])
    d = (x1 ** 2 + x2 ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta2 = math.atan2(math.sqrt(1 - d ** 2), d)
    theta1 = math.atan2(x2, x1) - math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    res_theta_esp = [theta1, theta2]
    print("résultat de [θ1,θ2] espéré: ", res_theta_esp)
    print("erreur de prediction : ", math.dist([pred_theta1, pred_theta2], res_theta_esp))

    print("\n****** PREDICTION CHEMIN ******")
    network.predict_path(0.5, 0.5, 2.6, 2.6)
