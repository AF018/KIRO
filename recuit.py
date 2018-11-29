# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:21:22 2018

@author: anatole parre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:08:45 2018

@author: anatole parre
"""

import numpy as np
import random as rd
import sklearn
import time
import copy

def generation_voisin(solution, k):
    neighbor = copy.deepcopy(solution)

    return neighbor
    
def generation_sol_initiale(input_sommets, mat_distance):
    # K means avec nombre de distributeurs
    cluster_nb = int(input_sommets[:, 2].sum())
    #k_means = sklearn.cluster.KMeans(input_sommets[:, 2].sum()).fit(input_sommets[0:2, :])

    distrib_indices = np.where(input_sommets[:, 2] == 1)[0]
    recept_indices = np.where(input_sommets[:, 2] ==0)[0]
    clustering = np.argmin(mat_distance[distrib_indices,:][:,recept_indices], axis=0)
    print(clustering.shape)

    clusters = [[]]*cluster_nb
    for sommet_idx in range(recept_indices.shape[0]):
        clusters[clustering[sommet_idx]].append(sommet_idx)
    
    for cluster_idx in range(cluster_nb):
        cluster_label = distrib_indices[cluster_idx]
    return(clustering)


""" 
def temperature(k, param_temperature, T0):
    T = T0*param_temperature**k
    return(T)
    
def energy(coords_pts, matAdjCom, matAdjCap, capteurs_total):
    #capteurs = sol[1]
    #penalite_com = np.sum(matAdjCom[np.array(capteurs)])
    #E = len(capteurs) - poids_com*penalite_com
    return solution.get_value()
    
def P(E1, E2, T):
    return(np.exp((E2-E1)/T))

def recuit_simule(mat_distance, Rcom, Rcapt):
    #parametres
    Kmax = 100
    param_temperature = 0.99
    T0 = 100
    nb_affichage = 100
    
    coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs = generation_sol_initiale(mat_distance, Rcom, Rcapt)
    #tool_box.trace(coords_pts, capteurs, Rcom, matAdjCap)

    energie_courante = energy(coords_pts, matAdjCom, matAdjCap, capteurs)
    meilleurs_energie = copy.deepcopy(energie_courante)
    meilleurs_capteurs = copy.deepcopy(capteurs)
    capteurs_courant = copy.deepcopy(capteurs)
    
    print("Longueur initiale " + str(len(capteurs)))
    for k in range(Kmax):
        T = temperature(k, param_temperature, T0)
        capteurs_voisin, capteurs_connexe = generation_voisin(coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs_courant, Rcom, Rcapt, k)
        energie_voisin = energy(coords_pts, matAdjCom, matAdjCap, capteurs_connexe)
                
        # Gestion de la solution courante
        if energie_voisin < energie_courante:
            capteurs_courant = copy.deepcopy(capteurs_voisin)
            energie_courante = copy.deepcopy(energie_voisin)

        else:
            if P(energie_courante, energie_voisin, T) > rd.random():
                capteurs_courant = copy.deepcopy(capteurs_voisin)
                energie_courante = copy.deepcopy(energie_voisin)
        
        # Gestion de la meilleure solution
        if energie_voisin < meilleurs_energie:
            meilleurs_capteurs = copy.deepcopy(capteurs_connexe)
            meilleurs_energie = copy.deepcopy(energie_voisin)
            
        #if k%nb_affichage == 0:
        #    trace(meilleure_solution[0], meilleure_solution[1], matAdjCom, matAdjCap)
    
    return len(meilleurs_capteurs)
    
if True:
    
    mat_distance = 'Instances\captANOR225_9_20.dat'
    #mat_distance = 10

    Rcom = 1
    Rcapt = 1
    
    print("Longueur optimale ",recuit_simule(mat_distance, Rcom, Rcapt))
"""