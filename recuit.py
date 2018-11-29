# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:21:22 2018

@author: anatole parre
"""

import numpy as np
import random as rd
import sklearn.cluster
import time
import copy

from tsp_solver.greedy import solve_tsp

def generation_voisin(solution, k):
    neighbor = copy.deepcopy(solution)

    return neighbor
    
def generation_sol_initiale(input_sommets, mat_dist):

    reseaux = []

    # K means avec nombre de distributeurs
    cluster_nb = int(input_sommets[:, 2].sum())
    #k_means = sklearn.cluster.KMeans(input_sommets[:, 2].sum()).fit(input_sommets[0:2, :])

    distrib_indices = np.where(input_sommets[:, 2] == 1)[0]
    recept_indices = np.where(input_sommets[:, 2] ==0)[0]
    clustering = np.argmin(mat_dist[:,distrib_indices], axis=1)

    # clusters = [[]]*cluster_nb
    # for sommet_idx in range(input_sommets.shape[0]):
    #     clusters[clustering[sommet_idx]].append(sommet_idx)

    for cluster_idx in range(cluster_nb):
        # cree un reseau par cluster
        reseau = {}
        distrib_index = distrib_indices[cluster_idx]
        print(distrib_indices, distrib_index)

        cluster_vertices_indices = np.where(clustering == cluster_idx)[0]
        k_means_nb = cluster_vertices_indices.shape[0] // 3

        k_means = sklearn.cluster.KMeans(k_means_nb).fit(input_sommets[cluster_vertices_indices, 0:2])
        k_mean_clusters = [[]]*(k_means_nb)
        for clust in range(k_means_nb):
            clust_idx = np.where(k_means.labels_ == clust)[0]
            k_mean_clusters[clust] = cluster_vertices_indices[clust_idx]

        selected_boucles = []
        dic_kmeans_cluster_to_selected = {}
        for k_mean_cluster in k_mean_clusters:
            if (distrib_index in k_mean_cluster):
                print("selection de distrib ", distrib_index)
                selected_idx = np.where(k_mean_cluster == distrib_index)[0][0]
            else:
                selected_idx = np.random.randint(0, len(k_mean_cluster))
            dic_kmeans_cluster_to_selected[k_mean_cluster[0]] = k_mean_cluster[selected_idx]
            selected_boucles.append(k_mean_cluster[selected_idx])
        print("k_mean_clusters", k_mean_clusters)
        print("selected_bouclses", selected_boucles)

        tsp_solver_mat = []
        for ind, point in enumerate(selected_boucles):
            tsp_solver_mat.append(mat_dist[point, selected_boucles[:ind]])
        tsp_path = solve_tsp(tsp_solver_mat)
        print(tsp_path)
        print("tsp_path", np.array(selected_boucles)[np.array(tsp_path)])

        reseau['boucle'] = np.array(selected_boucles)[np.array(tsp_path)]
        reseau['chaines'] = []

        for k_mean_cluster in k_mean_clusters:
            tsp_solver_mat = []
            for ind, point in enumerate(k_mean_cluster):
                tsp_solver_mat.append(mat_dist[point, k_mean_cluster[:ind]])
            tsp_path = solve_tsp(tsp_solver_mat)
            tsp_path = np.array(k_mean_cluster)[np.array(tsp_path)]
            selected_idx = dic_kmeans_cluster_to_selected[k_mean_cluster[0]]
            print("selected_idx", selected_idx)
            print("tsp_path", tsp_path)
            position_selected = np.where(tsp_path == selected_idx)[0][0]
            print(position_selected)
            tsp_path_1 = tsp_path[position_selected:]
            tsp_path_2 = tsp_path[:position_selected+1]
            tsp_path_2 = tsp_path_2[::-1]
            if (len(tsp_path_1) > 1):
                reseau['chaines'].append(tsp_path_1)
            if (len(tsp_path_2) > 1):
                reseau['chaines'].append(tsp_path_2)
            print("branch", tsp_path, tsp_path_1, tsp_path_2)
        reseaux.append(reseau)

    print(reseaux)

    return reseaux


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

def recuit_simule(mat_dist, Rcom, Rcapt):
    #parametres
    Kmax = 100
    param_temperature = 0.99
    T0 = 100
    nb_affichage = 100
    
    coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs = generation_sol_initiale(mat_dist, Rcom, Rcapt)
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
    
    mat_dist = 'Instances\captANOR225_9_20.dat'
    #mat_dist = 10

    Rcom = 1
    Rcapt = 1
    
    print("Longueur optimale ",recuit_simule(mat_dist, Rcom, Rcapt))
"""