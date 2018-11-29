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
import csv

from tsp_solver.greedy import solve_tsp

class archi():

    def __init__(self, points, length):

        self.reseaux = []

        self.points = points
        self.nb_nodes = len(self.points)
        self.is_in_boucle = [False] * self.nb_nodes
        self.length = length
        
        self.value = 0

    def is_feasible(self):
        if len(self.reseaux) == 0:
            return False
        else:
            tmp_ant = np.array([p[2] == 1 for p in self.points])
            for r in self.reseaux:
                s = sum([self.points[v][2] == 0 for v in r["boucle"]])
                if s > 30 or s == len(r["boucle"]):
                    return False
                for v in r["boucle"]:
                    if self.points[v][2] == 0:
                        if tmp_ant[v]:
                            return False
                        else:
                            tmp_ant[v] = True
                for c in r["chaines"]:
                    if len(c) > 6:
                        return False
                    for v in range(1, len(c)):
                        if self.points[c[v]][2] == 0:
                            if tmp_ant[c[v]]:
                                return False
                            else:
                                tmp_ant[c[v]] = True
            print(tmp_ant)
            return np.all(tmp_ant)

    def get_value(self):
        return self.value

    def compute_value(self):
        value = 0
        for r in self.reseaux:
            for i in range(len(r["boucle"]) - 1):
                print(r["boucle"][i])
                print(r["boucle"].size)
                print(self.length.size)
                value += self.length[r["boucle"][i], r["boucle"][i + 1]]
            value += self.length[r["boucle"][len(r["boucle"]) - 1], r["boucle"][0]]
            for c in r["chaines"]:
                for j in range(1, len(c)):
                    value += self.length[c[j - 1], c[j]]
        self.value = value

    def set_reseaux(self, reseaux):
        self.reseaux = reseaux
        self.is_in_boucle = [False] * self.nb_nodes
        for r in reseaux:
            for v in r["boucle"]:
                self.is_in_boucle[v] = True


    def where(self, index):
        if self.is_in_boucle[index]:
            for r in len(range(self.reseaux)):
                if index in self.reseaux[r]["boucle"]:
                    return r, None
        else:
            for r in len(range(self.reseaux)):
                for c in len(range(self.reseaux[r]["chaines"])):
                    if index in self.reseaux[r]["chaines"][c]:
                        return r, c
        return None, None
        


def read_file_distance(filename): #return matrix_distance
    
    with open(filename,"r",encoding="utf8") as f:
        content = f.readlines()
        names = [x.strip() for x in content]
        nb_points = int(np.sqrt(len(names))) #nombre de points
        distance_matrix = np.zeros((nb_points, nb_points))
        for ind, name in enumerate(names):
            j = ind%nb_points
            i = ind//nb_points
            distance_matrix[i,j] = names[ind]
    return(distance_matrix)
    
def read_file_nodes(filename): #return matrix_distance
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        rows = [row for row in spamreader]
        rows = rows[1:] #delete the first ['X', 'Y', 'test'] line
        nb_points = len(rows)
        nodes_coords = np.zeros((nb_points,3))
        for ind, row in enumerate(rows):
            if row[0] == 'X':
                pass
            else:
                nodes_coords[ind, 0] = row[0]
                nodes_coords[ind, 1] = row[1]
                if row[2] == 'terminal':
                    nodes_coords[ind, 2] = 0
                elif row[2] == 'distribution':
                    nodes_coords[ind, 2] = 1
                else:
                    print("Probleme")
        return(nodes_coords)

def write_results(reseaux, filename_write):
    with open(filename_write, 'w') as f:
        for reseau in reseaux:
            boucle = reseau['boucle']
            f.write("b")
            index = np.argmin(boucle)
            #print("boucle", boucle, index)
            #print("new_boucle", boucle[index:], boucle[:index])
            new_boucle = np.concatenate((boucle[index:], boucle[:index]))
            #print(new_boucle)
            for b in new_boucle:
                f.write(" " + str(b))
            f.write("\n")
            chaines = reseau['chaines']
            for chaine in chaines:
                f.write("c")
                for c in chaine:
                    f.write(" " + str(c))
                f.write("\n")
    
def affichage(nodes_coords):
    indexes_terminal = np.where(nodes_coords[:,2] == 0)
    plt.plot(nodes_coords[:,0], nodes_coords[:,1],'r.')
    plt.plot(nodes_coords[indexes_terminal,0], nodes_coords[indexes_terminal,1],'b.')
    plt.show()

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
        #print(distrib_indices, distrib_index)

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
                #print("selection de distrib ", distrib_index)
                selected_idx = np.where(k_mean_cluster == distrib_index)[0][0]
            else:
                selected_idx = np.random.randint(0, len(k_mean_cluster))
            dic_kmeans_cluster_to_selected[k_mean_cluster[0]] = k_mean_cluster[selected_idx]
            selected_boucles.append(k_mean_cluster[selected_idx])
        #print("k_mean_clusters", k_mean_clusters)
        #print("selected_bouclses", selected_boucles)

        tsp_solver_mat = []
        for ind, point in enumerate(selected_boucles):
            tsp_solver_mat.append(mat_dist[point, selected_boucles[:ind]])
        tsp_path = solve_tsp(tsp_solver_mat)
        #print(tsp_path)
        #print("tsp_path", np.array(selected_boucles)[np.array(tsp_path)])

        reseau['boucle'] = np.array(selected_boucles)[np.array(tsp_path)]
        reseau['chaines'] = []

        for k_mean_cluster in k_mean_clusters:
            k_mean_cluster_copy = copy.deepcopy(k_mean_cluster)
            while (len(k_mean_cluster_copy) != 1):
                tsp_solver_mat = []
                for ind, point in enumerate(k_mean_cluster_copy):
                    tsp_solver_mat.append(mat_dist[point, k_mean_cluster_copy[:ind]])
                tsp_path = solve_tsp(tsp_solver_mat)
                tsp_path = np.array(k_mean_cluster_copy)[np.array(tsp_path)]
                selected_idx = dic_kmeans_cluster_to_selected[k_mean_cluster[0]]
                #print("selected_idx", selected_idx)
                #print("tsp_path", tsp_path)
                position_selected = np.where(tsp_path == selected_idx)[0][0]
                #print(position_selected)
                tsp_path_1 = tsp_path[position_selected:][:5]
                tsp_path_2 = tsp_path[:position_selected+1]
                tsp_path_2 = tsp_path_2[::-1][:5]

                k_mean_cluster_copy = [idx for idx in k_mean_cluster_copy if ((idx == selected_idx) or (idx not in tsp_path_1 and idx not in tsp_path_2))]

                if (len(tsp_path_1) > 1):
                    reseau['chaines'].append(tsp_path_1)
                if (len(tsp_path_2) > 1):
                    reseau['chaines'].append(tsp_path_2)
                #print("branch", tsp_path, tsp_path_1, tsp_path_2)
        reseaux.append(reseau)

    #print(reseaux)

    return reseaux


def temperature(k, param_temperature, T0):
    T = T0*param_temperature**k
    return(T)
    
def energy(solution):
    solution.compute_value()
    return solution.get_value()
    
def P(E1, E2, T):
    return(np.exp((E2-E1)/T))

def recuit_simule(filename_distance, filename_nodes):
    #parametres
    Kmax = 100
    param_temperature = 0.99
    T0 = 100
    nb_affichage = 100
    
    mat_dist = read_file_distance(filename_distance)
    nodes_coords = read_file_nodes(filename_nodes)
    reseaux = generation_sol_initiale(nodes_coords, mat_dist)
    arc = archi(mat_dist, nodes_coords)
    arc.set_reseaux(reseaux)

    energie_courante = energy(arc)
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
    
    nom_ville = 'pim'

    filename_distance = nom_ville + '/distances.csv'
    filename_nodes = nom_ville + '/nodes.csv'
    
    print(recuit_simule(filename_distance, filename_nodes))
