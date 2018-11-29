# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:58:26 2018

@author: anatole parre
"""

nom_ville = 'pim'

filename_distance = nom_ville + '/distances.csv'
filename_nodes = nom_ville + '/nodes.csv'
filename_write = nom_ville + '/results.txt'

import numpy as np
import csv
import matplotlib.pyplot as plt

from recuit import generation_sol_initiale

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
            print("boucle", boucle, index)
            print("new_boucle", boucle[index:], boucle[:index])
            new_boucle = np.concatenate((boucle[index:], boucle[:index]))
            print(new_boucle)
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

mat_dist = read_file_distance(filename_distance)
nodes_coords = read_file_nodes(filename_nodes)
reseaux = generation_sol_initiale(nodes_coords, mat_dist)
write_results(reseaux, filename_write)