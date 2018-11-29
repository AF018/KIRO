# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:52:31 2018

@author: olivi
"""

import numpy as np

class archi():

    def __init__(self, points, length):

        self.reseaux = []

        self.points = points
        self.length = length
        
        self.nb_nodes = len(self.points)
        self.value = 0

    def is_feasible(self):
        if len(self.reseaux) == 0:
            return False
        else:
            tmp_ant = np.array([p[2] == 1 for p in self.points])
            for r in self.reseaux:
                if sum([self.points[v][2] == 0 for v in r["boucle"]]) > 30:
                    return False
                for v in r["boucle"]:
                    if self.points[v][2] == 0:
                        if tmp_ant[v]:
                            return False
                        else:
                            tmp_ant[v] = True
                for c in r["chaines"]:
                    if len(c) > 5:
                        return False
                    for v in range(1, len(c)):
                        if self.points[v][2] == 0:
                            if tmp_ant[v]:
                                return False
                            else:
                                tmp_ant[v] = True
            return np.all(tmp_ant)

    def get_value(self):
        return self.value

    def compute_value(self):
        value = 0
        for r in self.reseaux:
            for i in range(len(r["boucle"]) - 1):
                value += self.length[r["boucle"][i], r["boucle"][i + 1]]
            value += self.length[r["boucle"][len(r["boucle"]) - 1], r["boucle"][0]]
            for c in r["chaines"]:
                for j in range(1, len(c)):
                    value += self.length[c[j - 1], c[j]]
        self.value = value

    def set_reseaux(self, reseaux):
        self.reseaux = reseaux
