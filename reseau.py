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
        
