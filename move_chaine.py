# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:23:35 2018

@author: olivi
"""

import numpy as np
from reseau import *

def move_chaine(archi):

    for r in range(len(archi.reseaux)):
        i = 0        
        for c in archi.reseaux[r]["chaines"]:
            l = archi.length[:, c[1]]
            l[c] = np.inf
            not_found = True
            while not_found:
                a = np.argmin(l)
                if archi.is_in_boucle[a]:
                    archi.reseaux[r]["chaines"][i][0] = a
                    not_found = False
                else:
                    l[a] = np.inf

def switch(archi, p1, p2):
    if archi.points[p1][2] == 1 and not archi.is_in_boucle[p2]:
        return
    if archi.points[p2][2] == 1 and not archi.is_in_boucle[p1]:
        return
    if archi.is_in_boucle[p1] and archi.is_in_boucle[p2]:
        r1, c = archi.where(p1)
        r2, c = archi.where(p2)
        if archi.points[p1][2] == 1 and archi.points[p2][2] == 0 and r1 != r2:
            return
        if archi.points[p1][2] == 0 and archi.points[p2][2] == 1 and r1 != r2:
            return
        pos1 = archi.reseaux[r1]["boucle"].index(p1)
        pos2 = archi.reseaux[r2]["boucle"].index(p2)
        bef1 = archi.reseaux[r1]["boucle"][(pos1 - 1) % len(archi.reseaux[r1]["boucle"])]
        bef2 = archi.reseaux[r2]["boucle"][(pos2 - 1) % len(archi.reseaux[r2]["boucle"])]
        aft1 = archi.reseaux[r1]["boucle"][(pos1 + 1) % len(archi.reseaux[r1]["boucle"])]
        aft2 = archi.reseaux[r2]["boucle"][(pos2 + 1) % len(archi.reseaux[r2]["boucle"])]
        diff = archi.length[bef1, p1] + archi.length[p1, aft1] + \
               archi.length[bef2, p2] + archi.length[p2, aft2] - \
               archi.length[bef2, p1] - archi.length[p1, aft2] - \
               archi.length[bef1, p2] - archi.length[p2, aft1]
        if diff <= 0:
            archi.reseaux[r1]["boucle"][pos1] = p2
            archi.reseaux[r2]["boucle"][pos2] = p1
    elif not archi.is_in_boucle[p1] and not archi.is_in_boucle[p2]:
        r1, c1 = archi.where(p1)
        r2, c2 = archi.where(p2)
        pos1 = archi.reseaux[r1]["chaines"][c1].index(p1)
        pos2 = archi.reseaux[r2]["chaines"][c2].index(p2)
        bef1 = archi.reseaux[r1]["chaines"][c1][(pos1 - 1) % len(archi.reseaux[r1]["chaines"][c1])]
        bef2 = archi.reseaux[r2]["chaines"][c2][(pos2 - 1) % len(archi.reseaux[r2]["chaines"][c2])]
        aft1 = archi.reseaux[r1]["chaines"][c1][(pos1 + 1) % len(archi.reseaux[r1]["chaines"][c1])]
        if aft1 >= len(archi.reseaux[r1]["chaines"][c1]):
            aft1 = p1
        aft2 = archi.reseaux[r2]["chaines"][c2][(pos2 + 1) % len(archi.reseaux[r2]["chaines"][c2])]
        if aft2 >= len(archi.reseaux[r2]["chaines"][c2]):
            aft2 = p2
        diff = archi.length[bef1, pos1] + archi.length[pos1, aft1] + \
               archi.length[bef2, pos2] + archi.length[pos2, aft2] - \
               archi.length[bef2, pos1] - archi.length[pos1, aft2] - \
               archi.length[bef1, pos2] - archi.length[pos2, aft1]
        if diff <= 0:
            archi.reseaux[r1]["boucle"][pos1] = p2
            archi.reseaux[r2]["boucle"][pos2] = p1
    elif not archi.is_in_boucle[p1] and archi.is_in_boucle[p2]:
        r1, c1 = archi.where(p1)
        r2, c2 = archi.where(p2)
        pos1 = archi.reseaux[r1]["chaines"][c1].index(p1)
        pos2 = archi.reseaux[r2]["boucle"].index(p2)
        bef1 = archi.reseaux[r1]["chaines"][c1][(pos1 - 1) % len(archi.reseaux[r1]["chaines"][c1])]
        bef2 = archi.reseaux[r2]["boucle"][(pos2 - 1) % len(archi.reseaux[r2]["boucle"])]
        aft1 = archi.reseaux[r1]["chaines"][c1][(pos1 + 1) % len(archi.reseaux[r1]["chaines"][c1])]
        if aft1 >= len(archi.reseaux[r1]["chaines"][c1]):
            aft1 = p1
        aft2 = archi.reseaux[r2]["boucle"][(pos2 + 1) % len(archi.reseaux[r2]["boucle"])]
        diff = archi.length[bef1, pos1] + archi.length[pos1, aft1] + \
               archi.length[bef2, pos2] + archi.length[pos2, aft2] - \
               archi.length[bef2, pos1] - archi.length[pos1, aft2] - \
               archi.length[bef1, pos2] - archi.length[pos2, aft1]
        if diff <= 0:
            archi.reseaux[r1]["boucle"][pos1] = p2
            archi.reseaux[r2]["boucle"][pos2] = p1
            archi.is_in_boucle[p1] = True
            archi.is_in_boucle[p2] = False
    elif archi.is_in_boucle[p1] and not archi.is_in_boucle[p2]:
        r1, c1 = archi.where(p1)
        r2, c2 = archi.where(p2)
        pos2 = archi.reseaux[r2]["chaines"][c2].index(p2)
        pos1 = archi.reseaux[r1]["boucle"].index(p1)
        bef2 = archi.reseaux[r2]["chaines"][c1][(pos2 - 1) % len(archi.reseaux[r2]["chaines"][c2])]
        bef1 = archi.reseaux[r1]["boucle"][(pos1 - 1) % len(archi.reseaux[r1]["boucle"])]
        aft2 = archi.reseaux[r1]["chaines"][c1][(pos1 + 1) % len(archi.reseaux[r1]["chaines"][c1])]
        if aft2 >= len(archi.reseaux[r2]["chaines"][c2]):
            aft2 = p2
        aft1 = archi.reseaux[r1]["boucle"][(pos1 + 1) % len(archi.reseaux[r1]["boucle"])]
        diff = archi.length[bef1, pos1] + archi.length[pos1, aft1] + \
               archi.length[bef2, pos2] + archi.length[pos2, aft2] - \
               archi.length[bef2, pos1] - archi.length[pos1, aft2] - \
               archi.length[bef1, pos2] - archi.length[pos2, aft1]
        if diff <= 0:
            archi.reseaux[r1]["boucle"][pos1] = p2
            archi.reseaux[r2]["boucle"][pos2] = p1
            archi.is_in_boucle[p1] = False
            archi.is_in_boucle[p2] = True

def move_node(archi, p):
    if archi.points[p][2] == 1:
        return
    if archi.is_in_boucle[p]:
        r, c = archi.where(p)
        l = archi.length[p, :]
        l[p] = np.inf
        not_found = True
        v = None
        rv = None
        cv = None
        while not_found and np.min(l) != np.inf:
            v = np.argmin(l)
            rv, cv = archi.where(v)
            if rv != r or cv != c:
                if cv is None and sum([archi.points[v][2] == 0 for v in archi.reseaux[rv]["boucle"]]) < 30:
                    not_found = False
                if cv is not None and len(archi.reseaux[rv]["chaines"][cv]) < 6:
                    not_found = False
            l[v] = np.inf
        if np.min(l) != np.inf:
            return
        if cv is None:
            if c is None:
                archi.reseaux[r]["boucle"].remove(p)
            else:
                archi.reseaux[r]["chaines"][c].remove(p)
            index = archi.reseaux[rv]["boucle"].index(v)
            archi.reseaux[rv]["boucle"] = archi.reseaux[rv]["boucle"][0:index] + [p] + archi.reseaux[rv]["boucle"][index:]
        else:
            if c is None:
                archi.reseaux[r]["boucle"].remove(p)
            else:
                archi.reseaux[r]["chaines"][c].remove(p)
            index = archi.reseaux[rv]["chaines"][cv].index(v)
            archi.reseaux[rv]["chaines"][cv] = archi.reseaux[rv]["chaines"][cv][0:index] + [p] + archi.reseaux[rv]["chaines"][cv][index:]
        
            