# ============================================================= 
# Some auxiliary functions to deal with simulated annealing
# Author: leogabac
# ============================================================= 

import os
import sys

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import isclose

ureg = ice.ureg
idx = pd.IndexSlice


# Some auxiliary functions

def flip_colloid_at_index(col, index):
    """
        Flips the direction of a given colloid at a certain index.
    """

    #col2 = col.copy(deep = True) 
    c = col[index]
    c.colloid = -c.colloid
    c.direction = -c.direction
    col[index] = c
    return col

def flip_colloids(col, amount = 1, indices = None):
    """
        Flips many colloids randomly.
        Give an indices list for nonrandom flips
    """

    if indices is None:
        indices = np.random.randint(0,len(col)-1,amount)

    for index in indices:
        col = flip_colloid_at_index(col,index)
    return col

def is_accepted(dE,T):
    """
        Acceptation function for simulated annealing.
        Takes dE (Energy difference) and T (Temperature)
    """

    if dE < 0:
        return True
    else:
        r = np.random.rand()
        if r < np.exp(-dE/T):
            return True
        else:
            return False
        
def get_index_from_position(col,pos):
    """
        Given the position of a colloid, returns its index in the colloidal ice object.
    """

    for idx,c in enumerate(col):
        currentPos = c.center.magnitude.round()
        sepNorm = np.linalg.norm(currentPos - pos)

        if isclose(0,sepNorm,rel_tol=1e-16):
            return idx


def fix_position(position,a,size):
    """
        Fixes the position to fit in the box
        0 < x < size*a, and 
        0 < y < size*a 
    """
    L = size*a

    # Apply BC to x
    position[0] = position[0] % L
    if position[0]<0:
        position[0] += L

    # Apply BC to y
    position[1] = position[1] % L
    if position[1]<0:
        position[1] += L

    return position


def is_horizontal(direction):
    """
        Checks if a given direction is horizontal.
    """
    x = np.array([1,0,0])
    dotP = np.dot(direction,x)

    if isclose(abs(dotP),1,rel_tol=1e-3):
        return True
    else:
        return False

def flip_loop(col, a=30, size=10):
    """
        Given a colloidal_ice object. Flips spins in a counter clockwise loop.
    """

    sel = np.random.randint(0,len(col))
    if is_horizontal(col[sel].direction):
            displacements = [
            np.array([0,0,0]),
            np.array([0,a,0]),
            np.array([a/2,a/2,0]),
            np.array([-a/2,a/2,0]) ]
    else:
            displacements = [
            np.array([0,0,0]),
            np.array([-a,0,0]),
            np.array([-a/2,a/2,0]),
            np.array([-a/2,-a/2,0]) ]

    positions = [ col[sel].center.magnitude + d for d in displacements]
    positions = [ fix_position(x,a,size).round() for x in positions]
    idxs = [get_index_from_position(col,x) for x in positions]

    col2 = flip_colloids(col,indices=idxs)
    return col2