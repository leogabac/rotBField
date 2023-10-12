# ========================================================================
# Trajectories to vertices
# Author: leogabac
# Info
# This script takes several trajectories from niSim.py
# and produces the vertices datasets por post counting
# ========================================================================


import os
import sys


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore some messages from pandas

import pandas as pd

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

ureg = ice.ureg
idx = pd.IndexSlice

trjPath = "../data/rrot5mT60s60s/trj/"
ctrjPath = "../data/rrot5mT60s60s/ctrj/"
verticesPath = "../data/rrot5mT60s60s/vertices/"

# Get the number of realizations
_, _, files = next(os.walk(trjPath))
realizations = len(files)

for i in range(1,realizations+1):
    print("============================================================")
    print(f"Working on realization {i}")

    # Importing files
    trjFile = trjPath + f"trj{i}.csv"
    print(f"Opening " + trjFile)
    ctrjFile = ctrjPath + f"ctrj{i}.csv"
    print(f"Opening " + ctrjFile)
    trj_raw = trj = pd.read_csv(trjFile, index_col=[0,1])
    ctrj_raw = pd.read_csv(ctrjFile, index_col=[0,1])

    v = ice.vertices()
    frames = ctrj_raw.index.get_level_values("frame").unique()

    verticesFile = verticesPath + f"vertices{i}.csv"
    v.trj_to_vertices(ctrj_raw.loc[frames[::5]])

    print(f"Saving vertices to " + verticesFile)
    v.vertices.to_csv(verticesFile)
