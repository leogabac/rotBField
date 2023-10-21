# ============================================================= 
# Some auxiliary functions to deal with colloidal ice systems
# Author: leogabac
# ============================================================= 

import os
import sys

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
idx = pd.IndexSlice


def plotColloid(rawtrj, frame):

    """ 
        For a given trajectory, returns a plot with the colloid.
        Returns a fig,axes tuple.
    """

    f, ax = plt.subplots(figsize=(5, 5)); # Initialize

    trj_particle = rawtrj[rawtrj.type==1]
    trj_trap = rawtrj[rawtrj.type==2]

    xparticle = np.array(trj_particle.loc[idx[frame,:],"x"])
    yparticle = np.array(trj_particle.loc[idx[frame,:],"y"])
    ax.plot(xparticle,yparticle,'o', color='y')

    xtrap = np.array(trj_trap.loc[idx[frame,:],"x"])
    ytrap = np.array(trj_trap.loc[idx[frame,:],"y"])
    ax.plot(xtrap,ytrap,'o', color='g')

    ax.axis("square");
    return f,ax

def classifyVertices(vrt):
    """
        Classifies the vertices in I, II, III, IV, V, VI types.
        Returns a DataFrame
    """

    vrt["type"] = np.NaN

    vrt.loc[vrt.eval("coordination==4 & charge == -4"),"type"] = "I"
    vrt.loc[vrt.eval("coordination==4 & charge == -2"),"type"] = "II"
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)==0"),"type"] = "III"
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)>0"),"type"] = "IV" # Dipolo
    vrt.loc[vrt.eval("coordination==4 & charge == 2"),"type"] = "V"
    vrt.loc[vrt.eval("coordination==4 & charge == 4"),"type"] = "VI"
    return vrt

def getVerticesDict(verticesFolderPath):

    """
        Walks verticesFolderPath and imports all DFs into a Dictionary, classifies the vertices and drops boundaries.
        Returns a dictionary with all the DataFrames.
    """

    _, _, files = next(os.walk(verticesFolderPath))
    verticesExp = {} # Initialize
    numberExperiments = len(files)
    for i in range(1,numberExperiments+1):
        filePath = verticesFolderPath + f"vertices{i}.csv"
        vrt = pd.read_csv(filePath, index_col=[0,1])
        vrt = classifyVertices(vrt)
        vrt = vrt.dropna()
        verticesExp[f"{i}"] = vrt
    return verticesExp

def getVerticesCount(verticesDict):
    
    """
        Loops the verticesDict with all experiments and gets the counts for vertex type
        Returns a dictionary with the counts DF for all experiments
    """

    countsDict = {}
    for key,experiment in verticesDict.items():
        currentCount = ice.count_vertices(experiment)
        countsDict[key] = currentCount
    
    return countsDict

def getPaintedFrame(trj,ctrj,frame,framerate):
    v = ice.vertices()
    v = v.trj_to_vertices(ctrj.loc[frame])
    currentTime = frame / framerate
    f,ax = plotColloid(trj,frame)
    ax.set_title('t = {} s'.format(currentTime))
    v.display(ax)

def saveAllPaintedFrames(trj,ctrj,frames,framerate,path):
    for frame in frames:
        figPath = path + f"{frame}.png";
        try:
            getPaintedFrame(trj,ctrj,frame,framerate);
            print(frame)
            plt.savefig(figPath, dpi=300);
            plt.close()
        except:
            print("skip")
            continue
    return None
    
        



