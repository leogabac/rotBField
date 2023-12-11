# ============================================================= 
# Some auxiliary functions to deal with colloidal ice systems
# God bless whoever reads this code
# Author: leogabac
# ============================================================= 

import os
import sys

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ureg = ice.ureg
idx = pd.IndexSlice


def plotColloid(trj, frame):

    """ 
        Plots a particle system at a given frame for a given lammps trajectory.
        Returns a (fig,ax).
        ----------
        Parameters:
        * trj (pd Dataframe): lammps trajectory.
        * frame
    """

    f, ax = plt.subplots(figsize=(5, 5)); # Initialize

    trj_particle = trj[trj.type==1]
    trj_trap = trj[trj.type==2]

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
        ----------
        Parameters:
        * vrt (pd Dataframe): Vertices df
    """

    vrt["type"] = np.NaN

    vrt.loc[vrt.eval("coordination==4 & charge == -4"),"type"] = "I"
    vrt.loc[vrt.eval("coordination==4 & charge == -2"),"type"] = "II"
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)==0"),"type"] = "III"
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)>0"),"type"] = "IV" # Dipolo
    vrt.loc[vrt.eval("coordination==4 & charge == 2"),"type"] = "V"
    vrt.loc[vrt.eval("coordination==4 & charge == 4"),"type"] = "VI"
    return vrt

def getVerticesDict(path):

    """
        Walks path and imports all DFs into a Dictionary, classifies the vertices and drops boundaries.
        Returns a dictionary with all the DataFrames.
        ----------
        Parameters:
        * path: Path where the vertices are located.
    """

    _, _, files = next(os.walk(path))
    verticesExp = {} # Initialize
    numberExperiments = len(files)
    for i in range(1,numberExperiments+1):
        filePath = path + f"vertices{i}.csv"
        vrt = pd.read_csv(filePath, index_col=[0,1])
        vrt = classifyVertices(vrt)
        vrt = vrt.dropna()
        verticesExp[f"{i}"] = vrt
    return verticesExp

def getVerticesCount(verticesDict):
    
    """
        Loops the verticesDict with all experiments and gets the counts for vertex type
        Returns a dictionary with the counts DF for all experiments
        ----------
        Parameters:
        * verticesDict: Dictionary from getVerticesDict()
    """

    countsDict = {}
    for key,experiment in verticesDict.items():
        currentCount = ice.count_vertices(experiment)
        countsDict[key] = currentCount
    
    return countsDict

def getVerticesAverage(counts,framerate):
    
    """
        Averages over all realizations.
        ----------
        Parameters:
        * counts (Dict): Counts dictionary with all experiments.
        * framerate: Framerate from the simulation
    """
    # Get a list containing all the different frames
    allFrames = counts["1"].index.get_level_values('frame').unique().to_list()
    time = np.array(allFrames)/framerate
    
    numberFrames = len(allFrames)
    numberRealizations = len(counts)

    fractions = pd.DataFrame(columns=["time","I","II","III","IV","V","VI"], data = np.zeros((numberFrames,7)))

    for key,experiment in counts.items():
        for vertexType,vrt in experiment.groupby("type"):
            vertexFraction = np.array(vrt.fraction)
            fractions[vertexType] += vertexFraction

    fractions = fractions / numberRealizations
    # fractions["theta"] = time * np.pi/2/60 * (180/np.pi)
    fractions["time"] = time
    return fractions

def getPaintedFrame(trj,ctrj,frame,framerate):

    """
        Visualize the charges of a particular frame.
        ----------
        Parameters:
        * trj (pd Dataframe): lammps trj
        * ctrj (pd Dataframe): lammps ctrj
        * frame
        * framerate
    """
    v = ice.vertices()
    v = v.trj_to_vertices(ctrj.loc[frame])
    currentTime = frame / framerate
    f,ax = plotColloid(trj,frame)
    ax.set_title('t = {} s'.format(currentTime))
    v.display(ax)

def saveAllPaintedFrames(trj,ctrj,frames,framerate,path):
    
    """
        Save all painted frames of a simulation.
        ----------
        Parameters:
        * trj (pd Dataframe): lammps trj
        * ctrj (pd Dataframe): lammps ctrj
        * frames: list of frames to export
        * framerate
    """
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
    
def get_colloids_from_ctrj(ctrj,particle,trap,particle_radius,a,N):

    """
        Reconstruct the colloidal ice object from simulation parameters.
        ----------
        Parameters:
        * ctrj (pd Dataframe): lammps ctrj without "t" and "type" columns
        * particle: particle simulation object
        * trap: trap simulation object
        * particle_radius
        * a: lattice constant
        * N: system size
    """
    centers = [ row[:3].to_list() * ureg.um for _,row in ctrj.iterrows()]
    directions = [ row[3:6].to_list() * ureg.um for _,row in ctrj.iterrows()]
    arrangement = {
        "centers" : centers,
        "directions" : directions
    }

    col = ice.colloidal_ice(arrangement, particle, trap,
            height_spread = 0, 
            susceptibility_spread = 0.1,
            periodic = True)
    col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a
    
    return col

def get_colloids_from_ctrj2(ctrj,params):

    """
        Reconstruct the colloidal ice object from simulation parameters.
        Notice that this version uses a params dict.
        ----------
        Parameters:
        * ctrj (pd Dataframe): lammps ctrj without "t" and "type" columns
        * params: Dictionary with all simulation parameters
    """
    particle = params["particle"]
    trap = params["trap"]
    particle_radius = params["particle_radius"]
    a = params["lattice_constant"]
    N = params["size"]

    centers = [ row[:3].to_list() * ureg.um for _,row in ctrj.iterrows()]
    directions = [ row[3:6].to_list() * ureg.um for _,row in ctrj.iterrows()]
    arrangement = {
        "centers" : centers,
        "directions" : directions
    }

    col = ice.colloidal_ice(arrangement, particle, trap,
            height_spread = 0, 
            susceptibility_spread = 0.1,
            periodic = True)
    col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a
    
    return col

def count_vertices_single(vrt, column = "type"):
    """
        Counts the vertices of a single frame df.
        ----------
        Parameters:
        * vrt (pd Dataframe): Vertices dataframe.
        * column (optional)
    """
    vrt_count = vrt.groupby(column).count().iloc[:,0]
    types = vrt_count.index.get_level_values(column).unique()
    counts = pd.DataFrame({"counts": vrt_count.values}, index=types)
    counts["fraction"] = counts["counts"] / counts["counts"].sum() 
    return counts


def get_vertices_last_frame(path,last_frame=2399):

    """
        Computes the vertices of only the last frame.
        ----------
        Parameters:
        * path: Filepath where the ctrj file is located
        * last_frame
    """

    ctrj = pd.read_csv(path,index_col=[0,1])

    if last_frame is None:
        last_frame = ctrj.index.get_level_values("frame").unique().max()
    
    ctrj = ctrj.loc[idx[last_frame,:]].drop(["t", "type"],axis=1)

    try:
        v = ice.vertices()
        v = v.trj_to_vertices(ctrj)
    except:
        get_vertices_last_frame(path,last_frame=last_frame-1)
  

    return v.vertices

def get_vertices_at_frame(ctrj,frame):

    """
        Computes the vertices of a specific frame.
        ----------
        Parameters:
        * path: Filepath where the ctrj file is located
        * last: last frame of the simulation
    """

    
    ctrj = ctrj.loc[idx[frame,:]].drop(["t", "type"],axis=1)

    v = ice.vertices()
    v = v.trj_to_vertices(ctrj)
  

    return v.vertices



