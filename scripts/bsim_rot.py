# ========================================================================
# Non Interactive Simulations
# Author: leogabac
# Info
# This script runs a colloidal-ice simulation several times
# and saves trj and ctr in the data folder
# ========================================================================

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

sys.path.insert(0, '../../icenumerics/')
import icenumerics as ice

ureg = ice.ureg
idx = pd.IndexSlice


# Setup of the experiment
def main():
    sp = ice.spins()
    # Initialize some parameters
    trapSep = 10*ureg.um
    particleRadius = 5*ureg.um
    totalTime = 60*ureg.s

    L = 30*ureg.um
    N = 10

    # Creating colloid stuff
    sp.create_lattice("square",[N,N],lattice_constant=L, border="periodic")

    particle = ice.particle(radius = particleRadius,
                susceptibility = 0.0576,
                diffusion = 0.125*ureg.um**2/ureg.s,
                temperature = 300*ureg.K,
                density = 1000*ureg.kg/ureg.m**3)

    trap = ice.trap(trap_sep = trapSep,
                height = 4*ureg.pN*ureg.nm,
                stiffness = 1e-3*ureg.pN/ureg.nm)

    col = ice.colloidal_ice(sp, particle, trap,
                            height_spread = 0, 
                            susceptibility_spread = 0.1,
                            periodic = True)
    
    col.randomize()
    col.region = np.array([[0,0,-3*(particleRadius/L/N).magnitude],[1,1,3*(particleRadius/L/N).magnitude]])*N*L

    world = ice.world(
    field = 1*ureg.mT,
    temperature = 300*ureg.K,
    dipole_cutoff = 200*ureg.um)

    framespersec = 20*ureg.Hz;
    dt = 10*ureg.ms

    col.simulation(world,
                name = "test",
                include_timestamp = False,
                targetdir = r".",
                framerate = framespersec,
                timestep = dt,
                run_time = totalTime,
                output = ["x","y","z","mux","muy","muz"])

    # Field
    col.sim.field.fieldx = "v_Bmag*sin(PI/2/60*time/1e6)"
    col.sim.field.fieldy = "0"
    col.sim.field.fieldz = "v_Bmag*cos(PI/2/60*time/1e6)"

    col.run_simulation();

    col.load_simulation()

    return col


runs = 20

for i in range(1,runs+1):
    print("===== Experiment number " + str(i) + "=====" )
    col = main()
    filename = "trj" + str(i) + ".csv"
    col.trj.to_csv('../data/' + filename)
    print('Saved trj')
    filename = "ctrj" + str(i) + ".csv"
    trj = ice.get_ice_trj(col.trj, bounds = col.bnd)
    trj.to_csv('../data/' + filename)
    print('Saved centered trj')
