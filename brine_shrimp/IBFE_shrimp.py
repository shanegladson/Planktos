#! /usr/bin/env python3
'''
Read IBFE data for 3D cylinder model. Reproduce flowtank experiment with
random walk agents.
'''

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Planktos, data_IO

# Whether or not to show the cylinders based on the mesh data
PLOT_MODEL = True

# Intialize environment
envir = Planktos.environment()

############     Import IBMAR data on flow and extend domain     ############

print('Reading VTK data. This will take a while...')
envir.read_IBAMR3d_vtk_data('data/RAW_10x20x2cm_8_5s.vtk')
print('Domain set to {} mm.'.format(envir.L))
print('Flow mesh is {}.'.format(envir.flow[0].shape))
print('-------------------------------------------')
# Domain should be 80x320x80 mm
# Flow mesh is 256x1024x256, so resolution is 5/16 mm per unit grid
# Model sits (from,to): (2.5,77.5), (85,235), (0.5,20.5)
# Need: 182 mm downstream from model to capture both zones

# Extend domain downstream so Y is 440mm total length
# Need to extend flow mesh by 120mm/(5/16)=384 mesh units
envir.extend(y_plus=384)
print('Domain extended to {} mm'.format(envir.L))
print('Flow mesh is {}.'.format(envir.flow[0].shape))
# NOW:
# Domain should be 80x460x80 mm
# Model sits (from,to): (2.5,77.5), (85,235), (0.5,20.5)
model_bounds = (2.5,77.5,85,235,0,20.5)
print('-------------------------------------------')


############      Add swarm right in front of model      ############
s = envir.add_swarm(swarm_s=1, init='point', pos=(40,84,1))

# Specify amount of jitter (mean, covariance)
# Set sigma**2 as 0.5cm**2/sec = 50mm**2/sec, sigma~7mm
# (sigma**2=2*D, D for brine shrimp given in Kohler, Swank, Haefner, Powell 2010)
shrimp_walk = ([0,0,0], 50*np.eye(3))


############ Move the swarm according to the prescribed rules above ############

print('Moving swarm...')
dt = 0.1
last_time = 10 #240
for ii in range(last_time):
    s.move(dt, shrimp_walk)


############ Gather data about flow tank observation area ############
# Observation squares are 2cm**2 until the top
# Green zone begins right where the model ends, at y=235. Ends at y=275 mm.
g_bounds = (235, 275)
# Blue zone begins 14.2-4=10.2 cm later, at y=377 mm. Ends at y=417 mm.
b_bounds = (377, 417)
# Each cell is 2cm x 2cm
cell_size=20

gy_cells = [(g_bounds[0]+20*k, g_bounds[0]+20*k+20) for k in np.arange(8)%2]
by_cells = [(b_bounds[0]+20*k, b_bounds[0]+20*k+20) for k in np.arange(8)%2]
z_cells = [(20*k, 20*k+20) for k in np.arange(8)//2]

# Tabulate counts for each cell
print('Obtaining counts...')
g_cells_cnts = list()
b_cells_cnts = list()
for shrimps in s.pos_history: # each time point in history
    g_cells_cnts.append([])
    b_cells_cnts.append([])
    for gcell, zcell in zip(gy_cells, z_cells): # each green cell
        g_cells_cnts[-1].append(0)
        for ii in range(shrimps.shape[0]): # each shrimp
            if gcell[0] <= shrimps[ii,1] < gcell[1] and\
               zcell[0] <= shrimps[ii,2] < zcell[1]:
               g_cells_cnts[-1][-1] += 1
    for bcell, zcell in zip(by_cells, z_cells): # each blue cell
        b_cells_cnts[-1].append(0)
        for ii in range(shrimps.shape[0]): # each shrimp
            if bcell[0] <= shrimps[ii,1] < bcell[1] and\
               zcell[0] <= shrimps[ii,2] < zcell[1]:
               b_cells_cnts[-1][-1] += 1

# append last (current) time point
g_cells_cnts.append([])
b_cells_cnts.append([])
for gcell, zcell in zip(gy_cells, z_cells): # each green cell
    g_cells_cnts[-1].append(0)
    for ii in range(s.positions.shape[0]): # each shrimp
        if gcell[0] <= shrimps[ii,1] < gcell[1] and\
            zcell[0] <= shrimps[ii,2] < zcell[1]:
            g_cells_cnts[-1][-1] += 1
for bcell, zcell in zip(by_cells, z_cells): # each blue cell
    b_cells_cnts[-1].append(0)
    for ii in range(s.positions.shape[0]): # each shrimp
        if bcell[0] <= shrimps[ii,1] < bcell[1] and\
            zcell[0] <= shrimps[ii,2] < zcell[1]:
            b_cells_cnts[-1][-1] += 1


############ Create plot using all time points ############
plot_order = [7,8,5,6,3,4,1,2]
g_cells_cnts = np.array(g_cells_cnts)
b_cells_cnts = np.array(b_cells_cnts)
time_mesh = envir.time_history
time_mesh.append(envir.time)

plt.figure(figsize=(4.8, 6.4))
for n, plot in enumerate(plot_order):
    plt.subplot(4,2,plot)
    plt.plot(time_mesh, g_cells_cnts[:,n])
    plt.xlabel('time (s)')
    plt.ylabel('counts')
    plt.title('Cell number {}'.format(n))
plt.savefig('green_cell_plots.pdf')

plt.figure(figsize=(4.8, 6.4))
for n, plot in enumerate(plot_order):
    plt.subplot(4,2,plot)
    plt.plot(time_mesh, b_cells_cnts[:,n])
    plt.xlabel('time (s)')
    plt.ylabel('counts')
    plt.title('Cell number {}'.format(n))
plt.savefig('blue_cell_plots.pdf')
        

############ This bit plots the model as a translucent rectangle ############

def plot_model_rect(ax3d, bounds):
    '''Plot the model as a translucent rectangular prism

    Arguments:
        ax3d: Axes3D object
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)'''
    x_range = bounds[0:2]
    y_range = bounds[2:4]
    z_range = bounds[4:]

    xx, yy = np.meshgrid(x_range, y_range)
    ax3d.plot_wireframe(xx, yy, z_range[0]*np.ones_like(xx), color="lightgray")
    ax3d.plot_surface(xx, yy, z_range[0]*np.ones_like(xx), color="lightgray", alpha=0.2)
    ax3d.plot_wireframe(xx, yy, z_range[1]*np.ones_like(xx), color="lightgray")
    ax3d.plot_surface(xx, yy, z_range[1]*np.ones_like(xx), color="lightgray", alpha=0.2)

    yy, zz = np.meshgrid(y_range, z_range)
    ax3d.plot_wireframe(x_range[0]*np.ones_like(yy), yy, zz, color="lightgray")
    ax3d.plot_surface(x_range[0]*np.ones_like(yy), yy, zz, color="lightgray", alpha=0.2)
    ax3d.plot_wireframe(x_range[1]*np.ones_like(yy), yy, zz, color="lightgray")
    ax3d.plot_surface(x_range[1]*np.ones_like(yy), yy, zz, color="lightgray", alpha=0.2)

    xx, zz = np.meshgrid(x_range, z_range)
    ax3d.plot_wireframe(xx, y_range[0]*np.ones_like(xx), zz, color="lightgray")
    ax3d.plot_surface(xx, y_range[0]*np.ones_like(xx), zz, color="lightgray", alpha=0.2)
    ax3d.plot_wireframe(xx, y_range[1]*np.ones_like(xx), zz, color="lightgray")
    ax3d.plot_surface(xx, y_range[1]*np.ones_like(xx), zz, color="lightgray", alpha=0.2)


############ This bit plots the sample areas ############

def plot_sample_areas(ax3d, g_range, b_range, x_range):
    '''Plot the sample areas on the base of the domain

    Arguments:
        ax3d: Axes3D object
        g_bounds: (ymin, ymax)
        b_bounds: (ymin, ymax)
        x_bounds: (xmin, xmax)'''
    xx, yy = np.meshgrid(x_range, g_range)
    ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color="mediumseagreen")
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), color="mediumseagreen", alpha=0.2)
    
    xx, yy = np.meshgrid(x_range, b_range)
    ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color="cornflowerblue")
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), color="cornflowerblue", alpha=0.2)


if PLOT_MODEL:
    # Add model to plot list
    envir.plot_structs.append(plot_model_rect)
    envir.plot_structs_args.append((model_bounds,))
    # Add sample areas to plot list
    envir.plot_structs.append(plot_sample_areas)
    envir.plot_structs_args.append((g_bounds, b_bounds, (0, envir.L[0])))


############              Plot!               #############
print('Creating movie...')
s.plot_all('brine_shrimp_IBFE.mp4', fps=10)
#s.plot_all()
