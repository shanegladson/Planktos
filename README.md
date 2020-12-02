# Planktos Agent-based Modeling Framework

This project focuses on building a framework for ABMs of plankton and tiny
insects, or other small entities whose effect on the surrounding flow can be
considered negligable. Work is ongoing.

If you use this software in your project, please cite it as:  
Strickland, C. (2018), *Planktos agent-based modeling framework*. https://github.com/mountaindust/Planktos.  
A suggested BibTeX entry is included in the file Planktos.bib.

### Dependencies
- Python 3.5+
- numpy/scipy
- matplotlib 3.x
- pandas
- ffmpeg from conda-forge (not from default anaconda. Use `conda install -c conda-forge ffmpeg`.)
- vtk (if loading vtk data)
- numpy-stl (if loading stl data)
- pytest (if running tests)

If you need to convert data from IBAMR into vtk, you will also need a Python 2.7 environment with numpy and VisIt installed (VisIt's Python API is written in
Python 2.7).

### Tests
All tests can be run by typing `pytest` into a terminal in the base directory.

## Quickstart

There are several working examples in the examples folder, including a 2D simulation, 
a 2D simulation demonstrating individual variation, a 3D simulation, 
a simulation utilizing vtk data obtained from IBAMR which is located in the 
tests/IBAMR_test_data folder, and a simulation demonstrating subclassing of the update_positions method for user-defined agent behavior. There are also two examples demonstrating how to import vertex data (from IB2d and IBAMR), automatically
create immersed meshes out of this data, and then simulate agent movement with these meshes as solid boundaries which the agents respect. More examples will be added as functionality is added. To run any of these examples, change your working directory 
to the examples directory and then run the desired script.

When experimenting with different agent behavior than what is prescribed in the
swarm class by default (e.g. different movement rules), it is strongly suggested 
that you subclass swarm (found in framework.py) in an appropriate subfolder. That 
way, you can keep track of everything you have tried and its outcome. 

Research that utilizes this framework can be seen in:  
- Ozalp, Miller, Dombrowski, Braye, Dix, Pongracz, Howell, Klotsa, Pasour, 
Strickland (2020). Experiments and agent based models of zooplankton movement 
within complex flow environments, *Biomimetics*, 5(1), 2.

## API
Class: environment
   
- Properties
    - `L` list, length 2 (2D) or 3 (3D) with length of each domain dimension
    - `bndry` list of tuples giving the boundary conditions (strings) of each dimension
    - `flow_times` list of times at which fluid data is specified
    - `flow_points` list of spatial points (as tuples) where flow data is specified. These are assumed to be the same across time points
    - `flow` fluid flow velocity data. List of length 2 or 3, where each entry is an ndarray giving fluid velocity in the x, y, (and z) directions. If the flow is time-dependent, the first dimension of each ndarray is time, with the others being space. This implies that the velocity field must be specified on a regular spatial grid, and it is also assumed that the outermost points on the grid are on the boundary for interpolation purposes.
    - `swarms` list of swarm objects in the environment
    - `time` current time of the simulation
    - `time_history` history of time points simulated
    - `ibmesh` Nx2x2 or Nx3x3 ndarray of mesh elements, given as line segment vertices (2D) or triangle vertices (3D)
    - `max_meshpt_dist` max distance between two vertices in ibmesh. Used internally.
    - `struct_plots` additional items (structures) can be plotted along with the simulation by storing function handles in this list. The plotting routine will call each of them in order, passing the main axes handle as the first argument
    - `struct_plots_args` list of tuples supplying additional arguments to be passed to the struct_plots functions
    - `tiling` if the domain has been tiled, the amount of tiling is recorded here (x,y)
    - `orig_L` length of each domain dimension before tiling
    - `fluid_domain_LLC` if fluid was imported from data, the spatial coordinates of the lower left corner of the original data. This is used internally to aid subsequent translations
    - `a` optional parameter for storing porous region height. If specified, the plotting routine will add some random grass with that height.
    - `rho` optional parameter for storing dynamic fluid velocity
    - `mu` optional parameter for dynamic viscosity
    - `g` acceleration due to gravity (9.80665 m/s**2)
- Methods
    - `set_brinkman_flow` Given several (possibly time-dependent) fluid variables, calculate Brinkman flow on a regular grid with a given resolution and set that as the environment's fluid  velocity. Capable of handling both 2D and 3D domains.
    - `set_two_layer_channel_flow` Apply wide-channel flow with vegetation layer according to the two-layer model described in Defina and Bixio (2005) "Vegetated Open Channel Flow".
    - `set_canopy_flow` Apply flow within and above a uniform homogenous canopy according to the model described in Finnigan and Belcher (2004), "Flow over a hill covered with a plant canopy".
    - `read_IB2d_vtk_data` Read in 2D fluid velocity data from IB2d and set the environment's flow variable.
    - `read_IBAMR3d_vtk_data` Read in 3D fluid velocity data from vtk files obtained from IBAMR. See read_IBAMR3d_py27.py for converting IBAMR data to vtk format using VisIt.
    - `read_IBAMR3d_vtk_dataset` Read in multiple vtk files with naming scheme
    IBAMR_db_###.vtk where ### is the dump number (automatic format when using
    read_IBAMR3d_py27.py) for time varying flow.
    - `read_npy_vtk_data` Read in 2D numpy vtk flow data generated by IB2d and set
    the environment's flow variable.
    - `read_comsol_vtu_data` Read in 2D or 3D fluid velocity data from vtu files (either .vtu or .txt) obtained from COMSOL. This data must be on a regular grid and include a Grid specification at the top.
    - `read_stl_mesh_data` Reads in 3D immersed boundary data from an ascii or binary stl file. Only static meshes are supported.
    - `read_IB2d_vertex_data` Read in 2D immersed boundary data from a .vertex file used in IB2d. Will assume that vertices closer than half (+ epsilon) the Eulerian mesh resolution are connected linearly. Only static meshes are supported.
    - `read_vertex_data_to_convex_hull` Read in 2D or 3D vertex data from a vtk file or a .vertex file and create a structure by computing the convex hull. Only static meshes are supported.
    - `tile_flow` Tile the current fluid flow in the x and/or y directions. It is assumed that the flow is roughly periodic in the direction(s) specified - no checking will be done, and no errors thrown if not.
    - `extend` Extend the domain by duplicating the boundary flow a number of times in a given (or multiple) directions. Good when there is fully resolved fluid \
    flow before/after or on the sides of a structure.
    - `add_swarm` Add or initialize a swarm into the environment
    - `move_swarms` Call the move method of each swarm in the environment
    - `set_boundary_conditions` Check that each boundary condition is implemented before setting the bndry property.
    - `reset` Resets environment to time=0. Swarm history will be lost, and all swarms will maintain their last position. This is typically called automatically if the fluid flow has been altered by another method. If rm_swarms=True, remove all swarms.
    
Class: swarm

- Properties
    - `positions` list of current spatial positions, one for each agent
    - `pos_history` list of previous "positions" lists
    - `velocity` list of current velocities, one for each agent (for use in
    projectile motion)
    - `acceleration` list of current accelerations, one for each agent (for use
    in projectile motion)
    - `envir` environment object that this swarm belongs to
    - `rndState` random number generator (for reproducability)
    - `shared_props` properties shared by all members of the swarm. Includes:
        - `mu` default mean for Gaussian walk (zeros)
        - `cov` covariance matrix for Gaussian walk (identity matrix)
        - `char_L` characteristic length for agents' Reynolds number (if provided)
        - `phys` dictionary of physical attributes (if provided, for use in projectile motion)
    - `props` Pandas DataFrame of properties that vary by individual agent.
- Methods
    - `change_envir` manages a change from one environment to another
    - `calc_re` Calculate the Reynolds number based on environment variables.
    Requires rho and mu to be set in the environment, and char_L to be set in swarm
    - `move` move each agent in the swarm. Do not override: see update_positions.
    - `update_positions` updates the agents' physical locations. OVERRIDE THIS WHEN SUBCLASSING!
    - `get_prop` return the property requested as either a single value (if shared) or a numpy array
    - `add_prop` add a new property and check that it isn't in both props and shared_props
    - `get_fluid_drift` get the fluid velocity at each agent's position via interpolation
    - `get_fluid_gradient` get the gradient of the magnitude of the fluid velocity
    at each agent's position via interpolation
    - `get_projectile_motion` Return acceleration using equations of projectile motion. Includes drag, inertia, and background flow velocity. Does not include gravity.
    - `apply_boundary_condition` method used to enforce the boundary conditions during a move
    - `plot` plot the swarm's current position or a previous position at the time provided
    - `plot_all` plot all of the swarm's positions up to the current time

