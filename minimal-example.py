import pyviability as viab
from pyviability import helper

import datetime as dt
import functools as ft
from matplotlib import pyplot as plt
import numba as nb
import numpy as np

####################################
# Terms
#
# rescaled: refers to the state space and dynamics mapped to [0, 1]^2 (or the unit n-cube in n dimensions)
# rhs: right-hand-side function of an ordinary differential equation
# run function: The Saint-Pierre Algorithm needs a function that "steps" from one state to the next one. I call this run-function here.
# sunny: the function that returns whether a state (or an array of states) is in the sunny region or not
####################################

import examples.PopulationAndResourceModel as prm
# imports `rhs` and `sunny` function

# set the x and y size of the state space
xlim, ylim = boundaries = [[0, 35000], [0, 18000]]

# give the parameters
default_parameters = dict(phi=4, r=0.04, gamma=4 * 10 ** (-6), delta=-0.1, kappa=12000)
management_parameters = dict(phi=4, r=0.04, gamma=2.8 * 10 ** (-6), delta=-0.1, kappa=12000)

# set parameters for sunny and rescale it on [0, 1]^2
sunny = ft.partial(prm.easter_sunny, xMinimal=1000, yMinimal=3000)

# generate the rescaled grid
# scaling factor and offset are for backscaling of the grid to the original boundaries
# ignore the last arguments `_`, it exists for legacy reasons only
grid, scaling_factor, offset, _ = viab.generate_grid(boundaries,
                                                          n0=200
)
# generate the array that contains the states
# meaning of the state integers copied from libviability.py
# UNSET = 0
# SHELTER = 1
# GLADE = 2
# LAKE = 3
# SUNNY_UP = 4
# DARK_UP = 5
# BACKWATERS = 6
# SUNNY_DOWN = 7
# DARK_DOWN = 8
# SUNNY_EDDIES = 9
# DARK_EDDIES = 10
# SUNNY_ABYSS = 11
# DARK_ABYSS = 12
# TRENCH = 13
states = np.zeros(grid.shape[:-1], dtype=np.int16)

# create the run functions for each of the dynamics
# note, that both transofrmations described in the paper are applied here
# If there is no necessity for you to touch this, then I recommend not trying to understand it (:
default_run = viab.make_run_function(
    nb.jit(prm.easter_rhs, nopython=True), # jit the function for speed ... might not have too much effect here as the call overhead will be the worst, I guess
    helper.get_ordered_parameters(prm.easter_rhs, default_parameters), # see doc string
    offset,
    scaling_factor
)
management_run = viab.make_run_function(
    nb.jit(prm.easter_rhs, nopython=True),
    helper.get_ordered_parameters(prm.easter_rhs, management_parameters),
    offset,
    scaling_factor
)

# set parameters for sunny and rescale it on [0, 1]^2
sunny = viab.scaled_to_one_sunny(ft.partial(prm.easter_sunny, xMinimal=1000, yMinimal=3000), offset, scaling_factor)

start_time = dt.datetime.now()
# run the topology classification
viab.topology_classification(grid, states, [default_run], [management_run], sunny,
                             compute_eddies=False, # by default eddies are not computed because it takes ages. If needed, set to `True` here.
                             verbosity=0,
)
print("classification run time: {!s}".format(dt.datetime.now() - start_time))

fig = plt.figure(figsize=(10, 10), tight_layout=True)

viab.plot_areas(grid, states)

plt.show()