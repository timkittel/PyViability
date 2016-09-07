
from __future__ import print_function, division, generators

import PTopologyL as topo

import helper

import periodic_kdtree as periodkdt

import numpy as np
import numpy.linalg as la
import math

import numba as nb

import scipy.integrate as integ
import scipy.spatial as spat

import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings as warn

import itertools as it

# raise the odeing warning as an error because it indicates that we are at a
# fixed point (or the grid is not fine enough)
warn.filterwarnings("error", category=integ.odepack.ODEintWarning)

# these are automatically set during grid generation but need to be manually set
# when using own grid
BOUNDS_EPSILON = None  # should be set during grid Generation
STEPSIZE = None

# some constants so the calculation does end
# MAX_EVOLUTION_NUM = 20
MAX_ITERATION_EDDIES = 10
DEBUGGING = 0
GENERAL_VERBOSE = 0


# The ones below are just used by the default pre-calculation hook and the
# default state evaluation. They are just here so they are not used for
# something else.
KDTREE = None
STATES = None
BOUNDS = None
BASIS_VECTORS = None
BASIS_VECTORS_INV = None
OUT_OF_BOUNDS = None
COORDINATES = None
ALL_NEIGHBORS_DISTANCE = None

# ---- states ----
# encode the different states as integers, so arrays of integers can be used
# later in numpy arrays (which are very fast on integers)
# None state should never be used as it is used to indicate out of bounds
REGIONS = ["UNSET", "SHELTER", "GLADE", "LAKE", "SUNNY_UP", "DARK_UP", "BACKWATERS", "SUNNY_DOWN", "DARK_DOWN", "SUNNY_EDDIES", "DARK_EDDIES", "SUNNY_ABYSS", "DARK_ABYSS", "TRENCH"]
UNSET = 0
SHELTER = 1
GLADE = 2
LAKE = 3
SUNNY_UP = 4
DARK_UP = 5
BACKWATERS = 6
SUNNY_DOWN = 7
DARK_DOWN = 8
SUNNY_EDDIES = 9
DARK_EDDIES = 10
SUNNY_ABYSS = 11
DARK_ABYSS = 12
TRENCH = 13

assert set(REGIONS).issubset(globals())

OTHER_STATE = 14  # used for computation reasons only
OUT_OF_BOUNDS_STATE = 15


# ---- Colors ----
# identify the states with the corresponding colors in order to be consistent
# with the color definitions from the original paper
COLORS = {
        UNSET: "blue",
        -SHELTER: "blue",
        -GLADE: "blue",
        SHELTER: topo.cShelter,
        GLADE: topo.cGlade,
        LAKE: topo.cLake,
        SUNNY_UP: topo.cSunnyUp,
        DARK_UP: topo.cDarkUp,
        BACKWATERS: topo.cBackwaters,
        SUNNY_DOWN: topo.cSunnyDown,
        DARK_DOWN: topo.cDarkDown,
        SUNNY_EDDIES: topo.cSunnyEddie,
        DARK_EDDIES: topo.cDarkEddie,
        SUNNY_ABYSS: topo.cSunnyAbyss,
        DARK_ABYSS: topo.cDarkAbyss,
        TRENCH: topo.cTrench,
        }


def Delta_series(Delta_0, dim):
    q = Delta_0 ** 2
    return [np.sqrt(q * (n+2) / (2*n + 2)) for n in range(dim)]


def p_series(Delta_0, dim):
    """\
    returns the p vectors as an array p[i, j] where j enumerates the \
    vector (and thus dimension) and i the component"""
    p_all = np.zeros((dim, dim))
    for n, Delta_n in enumerate(Delta_series(Delta_0, dim)):
        p_all[:n, n] = np.sum(p_all[:n, :n], axis=1) / (n+1)
        p_all[n, n] = Delta_n
    return p_all


def generate_grid(boundaries, n0, grid_type, periodicity=[], verbosity=True):
    global MAX_NEIGHBOR_DISTANCE, BOUNDS_EPSILON, STEPSIZE, ALL_NEIGHBORS_DISTANCE, x_step

    assert grid_type in ["simplex-based", "orthogonal"], "unkown grid type '{!s}'".format(grid_type)

    boundaries = np.asarray(boundaries)
    periodicity = np.asarray(periodicity)

    dim = boundaries.shape[0]
    offset = boundaries[:, 0].astype(float)
    scaling_factor = boundaries[:,1] - boundaries[:,0]

    if not periodicity.size:
        periodicity = - np.ones((dim,))

    assert periodicity.shape == (dim,), "given boundaries do not match periodicity input"

    periodicity_bool = (periodicity > 0)

    #############################
    # generate the basic grid
    #############################
    grid_prep_aperiodic = np.linspace(0, 1, n0)
    grid_prep_periodic = np.linspace(0, 1, n0-1, endpoint=False)
    # the last point is not set as it would be the same as the first one in
    # a periodic grid
    grid_args = [grid_prep_periodic if periodicity_bool[d] else grid_prep_aperiodic for d in range(dim)]

    # create the grid
    grid = np.asarray(np.meshgrid(*grid_args))

    # move the axis with the dimenion to the back
    grid = np.rollaxis(grid, 0, dim + 1)

    # flattening the array
    grid = np.reshape(grid, (-1, dim))

    x_step = grid_prep_periodic[1]

    if grid_type in ["orthogonal"]:

        scaling_vectors = np.diag(1 / scaling_factor)
        assert x_step == grid_prep_aperiodic[1], "bug?"
        MAX_NEIGHBOR_DISTANCE = 1.5 * x_step
        BOUNDS_EPSILON = 0.1 * x_step
        ALL_NEIGHBORS_DISTANCE = 2 * np.sqrt(dim) * x_step + BOUNDS_EPSILON
        # print("x_step", x_step)
        # print("ALL_NEIGHBORS_DISTANCE", ALL_NEIGHBORS_DISTANCE)
        # assert False
        # ALL_NEIGHBORS_DISTANCE = np.sqrt(dim) * x_step + BOUNDS_EPSILON
        # STEPSIZE = ALL_NEIGHBORS_DISTANCE
        STEPSIZE = 1.5 * x_step
        # STEPSIZE = 2.5 * x_step

    elif grid_type in ["simplex-based"]:
        if np.any(periodicity_bool[1:]):
            # the periodic binary tree can handle orthogonal periodicity only
            # because the first basis vector for the simplex based grid is
            # parallel to the x-axis, orthogonality in the first dimension is
            # okai and the if statement above tests only periodicity_bool[1:]
            raise NotImplementedError("The generation of the simplex-based grid is not yet compatible with periodic state spaces (except in the first dimension).")

        basis_vectors = p_series(1., dim)
        scaling_vectors = basis_vectors / scaling_factor[None, :]

        grid = np.tensordot(grid, basis_vectors, axes=[(1,), (1,)])

        # # Delta_0 is the initial distance, ie. the one in the lowest dimension
        # Delta_0 = 1 / n0
        # # calculate the spacing in each dimension
        # Delta_all = np.array(list(Delta_series(Delta_0, dim)))
#
        # # n_all is the number of points in each dimension
        # n_all = (1 / Delta_all).astype(np.int)
        # n_all -= n_all % 2 # else the modulo below could shift the grid
#
        # # boundaries of the generated grid are [0, x_max[0]] x [0, x_max[1]] x ...
        # # x_max is in general !=1 because n_all was cut off to integers
        # x_max = n_all * Delta_all
        # # correct scaling factor a bit because the grid has _not_ [0,1]^dim as boundaries
        # scaling_factor = scaling_factor / x_max
#
        # # generate the base vectors
        # p_all = p_series(Delta_0, dim)
#
        # # generate a pre-grid that contains the coefficients for each base vector
        # pre_grid = np.array(np.meshgrid(*[np.arange(n) for n in n_all]))
#
        # # generate the actual grid
        # grid = np.tensordot(p_all, pre_grid, axes=[(1,), (0)])
        # grid = np.reshape(grid, (dim, -1))
#
        # # move everything within the boundaries given by x_max
        # grid %= x_max[:, np.newaxis]
#
        # # move the axis with the dimenion to the back
        # grid = np.rollaxis(grid, 0, 2)

        # when recursively going through, then add the direct neighbors only
        MAX_NEIGHBOR_DISTANCE = 1.01 * x_step
        # x_step = Delta_0 # Delta_0 is side length of the simplices
        BOUNDS_EPSILON = 0.1 * x_step
        STEPSIZE = 2.5 * x_step # seems to be correct
        ALL_NEIGHBORS_DISTANCE = la.norm(np.sum(basis_vectors, axis=1)) * x_step + BOUNDS_EPSILON

    if verbosity:
        print("created {:d} points".format(grid.shape[0]))

    return grid, scaling_vectors, offset, x_step


def _generate_viability_single_point(evolutions, state_evaluation, use_numba=False, nb_nopython=False):
    if use_numba:
        raise NotImplementedError("numba usage doesn't really make sense here, because KDTREE cannot be numba jitted")

        # isdispatcher = lambda x: isinstance(x, nb.dispatcher.Dispatcher)
        # if not (isdispatcher(state_evaluation) and all(map(isdispatcher, evolutions))):
            # warn.warn("you want to use numba, but some of the input stuff doesn't seem to be ready for compilation")

    else:

        def _viability_single_point(coordinate_index, coordinates, states,
                                    stop_states, succesful_state, else_state):
            """Calculate whether a coordinate with value 'stop_value' can be reached from 'coordinates[coordinate_index]'."""

            start = coordinates[coordinate_index]
            start_state = states[coordinate_index]

            global DEBUGGING
            # DEBUGGING = True
            # DEBUGGING = (start_state == 1)
            # DEBUGGING = (coordinate_index == (10 * 80 - 64,))
            # DEBUGGING = DEBUGGING and la.norm(start - np.array([1.164, 0.679])) < 0.01
            DEBUGGING = DEBUGGING and start[0] < 0.01
            # DEBUGGING = DEBUGGING and start_state == 1
            # DEBUGGING = DEBUGGING or la.norm(start - np.array([0.1, 0.606])) < 0.02
            # DEBUGGING = True

            if DEBUGGING:
                print()

            for evol_num, evol in enumerate(evolutions):
                traj = evol(start, STEPSIZE)


                final_state = state_evaluation(traj)

                if final_state in stop_states: # and constraint(point) and final_distance < MAX_FINAL_DISTANCE:

                    if DEBUGGING:
                        print( "%i:"%evol_num, coordinate_index, start, start_state, "-->", final_state )
                    return succesful_state

                # run the other evolutions to check whether they can reach a point with 'stop_state'
                if DEBUGGING:
                    print("%i:"%evol_num, coordinate_index, start, start_state, "## break")

            # didn't find an option leading to a point with 'stop_state'
            if DEBUGGING:
                print("all:", coordinate_index, start, start_state, "-->", else_state)
            return else_state
    return _viability_single_point


def _state_evaluation_kdtree_line(traj):
    """deprecated (for now ^^)"""
    start_point = traj[0]
    final_point = traj[-1]
    # print("start_point", start_point)
    # print("final_point", final_point)

    if OUT_OF_BOUNDS:
        # check whether out-of-bounds
        projected_values = np.tensordot(BASIS_VECTORS_INV, final_point, axes=[(1,), (0,)])
        if np.any( BOUNDS[:,0] > projected_values) or np.any( BOUNDS[:,1] < projected_values ) :  # is the point out-of-bounds?
            if DEBUGGING:
                print("out-of-bounds")
            return None  # "out-of-bounds state"

    # assert False, "out of bounds doesn't seem to work?"

    # if not out-of-bounds, determine where it went to

    # print("ALL_NEIGHBORS_DISTANCE", ALL_NEIGHBORS_DISTANCE)
    # print("start_point", start_point)
    neighbor_indices = KDTREE.query_ball_point(start_point, ALL_NEIGHBORS_DISTANCE)
    # print("neighbor_indices", neighbor_indices)
    neighbors = KDTREE.data[neighbor_indices]
    # print("neighbors", neighbors)
    # print("KDTREE.data[2]", KDTREE.data[2])
    # print("diff", neighbors - KDTREE.data[2])
    # print("norm(diff)", la.norm(neighbors - KDTREE.data[2], axis=-1))
    if hasattr(KDTREE, "bounds"):
        if DEBUGGING:
            print("bounds", KDTREE.bounds)
        bool_bounds = (KDTREE.bounds > 0)
        newbounds = KDTREE.bounds[bool_bounds]
        _start_point = np.copy(start_point)
        _start_point[bool_bounds] = start_point[bool_bounds] % newbounds
    else:
        _start_point = start_point
    _start_point_local_index = np.argmax(np.logical_and.reduce(np.isclose(neighbors, _start_point[None, :]), axis=1))
    _start_point_global_index = neighbor_indices.pop(_start_point_local_index)
    neighbors = np.delete(neighbors, _start_point_local_index, axis=0)
    del _start_point_local_index

    if DEBUGGING:
        # print("start_point", start_point)
        # print(neighbors.shape)
        # print("neighbors")
        # print(neighbors)
        plt.plot(start_point[0], start_point[1], color = "black",
                linestyle = "", marker = ".", markersize = 40 ,zorder=0)
        plt.plot(_start_point[0], _start_point[1], color = "black",
                linestyle = "", marker = ".", markersize = 40 ,zorder=0)
        plt.plot(neighbors[:, 0], neighbors[:, 1], color = "blue",
                linestyle = "", marker = ".", markersize = 50 ,zorder=0)

    a = final_point - start_point
    if np.allclose(a, 0):
        closest_index = _start_point_global_index
    else:
        # print("neighbors", neighbors)
        b = neighbors - start_point[None, :]

        # take care of the periodic boundaries
        if hasattr(KDTREE, "bounds"):
            newbounds = np.ones_like(KDTREE.bounds)
            # newbounds = np.array(KDTREE.bounds)
            # newbounds[newbounds <= 0] = np.infty
            shiftbounds = 0.5 * np.ones_like(newbounds)
            warn.warn("using cheap fix for periodic boundary here")
            # if DEBUGGING:
                # print("a", a)
                # print("b", b)
            a = (a + shiftbounds) % newbounds - shiftbounds
            b = (b + shiftbounds[None, :]) % newbounds[None, :] - shiftbounds[None, :]
        # if DEBUGGING:
            # print("a", a)
            # print("b", b)

        _p = np.tensordot(a, b, axes=[(0,), (1,)])

        distances_to_line_squared = np.sum(b * b, axis=1) - \
            _p * np.abs(_p) / np.dot(a, a)  # the signum of _p is used to find the correct side

        _n_index = np.argmin(distances_to_line_squared)

        closest_index = neighbor_indices[_n_index]

    final_state = STATES[closest_index]

    if DEBUGGING:
        print("evaluation:", start_point, "via", final_point, "to", KDTREE.data[closest_index], "with state", final_state)

    return final_state


@nb.jit
def state_evaluation_kdtree_numba(traj):
    point = traj[-1]

    if OUT_OF_BOUNDS:
        projected_values = np.zeros_like(point)
        dim = point.shape[0]
        for i in range(dim):
            projected_values += BASIS_VECTORS_INV[:, i] * point[i]

        out = False
        for i in range(dim):
            if (BOUNDS[i, 0] > projected_values[i]) or (BOUNDS[i, 1] < projected_values[i]):
                out = True
                break
        if out:
            return OUT_OF_BOUNDS_STATE

    final_distance, tree_index = KDTREE.query(point, 1)
    return STATES[tree_index]


def state_evaluation_kdtree(traj):
    point = traj[-1]
    if OUT_OF_BOUNDS:
        projected_values = np.tensordot(BASIS_VECTORS_INV, point, axes=[(1,), (0,)])
        if np.any( BOUNDS[:,0] > projected_values) or np.any( BOUNDS[:,1] < projected_values ) :  # is the point out-of-bounds?
            if DEBUGGING:
                print("out-of-bounds")
            return OUT_OF_BOUNDS_STATE
    final_distance, tree_index = KDTREE.query(point, 1)
    return STATES[tree_index]


def pre_calculation_hook_kdtree(coordinates, states,
                                is_sunny=None,
                                periodicity=None,
                                grid_type=None,
                                out_of_bounds=True):
    global KDTREE, STATES, BASIS_VECTORS, BASIS_VECTORS_INV, BOUNDS, OUT_OF_BOUNDS
    STATES = states

    dim = np.shape(coordinates)[-1]
    periodicity_bool = (periodicity > 0)

    # check, if there are periodic boundaries and if so, use different tree form
    if np.any(periodicity_bool):
        assert dim == len(periodicity_bool), "Given boundaries don't match with " \
                                                    "dimensions of coordinates. " \
                                                    "Write '-1' if boundary is not periodic!"
        assert (grid_type in ["orthogonal"]) or ((grid_type in ["simplex-based"]) and not np.any(periodicity_bool[1:])),\
            "does PeriodicCKDTREE support the periodicity for your grid?"
        KDTREE = periodkdt.PeriodicCKDTree(periodicity, coordinates)
    else:
        KDTREE = spat.cKDTree(coordinates)

    OUT_OF_BOUNDS = not (out_of_bounds is False)
    if OUT_OF_BOUNDS:
        if out_of_bounds is True:
            out_of_bounds = [[True, True]] * dim
        out_of_bounds = np.asarray(out_of_bounds)

        if out_of_bounds.shape == (dim,):
            out_of_bounds = np.repeat(out_of_bounds[:, None], 2, axis=1)
        assert out_of_bounds.shape == (dim, 2)


        dim = coordinates.shape[-1]
        BOUNDS = np.zeros((dim, 2))

        if grid_type == "orthogonal":
            basis_vectors = np.eye(dim)
        elif grid_type == "simplex-based":
            basis_vectors = p_series(1, dim)

        BASIS_VECTORS = basis_vectors
        BASIS_VECTORS_INV = la.inv(BASIS_VECTORS)

        for d in range(dim):
            if periodicity_bool[d]:
                BOUNDS[d,:] = -np.inf, np.inf
                # this basically means, because of periodicity, the trajectories
                # cannot run out-of-bounds
            else:
                # project the values on the basis vector with a scalar product
                # for that reason, basis vectors need to be normalized
                # projected_values = np.tensordot(coordinates, basis_vectors[:,d], axes=[(1,), (0,)])

                # actually the idea above is correct and this is simply the result
                # combined with the checking whether out-of-bounds should be
                # applied
                BOUNDS[d,:] = np.where(out_of_bounds[d], (-BOUNDS_EPSILON, 1 + BOUNDS_EPSILON), (-np.infty, np.infty))

                # BOUNDS[d,:] = np.min(projected_values) - BOUNDS_EPSILON, np.max(projected_values) + BOUNDS_EPSILON
                # BOUNDS[d,:] = np.min(coordinates[:,d]) - BOUNDS_EPSILON, np.max(coordinates[:,d]) + BOUNDS_EPSILON

        projected_values = np.tensordot(coordinates, BASIS_VECTORS_INV, axes=[(1,), (1,)])
        assert np.all( BOUNDS[None, :,0] < projected_values) \
            and np.all( BOUNDS[None, :,1] > projected_values ),\
            "BOUNDS and coordinates do not fit together, did you set the correct grid_type argument?"


def viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state,
                          evolutions, state_evaluation,
                          use_numba=False,
                          nb_nopython=False):
    """do a single step of the viability calculation algorithm by checking which points stay immediately within the good_states"""

    changed = False

    assert len(coordinates.shape) == 2, "use flattened grid, plz"
    max_index = coordinates.shape[0]

    viability_single_point = _generate_viability_single_point(evolutions, state_evaluation,
                                                              use_numba=use_numba, nb_nopython=nb_nopython)

    for base_index in range(max_index):
        neighbors = [base_index]

        for index in neighbors: # iterate over the base_index and, if any changes happened, over the neighbors, too
            old_state = states[index]

            if old_state == work_state:
                new_state = viability_single_point(index, coordinates, states, good_states,
                                                   succesful_state, bad_state)

                if new_state != old_state:
                    changed = True
                    states[index] = new_state
                    # get_neighbor_indices(index, shape, neighbor_list = neighbors)
                    get_neighbor_indices_via_cKD(index,  neighbor_list=neighbors)

    return changed


def get_neighbor_indices_via_cKD(index, neighbor_list=[]):
    """extend 'neighbor_list' by 'tree_neighbors', a list that contains the nearest neighbors found trough cKDTree"""

    index = np.asarray(index).astype(int)

    tree_neighbors = KDTREE.query_ball_point(KDTREE.data[index].flatten(), MAX_NEIGHBOR_DISTANCE)

    neighbor_list.extend(tree_neighbors)

    return neighbor_list


def get_neighbor_indices(index, shape, neighbor_list = []):
    """append all neighboring indices of 'index' to 'neighbor_list' if they are within 'shape'"""

    index = np.asarray(index)
    shape = np.asarray(shape)

    for diff_index in it.product([-1, 0, 1], repeat = len(index)):
        diff_index = np.asarray(diff_index)
        new_index = index + diff_index

        if np.count_nonzero(diff_index) and np.all( new_index >= 0 ) and np.all( new_index < shape ):
            neighbor_list.append(tuple(new_index))

    return neighbor_list


def viability_kernel(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions,
                     periodic_boundaries = [],
                     pre_calculation_hook = pre_calculation_hook_kdtree,  # None means nothing to be done
                     state_evaluation = state_evaluation_kdtree,
                     ):
    """calculate the viability kernel by iterating through the viability kernel steps
    until convergence (no further change)"""
    # assert coordinates.shape[:-1] == states.shape[:-1], "'coordinates' and 'states' don't match in shape"

    assert "x_step" in globals() # needs to be set by the user for now ... will be changed later
    assert "BOUNDS_EPSILON" in globals() # needs to be set by the user for now ... will be changed later
    # assert "MAX_FINAL_DISTANCE" in globals() # needs to be set by the user for now ... will be changed later
    assert "MAX_NEIGHBOR_DISTANCE" in globals() # needs to be set by the user for now ... will be changed later
    global x_half_step
    x_half_step = x_step/2
    if not "STEPSIZE" in globals():
        global STEPSIZE
        # fix stepsize on that for now if nothing else has been given by the
        # user
        STEPSIZE = 2 * x_step

    if not pre_calculation_hook is None:
        # run the pre-calculation hook (defaults to creation of the KD-Tree)
        pre_calculation_hook(coordinates, states, None, periodic_boundaries)

    # actually only on step is needed due to the recursive checks (i.e. first
    # checking all neighbors of a point that changed state)
    return viability_kernel_step(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, state_evaluation)


def viability_capture_basin(coordinates, states, target_states, reached_state, bad_state, work_state, evolutions,
                    pre_calculation_hook = pre_calculation_hook_kdtree,  # None means nothing to be done
                    state_evaluation = state_evaluation_kdtree,
                    ):
    """reuse the viability kernel algorithm to calculate the capture basin"""

    if work_state in states and any( ( target_state in states for target_state in target_states) ):
        # num_work = np.count_nonzero(work_state == states)
        viability_kernel(coordinates, states, target_states + [reached_state], work_state, reached_state,
                                     work_state, evolutions, pre_calculation_hook = pre_calculation_hook ,state_evaluation = state_evaluation)
        # changed = (num_work == np.count_nonzero(reached_state == states))
    else:
        print("empty work or target set")
        # changed = False
    # all the points that still have the state work_state are not part of the capture basin and are set to be bad_states
    changed = (work_state in states)
    states[ states == work_state ] = bad_state
    return changed

# below are just helper functions


def print_evaluation(states, print_empty_regions=True, print_unknown=True):
    total = states.size
    total_length = str(len(str(total)))
    num_sum = 0
    current_globals = globals()
    print("Evaluation (relative normalized Volume):")
    for region in REGIONS:
        num = np.count_nonzero(states == current_globals[region])
        if print_empty_regions or num > 0:
            num_sum += num
            print(("{:<15}: {:>6.2f}% ( {:>"+total_length+"} )").format(region,  num / total * 100, num))
    print()
    if print_unknown and num_sum != total:
        print(("{:<15}: {:>6.2f}% ( {:>"+total_length+"} )").format("UNKNOWN",  (total - num_sum) / total * 100, total - num_sum))
        print()


def plot_points(coords, states):
    """plot the current states in the viability calculation as points"""

    assert set(states.flatten()).issubset(COLORS)
    for color_state in COLORS:
        plt.plot(coords[ states == color_state , 0], coords[ states == color_state , 1], color = COLORS[color_state],
                 linestyle = "", marker = ".", markersize = 30 ,zorder=0)


def plot_areas(coords, states):
    """plot the current states in the viability calculation as areas"""

    states = states.flatten()
    assert set(states).issubset(COLORS)
    coords = np.reshape(coords, states.shape + (-1,))
    x, y = coords[:,0], coords[:,1]

    color_states = sorted(COLORS)
    cmap = mpl.colors.ListedColormap([ COLORS[state] for state in color_states ])
    bounds = color_states[:1] + [ state + 0.5 for state in color_states[:-1]] + color_states[-1:]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.tripcolor(x, y, states, cmap = cmap, norm = norm, shading = "gouraud")


def make_run_function(rhs,
                      ordered_params,
                      offset,
                      scaling_vector,
                      returning="integration",
                      remember=True,
                      use_numba=True,
                      nb_nopython=True,
                      ):

    S = np.array(scaling_vector, order="C", copy=True)
    Sinv = np.array(la.inv(S), order="C", copy=True)


    # ----------------------------------------

    def rhs_scaled_to_one_PS(y, t):
        """\
        for 2D plotting of the whole phase space plot
        rescales space only, because that should be enough for the phase space plot
        """
        x = offset[:, None, None] + np.tensordot(Sinv, y, axes=[(1,), (0,)])
        val = rhs(x, t, *ordered_params)  # calculate the rhs
        val = np.tensordot(S, val, axes=[(1,), (0,)])
        return val

    # ----------------------------------------

    if use_numba:
        @nb.jit(nopython=nb_nopython)
        def rhs_rescaled(y, t, *args):
            dim = len(y)

            # because of the rescaling to 1 in every dimension
            # transforming y -> x
            x = np.copy(offset)
            for i in range(dim):
                x += Sinv[:, i] * y[i]
                # x += Sinv[:, i] * y[i]
            dx = rhs(x, t, *args)
            # transforming dx -> dy
            dy = np.zeros_like(y)
            for i in range(dim):
                dy += S[:, i] * dx[i]
                # dy += S[:, i] * dx[i]

            # normalization of dy
            dy_norm = 0.
            for i in range(dim):
                dy_norm += dy[i]*dy[i]
            dy_norm = math.sqrt(dy_norm)

            # check whether it's a fixed point
            if dy_norm == 0.:
                return np.zeros_like(dy)

            return dy / dy_norm

    else:
        def rhs_rescaled(y, t, *args):
            x = offset + np.dot(Sinv, y)
            dx = np.dot(S, rhs(x, t, *args)) # calculate the rhs
            return dx / np.sqrt(np.sum(dx ** 2, axis=-1))  # normalize it


    if use_numba:
        @nb.jit(nopython=nb_nopython)
        def normalized_linear_approximation(x, dt):
            xdot = rhs_rescaled(x, dt, *ordered_params)
            traj = np.empty((2, x.shape[0]))
            traj[0] = x
            if np.any(np.isinf(xdot)): # raise artifiially the warning if inf turns up
                # warn.warn("got a inf in the RHS function; assume {!s} to be a stable fixed point and returning the starting point".format(x),
                        # category=RuntimeWarning)
                traj[1] = traj[0]
            else:
                traj[1] = x + xdot*dt
            return traj
    else:
        def normalized_linear_approximation(x, dt):
            xdot = rhs_rescaled(x, dt, *ordered_params)
            traj = np.array([x, x + xdot*dt])
            if np.any(np.isinf(xdot)): # raise artifiially the warning if inf turns up
                warn.warn("got a inf in the RHS function; assume {!s} to be a stable fixed point and returning the starting point".format(x),
                        category=RuntimeWarning)
                traj[1] = traj[0]
                if DEBUGGING:
                    # plot the point, but a bit larger than the color one later
                    plt.plot(p[0], p[1], color = "red",
                        linestyle = "", marker = ".", markersize = 45 ,zorder=0)

            elif DEBUGGING:
                plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=3)
            return traj


    @nb.jit
    def distance_normalized_rhs(x, lam, x0, *args):
        val = rhs_scaled_to_one(x, lam, *args)  # calculate the rhs
        if lam == 0:
            return val / np.sqrt(np.sum( val ** 2, axis=-1) )
        return val * lam / np.sum( (x-x0) * val, axis=-1)


    @helper.remembering(remember = remember)
    def integration(p, stepsize):
        if DEBUGGING:
            integ_time = np.linspace(0, stepsize, 100)
        else:
            integ_time = [0, stepsize]
        try:
            with helper.stdout_redirected():
                traj = integ.odeint(distance_normalized_rhs, p, integ_time,
                                    args=(p,) + ordered_params,
                                    printmessg = False
                                    )
            if np.any(np.isnan(traj[-1])): # raise artifiially the warning if nan turns up
                raise integ.odepack.ODEintWarning("got a nan")
        except integ.odepack.ODEintWarning:
            warn.warn("got an integration warning; assume {!s} to be a stable fixed point and returning the starting point".format(p),
                      category=RuntimeWarning)
            if DEBUGGING:
                # plot the point, but a bit larger than the color one later
                plt.plot(p[0], p[1], color = "red",
                    linestyle = "", marker = ".", markersize = 45 ,zorder=0)
            return np.asarray([p, p])

        if DEBUGGING:
            plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=3)
            return np.asarray([traj[0], traj[-1]])
        else:
            return traj

    if returning == "integration":
        return integration
    elif returning == "linear":
        return normalized_linear_approximation
    elif returning == "PS":
        return rhs_scaled_to_one_PS
    else:
        raise NameError("I don't know what to do with returning={!r}".format(returning))


def scaled_to_one_sunny(is_sunny, offset, scaling_vector):
    S = scaling_vector
    Sinv = la.inv(S)

    def scaled_sunny(grid):
        new_grid = np.tensordot(grid, Sinv, axes=[(1,), (1,)]) + offset[None, :]
        # new_grid = backscaling_grid(grid, scaling_vector, offset)
        # new_grid = np.dot(Sinv, grid) + offset
        val = is_sunny(new_grid)  # calculate the rhs
        return val  # normalize it

    return scaled_sunny


def trajectory_length(traj):
    return np.sum( la.norm( traj[1:] - traj[:-1], axis = -1) )


def trajectory_length_index(traj, target_length):
    lengths = np.cumsum( la.norm( traj[1:] - traj[:-1], axis = -1) )

    if target_length < lengths[-1]:
        return traj.shape[0] # incl. last element
    index_0, index_1 = 0, traj.shape[0] - 1

    while not index_0 in [index_1, index_1 - 1]:
        middle_index = int( (index_0 + index_1)/2 )

        if lengths[middle_index] <= target_length:
            index_0 = middle_index
        else:
            index_1 = middle_index

    return index_1


def backscaling_grid(grid, scaling_vector, offset):
    S = scaling_vector
    Sinv = la.inv(S)
    new_grid = np.tensordot(grid, Sinv, axes=[(1,), (1,)]) + offset[None, :]
    return new_grid


def topology_classification(coordinates, states, default_evols, management_evols, is_sunny,
                            periodic_boundaries = [],
                            upgradeable_initial_states = False,
                            compute_eddies = False,
                            pre_calculation_hook = pre_calculation_hook_kdtree,  # None means nothing to be done
                            state_evaluation = state_evaluation_kdtree_numba,
                            grid_type = "orthogonal",
                            out_of_bounds=True, # either bool or bool array with shape (dim, ) or shape (dim, 2) with values for each boundary
                            ):
    """calculates different regions of the state space using viability theory algorithms"""

    # upgrading initial states to higher is not yet implemented
    if upgradeable_initial_states:
        raise NotImplementedError("upgrading of initally given states not yet implemented")

    coordinates = np.asarray(coordinates)
    states = np.asarray(states)

    grid_size, dim = coordinates.shape
    assert states.shape == (grid_size,), "coordinates and states input doesn't match"

    if periodic_boundaries == []:
        periodic_boundaries = - np.ones(dim)
    periodic_boundaries = np.asarray(periodic_boundaries)

    if not pre_calculation_hook is None:
        # run the pre-calculation hook (defaults to creation of the KD-Tree)
        pre_calculation_hook(coordinates, states,
                             is_sunny=is_sunny,
                             periodicity=periodic_boundaries,
                             grid_type=grid_type,
                             out_of_bounds=out_of_bounds)

    # make sure, evols can be treated as lists
    default_evols = list(default_evols)
    management_evols = list(management_evols)

    all_evols = management_evols + default_evols

    # better remove this and use directly the lower level stuff, see issue #13
    viability_kwargs = dict(
        pre_calculation_hook = None,
        state_evaluation = state_evaluation,
    )

    shelter_empty = False
    backwater_empty = False

    if all_evols:
        if default_evols:
            # calculate shelter
            print('###### calculating shelter')
            states[(states == UNSET) & is_sunny(coordinates)] = SHELTER # initial state for shelter calculation
            # viability_kernel(coordinates, states, good_states, bad_state, succesful_state, work_state, evolutions, **viability_kwargs)
            viability_kernel(coordinates, states, [SHELTER, -SHELTER], UNSET, SHELTER, SHELTER, default_evols, **viability_kwargs)

            if not np.any(states == SHELTER):
                print('shelter empty')
                shelter_empty = True

            if not shelter_empty:
                if management_evols:
                    # calculate glade
                    print('###### calculating glade')

                    states[(states == UNSET) & is_sunny(coordinates)] = SUNNY_UP

                    #viability_capture_basin(coordinates, states, target_states, reached_state, bad_state, work_state, evolutions, **viability_kwargs):
                    viability_capture_basin(coordinates, states, [SHELTER, -SHELTER], GLADE, UNSET, SUNNY_UP, all_evols, **viability_kwargs)
                else:
                    print('###### no management dynamics given, skipping glade')

                # calculate remaining upstream dark and sunny
                print('###### calculating rest of upstream (lake, dark and sunny)')
                states[(states == UNSET)] = DARK_UP
                viability_capture_basin(coordinates, states, [SHELTER, GLADE, -SUNNY_UP, -DARK_UP, -LAKE], SUNNY_UP, UNSET, DARK_UP, all_evols, **viability_kwargs)

                states[~is_sunny(coordinates) & (states == SUNNY_UP)] = DARK_UP

                if management_evols:
                    # calculate Lake
                    print('###### calculating lake')
                    states[is_sunny(coordinates) & (states == SUNNY_UP)] = LAKE
                    viability_kernel(coordinates, states, [SHELTER, GLADE, LAKE, -LAKE], SUNNY_UP, LAKE, LAKE, all_evols, **viability_kwargs)
                else:
                    print('###### no management dynamics given, skipping lake')
        else:
            print('###### no default dynamics given, skipping upstream')

        if management_evols:
            # calculate Bachwater
            print('###### calculating backwater')
            states[is_sunny(coordinates) & (states == UNSET)] = BACKWATERS
            viability_kernel(coordinates, states, [BACKWATERS, -BACKWATERS], UNSET, BACKWATERS, BACKWATERS, all_evols, **viability_kwargs)

            if not np.any(states == BACKWATERS):
                print('backwater empty')
                backwater_empty = True

            if not backwater_empty:
                # calculate remaining downstream dark and sunny
                print('###### calculating remaining downstream (dark and sunny)')
                states[(states == UNSET)] = DARK_DOWN
                viability_capture_basin(coordinates, states, [BACKWATERS, -SUNNY_DOWN, -DARK_DOWN], SUNNY_DOWN, UNSET, DARK_DOWN, all_evols, **viability_kwargs)
                states[~is_sunny(coordinates) & (states == SUNNY_DOWN)] = DARK_DOWN
        else:
            print('###### no management dynamics given, skipping downstream')




        # calculate trench and set the rest as preliminary estimation for the eddies
        print('###### calculating dark Eddies/Abyss')
        states[is_sunny(coordinates) & (states == UNSET)] = SUNNY_EDDIES

        # look only at the coordinates with state == UNSET
        viability_capture_basin(coordinates, states,
                                [SHELTER, GLADE, SUNNY_UP, DARK_UP, LAKE, BACKWATERS, SUNNY_DOWN, SUNNY_EDDIES, SUNNY_ABYSS, -SHELTER, -GLADE, -SUNNY_UP, -DARK_UP , -LAKE, -BACKWATERS, -SUNNY_DOWN, -SUNNY_EDDIES, -SUNNY_ABYSS],
                                DARK_EDDIES, TRENCH, UNSET, all_evols, **viability_kwargs)
        if compute_eddies:

            # the preliminary estimations for sunny and dark eddie are set
            states[(states == SUNNY_EDDIES)] = UNSET
            viability_capture_basin(coordinates, states,
                                    [DARK_EDDIES, -DARK_EDDIES],
                                    SUNNY_EDDIES, SUNNY_ABYSS, UNSET, all_evols, **viability_kwargs)


            for num in range(MAX_ITERATION_EDDIES):
                states[(states == DARK_EDDIES)] = UNSET
                changed = viability_capture_basin(coordinates, states,
                                        [SUNNY_EDDIES, -SUNNY_EDDIES],
                                                DARK_EDDIES, DARK_ABYSS, UNSET, all_evols, **viability_kwargs)
                if not changed:
                    break
                states[(states == SUNNY_EDDIES)] = UNSET
                changed = viability_capture_basin(coordinates, states,
                                        [DARK_EDDIES, -DARK_EDDIES],
                                        SUNNY_EDDIES, SUNNY_ABYSS, UNSET, all_evols, **viability_kwargs)
                if not changed:
                    break
            else:
                warn.warn("reached MAX_ITERATION_EDDIES = %i during the Eddies calculation"%MAX_ITERATION_EDDIES)
        else:
            # assume all eddies are abysses
                states[(states == SUNNY_EDDIES)] = SUNNY_ABYSS
                states[(states == UNSET)] = DARK_ABYSS
                states[(states == DARK_EDDIES)] = DARK_ABYSS


    # All initially given states are set to positive counterparts
    states[(states < UNSET)] *= -1

    return states


