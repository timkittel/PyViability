from __future__ import print_function, division, generators

from . import PTopologyL as topo

from . import periodic_kdtree as periodkdt

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
import datetime as dt

# raise the odeing warning as an error because it indicates that we are at a
# fixed point (or the grid is not fine enough)
warn.filterwarnings("error", category=integ.odepack.ODEintWarning)

# these are automatically set during grid generation but need to be manually set
# when using own grid
BOUNDS_EPSILON = None  # should be set during grid Generation
STEPSIZE = None

# some constants so the calculation does end
MAX_ITERATION_EDDIES = 10
DEBUGGING = 0
VERBOSITY = 0


# the stuff for remembering the status
STATUS = ""
STATUS_PREFIX = None

TOPOLOGY_STEP_LIST = ["GENERAL", "SHELTER", "GLADE", "REMUP", "MANAGEABLE", "REMDOWN", "REST", "GENERAL", ""]  # the empty one is so that can be given, too
STATUS_TOPOLOGY = "TOPOLOGY"
STATUS_INFIX = " "
STATUS_PREPARATION = "PREPARATION"
STATUS_COMPUTATION = "COMPUTATION"
STATUS_POSTPROCESSING = "POSTPROCESSING"
STATUS_DONE = "DONE"
STATUS_EDDIES_DARK = "EDDIES_DARK"
STATUS_EDDIES_SUNNY = "EDDIES_SUNNY"

# verbose printing stuff
PRINT_PREFIX = ""
PRINT_INFIX = " :"
PRINT_INFIX_BEHIND = "( "
PRINT_POSTFIX_BEHIND = " )"

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


# ---- stuff for remembering the paths ----
PATHS = {}
PATHS_LAKE = {}
PATHS_INDEX_TYPE = np.int64
PATHS_INDEX_DEFAULT = np.iinfo(PATHS_INDEX_TYPE).min
PATHS_MANAGEMENT_TYPE = np.int16
PATHS_MANAGEMENT_DEFAULT = np.iinfo(PATHS_MANAGEMENT_TYPE).min

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


def printv(*args, verbosity=1, date_behind = False, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if verbosity <= VERBOSITY:
        date = dt.datetime.now().ctime()
        
        print_args = list(args)
        if date_behind:
            print_args.append(PRINT_INFIX_BEHIND + date + PRINT_POSTFIX_BEHIND)
        else:
            print_args.insert(0, PRINT_PREFIX + date + PRINT_INFIX)
        print(*print_args, **kwargs)


def printd(*args, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if DEBUGGING:
        print(*args, **kwargs)


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
    global MAX_NEIGHBOR_DISTANCE, BOUNDS_EPSILON, STEPSIZE, ALL_NEIGHBORS_DISTANCE

    assert grid_type in ["simplex-based", "orthogonal"], "unkown grid type '{!s}'".format(grid_type)

    boundaries = np.asarray(boundaries)
    periodicity = np.asarray(periodicity)

    dim = boundaries.shape[0]
    offset = boundaries[:, 0].astype(float)
    scaling_factor = boundaries[:, 1] - boundaries[:, 0]

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
        # STEPSIZE = 1.5 * x_step
        # STEPSIZE = 2.5 * x_step
        STEPSIZE = 2. * x_step

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

        # when recursively going through, then add the direct neighbors only
        MAX_NEIGHBOR_DISTANCE = 1.01 * x_step
        # x_step = Delta_0 # Delta_0 is side length of the simplices
        BOUNDS_EPSILON = 0.1 * x_step
        STEPSIZE = 2.5 * x_step  # seems to be correct
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

            global DEBUGGING, PATHS
            # DEBUGGING = True
            # DEBUGGING = DEBUGGING and (start_state == 1)
            # DEBUGGING = (coordinate_index == (10 * 80 - 64,))
            # DEBUGGING = DEBUGGING and la.norm(start - np.array([ 0.51253769,  0.97392462,  1.        ])) < 0.02
            # DEBUGGING = DEBUGGING and la.norm(start - np.array([0.008, 0.747])) < 0.001
            # DEBUGGING = DEBUGGING and start[0] < 0.01
            # DEBUGGING = DEBUGGING and start_state == 1
            # DEBUGGING = DEBUGGING or la.norm(start - np.array([0.1, 0.606])) < 0.02
            # DEBUGGING = True
            # print("DEBUGGING", DEBUGGING)

            if DEBUGGING:
                print()

            for evol_num, evol in enumerate(evolutions):
                traj = evol(start, STEPSIZE)

                final_index, final_state = state_evaluation(traj)

                if final_state in stop_states:  # and constraint(point) and final_distance < MAX_FINAL_DISTANCE:

                    if PATHS:
                        PATHS["reached point"][coordinate_index][:] = traj[-1]
                        PATHS["next point index"][coordinate_index] = final_index
                        PATHS["choice"][coordinate_index] = evol_num

                    if DEBUGGING:
                        print( "%i:" % evol_num, coordinate_index, start, start_state, "-->", final_state )
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
        if np.any( BOUNDS[:, 0] > projected_values) or np.any( BOUNDS[:, 1] < projected_values ):  # is the point out-of-bounds?
            if DEBUGGING:
                print("out-of-bounds")
            return -1, OUT_OF_BOUNDS_STATE

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
        plt.plot(start_point[0], start_point[1], color="black",
                 linestyle="", marker=".", markersize=40, zorder=0)
        plt.plot(_start_point[0], _start_point[1], color="black",
                 linestyle="", marker=".", markersize=40, zorder=0)
        plt.plot(neighbors[:, 0], neighbors[:, 1], color="blue",
                 linestyle="", marker=".", markersize=50, zorder=0)

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

    return closest_index, final_state


# @nb.jit
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
            if DEBUGGING:
                print("out-of-bounds")
            return -1, OUT_OF_BOUNDS_STATE

    _, tree_index = KDTREE.query(point, 1)

    if DEBUGGING:
        print("evaluation:", traj[0], "via", traj[1], "to", KDTREE.data[tree_index], "with state", STATES[tree_index])

    return tree_index, STATES[tree_index]


def state_evaluation_kdtree(traj):
    point = traj[-1]
    if OUT_OF_BOUNDS:
        projected_values = np.tensordot(BASIS_VECTORS_INV, point, axes=[(1,), (0,)])
        if np.any( BOUNDS[:, 0] > projected_values) or np.any( BOUNDS[:, 1] < projected_values ):  # is the point out-of-bounds?
            if DEBUGGING:
                print("out-of-bounds")
            return OUT_OF_BOUNDS_STATE
    _, tree_index = KDTREE.query(point, 1)
    return tree_index, STATES[tree_index]


def pre_calculation_hook_kdtree(coordinates, states,
                                is_sunny=None,
                                periodicity=None,
                                grid_type=None,
                                out_of_bounds=True):
    global KDTREE, STATES, BASIS_VECTORS, BASIS_VECTORS_INV, BOUNDS, OUT_OF_BOUNDS
    STATES = states

    dim = np.shape(coordinates)[-1]
    periodicity_bool = (periodicity > 0)

    printv("creating k-d tree ...", end=" ", flush=True)

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
    printv("done", date_behind=True)

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
                BOUNDS[d, :] = -np.inf, np.inf
                # this basically means, because of periodicity, the trajectories
                # cannot run out-of-bounds
            else:
                # project the values on the basis vector with a scalar product
                # for that reason, basis vectors need to be normalized
                # projected_values = np.tensordot(coordinates, basis_vectors[:,d], axes=[(1,), (0,)])

                # actually the idea above is correct and this is simply the result
                # combined with the checking whether out-of-bounds should be
                # applied
                BOUNDS[d, :] = np.where(out_of_bounds[d], (-BOUNDS_EPSILON, 1 + BOUNDS_EPSILON), (-np.infty, np.infty))

                # BOUNDS[d,:] = np.min(projected_values) - BOUNDS_EPSILON, np.max(projected_values) + BOUNDS_EPSILON
                # BOUNDS[d,:] = np.min(coordinates[:,d]) - BOUNDS_EPSILON, np.max(coordinates[:,d]) + BOUNDS_EPSILON

        projected_values = np.tensordot(coordinates, BASIS_VECTORS_INV, axes=[(1,), (1,)])
        assert np.all( BOUNDS[None, :, 0] < projected_values) \
            and np.all( BOUNDS[None, :, 1] > projected_values ),\
            "BOUNDS and coordinates do not fit together, did you set the correct grid_type argument?"


def _viability_kernel_step(coordinates, states, *, 
                          good_states, bad_states, succesful_states, work_states,
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

        for index in neighbors:  # iterate over the base_index and, if any changes happened, over the neighbors, too
            old_state = states[index]

            if old_state in work_states:
                state_index = work_states.index(old_state)
                succesful_state = succesful_states[state_index]
                bad_state = bad_states[state_index]
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


def get_neighbor_indices(index, shape, neighbor_list=[]):
    """append all neighboring indices of 'index' to 'neighbor_list' if they are within 'shape'"""

    index = np.asarray(index)
    shape = np.asarray(shape)

    for diff_index in it.product([-1, 0, 1], repeat=len(index)):
        diff_index = np.asarray(diff_index)
        new_index = index + diff_index

        if np.count_nonzero(diff_index) and np.all( new_index >= 0 ) and np.all( new_index < shape ):
            neighbor_list.append(tuple(new_index))

    return neighbor_list


def _viability_kernel(coordinates, states, *, 
                     good_states, bad_state, succesful_state, work_state, evolutions,
                     state_evaluation,
                     periodic_boundaries=[]
                     ):
    """calculate the viability kernel by iterating through the viability kernel steps
    until convergence (no further change)"""
    # assert coordinates.shape[:-1] == states.shape[:-1], "'coordinates' and 'states' don't match in shape"

    # assert "x_step" in globals()  # needs to be set by the user for now ... will be changed later
    assert "BOUNDS_EPSILON" in globals()  # needs to be set by the user for now ... will be changed later
    # assert "MAX_FINAL_DISTANCE" in globals()  # needs to be set by the user for now ... will be changed later
    assert "MAX_NEIGHBOR_DISTANCE" in globals()  # needs to be set by the user for now ... will be changed later
    assert "STEPSIZE" in globals()  # as the above comments
    # global x_half_step
    # x_half_step = x_step/2

    try:
       work_states = list(work_state)
    except TypeError:
        work_states = [work_state]

    try:
        succesful_states = list(succesful_state)
    except TypeError:
        succesful_states = [succesful_state]

    try:
        bad_states = list(bad_state)
    except TypeError:
        bad_states = [bad_state]

    # actually only one step is needed due to the recursive checks (i.e. first
    # checking all neighbors of a point that changed state)
    return _viability_kernel_step(coordinates, states, 
                                  good_states=good_states, 
                                  bad_states=bad_states, 
                                  succesful_states=succesful_states, 
                                  work_states=work_states, 
                                  evolutions=evolutions, 
                                  state_evaluation=state_evaluation)


def _viability_capture_basin(coordinates, states, *,
                            target_states, reached_state, bad_state, work_state, evolutions,
                            state_evaluation=state_evaluation_kdtree
                            ):
    """reuse the viability kernel algorithm to calculate the capture basin"""

    if work_state in states and any( ( target_state in states for target_state in target_states) ):
        # num_work = np.count_nonzero(work_state == states)
        _viability_kernel(coordinates, states, 
                         good_states=target_states + [reached_state], 
                         bad_state=work_state, 
                         succesful_state=reached_state,
                         work_state=work_state, 
                         evolutions=evolutions, 
                         state_evaluation=state_evaluation
                         )
        # changed = (num_work == np.count_nonzero(reached_state == states))
    else:
        printv("empty work or target set", verbosity=2)
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
            print(("{:<13} ({:>2}): {:>6.2f}% ( {:>"+total_length+"} )").format(region, globals()[region],  num / total * 100, num))
    print()
    if print_unknown and num_sum != total:
        print(("{:<15}: {:>6.2f}% ( {:>"+total_length+"} )").format("UNKNOWN",  (total - num_sum) / total * 100, total - num_sum))
        print()


def plot_points(coords, states, markersize=15, plot_unset=False):
    """plot the current states in the viability calculation as points"""

    assert set(states.flatten()).issubset(COLORS)

    plot_colors = list(COLORS)
    if not plot_unset:
        plot_colors.pop(UNSET)

    for color_state in plot_colors:
        plt.plot(coords[ states == color_state, 0], coords[ states == color_state, 1], color=COLORS[color_state],
                 linestyle="", marker=".", markersize=markersize, zorder=0)


def plot_areas(coords, states):
    """plot the current states in the viability calculation as areas"""

    states = states.flatten()
    assert set(states).issubset(COLORS)
    coords = np.reshape(coords, states.shape + (-1,))
    x, y = coords[:, 0], coords[:, 1]

    color_states = sorted(COLORS)
    cmap = mpl.colors.ListedColormap([ COLORS[state] for state in color_states ])
    bounds = color_states[:1] + [ state + 0.5 for state in color_states[:-1]] + color_states[-1:]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.tripcolor(x, y, states, cmap=cmap, norm=norm, shading="gouraud")


def make_run_function(rhs,
                      ordered_params,
                      offset,
                      scaling_vector,
                      returning="integration",
                      remember=True,
                      use_numba=True,
                      nb_nopython=True,
                      rescaling_epsilon=1e-6,
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

            return dy / (dy_norm + rescaling_epsilon)

    else:
        def rhs_rescaled(y, t, *args):
            # because of the rescaling to 1 in every dimension
            # transforming y -> x
            x = offset + np.dot(Sinv, y)
            # transforming dx -> dy
            dy = np.dot(S, rhs(x, t, *args))  # calculate the rhs
            # normalization of dy
            dy_norm = np.sqrt(np.sum(dy ** 2, axis=-1))
            if dy_norm == 0.:
                return np.zeros_like(dy)
            return dy / (dy_norm + rescaling_epsilon)

    if use_numba:
        @nb.jit(nopython=nb_nopython)
        def normalized_linear_approximation(x, dt):
            xdot = rhs_rescaled(x, dt, *ordered_params)
            traj = np.empty((2, x.shape[0]))
            traj[0] = x
            if np.any(np.isinf(xdot)):  # raise artifiially the warning if inf turns up
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
            if np.any(np.isinf(xdot)):  # raise artifiially the warning if inf turns up
                warn.warn("got a inf in the RHS function; assume {!s} to be a stable fixed point and returning the starting point".format(x),
                          category=RuntimeWarning)
                traj[1] = traj[0]
                if DEBUGGING:
                    p = traj[1]
                    # plot the point, but a bit larger than the color one later
                    plt.plot(p[0], p[1], color="red",
                             linestyle="", marker=".", markersize=45, zorder=0)

            elif DEBUGGING:
                plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=1)
            return traj

    # @nb.jit
    # def distance_normalized_rhs(x, lam, x0, *args):
        # val = rhs_scaled_to_one(x, lam, *args)  # calculate the rhs
        # if lam == 0:
            # return val / np.sqrt(np.sum( val ** 2, axis=-1) )
        # return val * lam / np.sum( (x-x0) * val, axis=-1)

    # @helper.remembering(remember=remember)
    # def integration(p, stepsize):
        # if DEBUGGING:
            # integ_time = np.linspace(0, stepsize, 100)
        # else:
            # integ_time = [0, stepsize]
        # try:
            # with helper.stdout_redirected():
                # traj = integ.odeint(distance_normalized_rhs, p, integ_time,
                                    # args=(p,) + ordered_params,
                                    # printmessg = False
                                    # )
            # if np.any(np.isnan(traj[-1])): # raise artifiially the warning if nan turns up
                # raise integ.odepack.ODEintWarning("got a nan")
        # except integ.odepack.ODEintWarning:
            # warn.warn("got an integration warning; assume {!s} to be a stable fixed point and returning the starting point".format(p),
                      # category=RuntimeWarning)
            # if DEBUGGING:
                # # plot the point, but a bit larger than the color one later
                # plt.plot(p[0], p[1], color = "red",
                    # linestyle = "", marker = ".", markersize = 45 ,zorder=0)
            # return np.asarray([p, p])
#
        # if DEBUGGING:
            # plt.plot(traj[:, 0], traj[:, 1], color="red", linewidth=3)
            # return np.asarray([traj[0], traj[-1]])
        # else:
            # return traj

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
    return np.sum( la.norm( traj[1:] - traj[:-1], axis=-1) )


def trajectory_length_index(traj, target_length):
    lengths = np.cumsum( la.norm( traj[1:] - traj[:-1], axis=-1) )

    if target_length < lengths[-1]:
        return traj.shape[0]  # incl. last element
    index_0, index_1 = 0, traj.shape[0] - 1

    while index_0 not in [index_1, index_1 - 1]:
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


def reset_initial_states(coordinates, states):
    # All initially given states are set to positive counterparts
    states[(states < UNSET)] *= -1


def set_global_status(*args, print_verbosity=None):
    assert not STATUS_PREFIX is None, "STATUS_PREFIX has to be set, maybe you found a bug?"
    global STATUS
    STATUS = STATUS_INFIX.join((STATUS_PREFIX,) + args)
    if print_verbosity is None:
        print_verbosity = 2
        if STATUS_PREFIX == STATUS_TOPOLOGY:
            if args[1] == STATUS_PREPARATION:
                print_verbosity = 1
    printv(STATUS, verbosity=print_verbosity)


def topology_classification(coordinates, states, default_evols, management_evols, is_sunny,
                            periodic_boundaries=[],
                            upgradeable_initial_states=False,
                            compute_eddies=False,
                            pre_calculation_hook=pre_calculation_hook_kdtree,  # None means nothing to be done
                            state_evaluation=state_evaluation_kdtree_numba,
                            post_computation_hook=reset_initial_states,
                            grid_type="orthogonal",
                            out_of_bounds=True,  # either bool or bool array with shape (dim, ) or shape (dim, 2) with values for each boundary
                            remember_paths=False,
                            verbosity=0,
                            stop_when_finished=TOPOLOGY_STEP_LIST[-1],  # means everything goes
                            ):
    """calculates different regions of the state space using viability theory algorithms
    """

    global VERBOSITY
    VERBOSITY = verbosity

    global STATUS_PREFIX
    STATUS_PREFIX = STATUS_TOPOLOGY
    current_step = 0
    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
    if isinstance(stop_when_finished, str):
        stop_when_finished = TOPOLOGY_STEP_LIST.index(stop_when_finished)
    assert isinstance(stop_when_finished, int) and stop_when_finished >= 0

    # upgrading initial states to higher is not yet implemented
    if upgradeable_initial_states:
        raise NotImplementedError("upgrading of initally given states not yet implemented")

    coordinates = np.asarray(coordinates)
    states = np.asarray(states)

    grid_size, dim = coordinates.shape
    assert states.shape == (grid_size,), "coordinates and states input doesn't match"


    if remember_paths:
        printv("generating PATHS and PATHS_LAKE arrays")
        global PATHS
        PATHS = {}
        PATHS["reached point"] = np.copy(coordinates)  # for the target point
        PATHS["next point index"] = np.ones((grid_size,), dtype=PATHS_INDEX_TYPE) * PATHS_INDEX_DEFAULT  # the coordinate where the target point get's associated to
        PATHS["choice"] = np.ones((grid_size,), dtype=PATHS_MANAGEMENT_TYPE) * PATHS_MANAGEMENT_DEFAULT  # for the number of the management option

        PATHS_LAKE = {}
        PATHS_LAKE["reached point"] = np.copy(PATHS["reached point"])
        PATHS_LAKE["next point index"] = np.copy(PATHS["next point index"])
        PATHS_LAKE["choice"] = np.copy(PATHS["choice"])

        # PATHS = (np.copy(coordinates),  # for the target point
                 # -np.ones((grid_size,), dtype=int),  # the coordinate where the target point get's associated to
                 # -np.ones((grid_size,), dtype=np.int16) )  # for the number of the management option

    if periodic_boundaries == []:
        periodic_boundaries = - np.ones(dim)
    periodic_boundaries = np.asarray(periodic_boundaries)

    if pre_calculation_hook is not None:
        # run the pre-calculation hook (defaults to creation of the KD-Tree)
        pre_calculation_hook(coordinates, states,
                             is_sunny=is_sunny,
                             periodicity=periodic_boundaries,
                             grid_type=grid_type,
                             out_of_bounds=out_of_bounds)

    # make sure, evols can be treated as lists
    default_evols = list(default_evols)
    management_evols = list(management_evols)

    all_evols = default_evols + management_evols

    # better remove this and use directly the lower level stuff, see issue #13
    viability_kwargs = dict(
        state_evaluation=state_evaluation,
    )

    shelter_empty = False
    backwater_empty = False

    if all_evols:
        if not default_evols:
            printv('no default dynamics given, skipping upstream')
            current_step += 3
        else:
            # shelter computation
            current_step += 1
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
            states[(states == UNSET) & is_sunny(coordinates)] = SHELTER  # initial state for shelter calculation
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION)
            _viability_kernel(coordinates, states, 
                    good_states=[SHELTER, -SHELTER], 
                    bad_state=UNSET, 
                    succesful_state=SHELTER, 
                    work_state=SHELTER, 
                    evolutions=default_evols,
                    **viability_kwargs)
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)
            if stop_when_finished == "shelter":
                # do the post computation hook (default, setting negative states positive)
                # and then exit
                post_computation_hook(coordinates, states)
                return 

            if not np.any(states == SHELTER):  # shelter is empty?
                printv('shelter empty, skip rest of upstream')
                current_step += 2
            else:
                current_step += 1
                if not management_evols:
                    printv('no management dynamics given, skipping glade')
                else:

                    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
                    states[(states == UNSET) & is_sunny(coordinates)] = SUNNY_UP

                    # glade computation
                    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION)
                    _viability_capture_basin(coordinates, states, 
                            target_states=[SHELTER, -SHELTER], 
                            reached_state=GLADE, 
                            bad_state=UNSET, 
                            work_state=SUNNY_UP, 
                            evolutions=all_evols, **viability_kwargs)
                    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)

                if stop_when_finished <= current_step:
                    # do the post computation hook (default, setting negative states positive)
                    # and then exit
                    post_computation_hook(coordinates, states)
                    return 

                current_step += 1
                # computation remaining upstream dark and sunny (containing possible lake)
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
                states[(states == UNSET)] = DARK_UP
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION)
                _viability_capture_basin(coordinates, states, 
                                        target_states=[SHELTER, -SHELTER, GLADE, -GLADE, -SUNNY_UP, -DARK_UP, -LAKE], 
                                        reached_state=SUNNY_UP, 
                                        bad_state=UNSET, 
                                        work_state=DARK_UP, 
                                        evolutions=all_evols,
                                        **viability_kwargs)

                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_POSTPROCESSING)
                states[~is_sunny(coordinates) & (states == SUNNY_UP)] = DARK_UP
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)

        assert current_step == TOPOLOGY_STEP_LIST.index("REMUP"), "consistency check failed, bug? ({} != {})".format(current_step, TOPOLOGY_STEP_LIST.index("REMUP"))

        if stop_when_finished <= current_step:
            # do the post computation hook (default, setting negative states positive)
            # and then exit
            post_computation_hook(coordinates, states)
            return 


        if not management_evols:
            printv('no management dynamics given, skipping lake and downstream')
            current_step += 2
        else:
            current_step += 1
            assert current_step == TOPOLOGY_STEP_LIST.index("MANAGEABLE"), "consistency check failed, bug? ({} != {})".format(current_step, TOPOLOGY_STEP_LIST.index("MANAGEABLE"))
            if remember_paths:
                _PATHS = PATHS
                PATHS = PATHS_LAKE
            # compute rest of manageable region (lake + backwaters)
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
            states[is_sunny(coordinates) & (states == SUNNY_UP)] = LAKE
            states[is_sunny(coordinates) & (states == UNSET)] = BACKWATERS
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION)
            _viability_kernel(coordinates, states, 
                    good_states=[
                                 SHELTER, -SHELTER,  # these should not be possible to be reached
                                 GLADE, -GLADE,  # these should not be possible to be reached
                                 LAKE, -LAKE,
                                 BACKWATERS, -BACKWATERS
                                 ], 
                    bad_state=[SUNNY_UP, UNSET], 
                    succesful_state=[LAKE, BACKWATERS], 
                    work_state=[LAKE, BACKWATERS], 
                    evolutions=all_evols,
                    **viability_kwargs)
            if remember_paths:
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_POSTPROCESSING)
                PATHS = _PATHS
                # don't need to set back PATHS_LAKE as these are just references to the dictionaries anyway
                del _PATHS  # just for keeping the code clean, not really necessary

                # write the backwaters part to the normal mask
                mask = (states == BACKWATERS)
                PATHS["reached point"][mask] = PATHS_LAKE["reached point"][mask]
                PATHS["next point index"][mask] = PATHS_LAKE["next point index"][mask]
                PATHS["choice"][mask] = PATHS_LAKE["choice"][mask]
                del mask
                if not np.any(states == LAKE):
                    printv("no lake found, removing PATHS_LAKE arrays")
                    PATHS_LAKE = {}
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)


            if stop_when_finished <= current_step:
                # do the post computation hook (default, setting negative states positive)
                # and then exit
                post_computation_hook(coordinates, states)
                return 


            current_step += 1
            if not np.any(states == BACKWATERS):
                printv('backwater empty, skip remaining downstream')
            else:
                # calculate remaining downstream dark and sunny
                assert current_step == TOPOLOGY_STEP_LIST.index("REMDOWN"), "consistency check failed, bug? ({} != {})".format(current_step, TOPOLOGY_STEP_LIST.index("REMDOWN"))
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
                states[(states == UNSET)] = DARK_DOWN
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION)
                _viability_capture_basin(coordinates, states, 
                        target_states=[BACKWATERS, -SUNNY_DOWN, -DARK_DOWN], 
                        reached_state=SUNNY_DOWN, 
                        bad_state=UNSET, 
                        work_state=DARK_DOWN, 
                        evolutions=all_evols, 
                        **viability_kwargs)
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_POSTPROCESSING)
                states[~is_sunny(coordinates) & (states == SUNNY_DOWN)] = DARK_DOWN
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)

        if stop_when_finished <= current_step:
            # do the post computation hook (default, setting negative states positive)
            # and then exit
            post_computation_hook(coordinates, states)
            return 

        current_step += 1
        # calculate trench and set the rest as preliminary estimation for the eddies
        set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_PREPARATION)
        states[is_sunny(coordinates) & (states == UNSET)] = SUNNY_EDDIES

        # look only at the coordinates with state == UNSET
        set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION)
        _viability_capture_basin(coordinates, states,
                                target_states=[
                                               SUNNY_EDDIES, -SUNNY_EDDIES, 
                                               SUNNY_ABYSS, -SUNNY_ABYSS  # a tad imprecise that both negative states are in here
                                               ],
                                reached_state=DARK_EDDIES, 
                                bad_state=TRENCH, 
                                work_state=UNSET, 
                                evolutions=all_evols, 
                                **viability_kwargs)
        if not compute_eddies:
            # assume all eddies are abysses
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_POSTPROCESSING)
            states[(states == SUNNY_EDDIES)] = SUNNY_ABYSS
            states[(states == UNSET)] = DARK_ABYSS
            states[(states == DARK_EDDIES)] = DARK_ABYSS
        else:

            # the preliminary estimations for sunny and dark eddie are set
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_SUNNY, STATUS_PREPARATION)
            states[(states == SUNNY_EDDIES)] = UNSET
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_SUNNY, STATUS_COMPUTATION)
            _viability_capture_basin(coordinates, states,
                                    target_states=[DARK_EDDIES, -DARK_EDDIES],
                                    reached_state=SUNNY_EDDIES, 
                                    bad_state=SUNNY_ABYSS, 
                                    work_state=UNSET, 
                                    evolutions=all_evols, 
                                    **viability_kwargs)
            set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_SUNNY, STATUS_DONE)

            for num in range(MAX_ITERATION_EDDIES):
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_DARK, STATUS_PREPARATION)
                states[(states == DARK_EDDIES)] = UNSET
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_DARK, STATUS_COMPUTATION)
                changed = _viability_capture_basin(coordinates, states,
                                                  target_states=[SUNNY_EDDIES, -SUNNY_EDDIES],
                                                  reached_state=DARK_EDDIES, 
                                                  bad_state=DARK_ABYSS, 
                                                  work_state=UNSET, 
                                                  evolutions=all_evols, 
                                                  **viability_kwargs)
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_DARK, STATUS_DONE)
                if not changed:
                    break
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_SUNNY, STATUS_PREPARATION)
                states[(states == SUNNY_EDDIES)] = UNSET
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_SUNNY, STATUS_COMPUTATION)
                changed = _viability_capture_basin(coordinates, states,
                                                  target_states=[DARK_EDDIES, -DARK_EDDIES],
                                                  reached_state=SUNNY_EDDIES, 
                                                  bad_state=SUNNY_ABYSS, 
                                                  work_state=UNSET, 
                                                  evolutions=all_evols, 
                                                  **viability_kwargs)
                set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_COMPUTATION, STATUS_EDDIES_SUNNY, STATUS_DONE)
                if not changed:
                    break
            else:
                warn.warn("reached MAX_ITERATION_EDDIES = %i during the Eddies calculation"%MAX_ITERATION_EDDIES)
    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)

    current_step += 1
    assert current_step == len(TOPOLOGY_STEP_LIST) - 2, "consistency check failed, bug? ({} != {})".format(current_step, len(TOPOLOGY_STEP_LIST) - 2)
    # there is a -2 in the assert statement because a trailing '""' is added to 'TOPOLOGY_STEP_LIST' in order to have '""' as the default for 'stop_when_finished'

    # computation is done,
    # do the post computation hook (default, setting negative states positive)
    # and then exit
    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_POSTPROCESSING)
    post_computation_hook(coordinates, states)
    set_global_status(TOPOLOGY_STEP_LIST[current_step], STATUS_DONE)

    # clean up
    global STATUS, STATUS_PREFIX
    STATUS = ""
    STATUS_PREFIX = None

    return 


