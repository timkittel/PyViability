"""
helper.py provides small utilities that don't really fit enywhere else but are needed in the package.
"""



# import contextlib as ctl
import inspect
# import numpy as np
# import os
# import sys
#
# from scipy.spatial import cKDTree


# REMEMBERED = {}  # used by remembering decorator

# def _plot_nothing(ax, traj, management_option):
#     pass
#
#
# def _plot2d(ax, traj, management_option):
#     ax.plot(traj[0], traj[1],
#                 color="lightblue" if management_option == 0 else "black")
#
#
# def _plot3d(ax, traj, management_option):
#     ax.plot3D(xs=traj[0], ys=traj[1], zs=traj[2],
#                 color="lightblue" if management_option == 0 else "black")
#
# def _dont_follow(*args, **kwargs):
#     return False


# def follow_point(starting_indices, paths, grid, states,
#                  verbosity=1,
#                  follow_condition=_dont_follow,
#                  plot_func=_plot_nothing,
#                  plot_axes=None,
#                  run_function=None,
#                  stepsize=None,
#                  ):
#
#     if not follow_condition is _dont_follow:
#         tree = cKDTree(grid)
#
#     starting_indices = list(starting_indices)
#     if verbosity >= 1:
#         print("starting points and states for paths:")
#         for ind in starting_indices:
#             print("{!s} --- {:>2}".format(grid[ind], states[ind]))
#         print()
#     plotted_indices = set()
#     print("calculating and plotting paths ... ", end="", flush=True)
#     for ind0 in starting_indices:
#         if ind0 in plotted_indices:
#             continue
#         plotted_indices.add(ind0)
#         x0 = grid[ind0]
#         ind1 = paths[1][ind]
#         if ind1 > 0:
#             x1 = paths[0][ind0]
#         else:
#             if follow_condition(ind0, grid, states):
#                 assert not run_function is None, "please provide a run_function"
#                 assert not stepsize is None, "please provide a stepsize"
#                 _, x1 = run_function(x0, stepsize)
#                 _, ind1 = tree.query(x1)
#             else:
#                 continue
#         plot_func(plot_axes, traj, management_option)
#
#
#         if verbosity >= 2:
#             print("{!s} --- {:>2}".format(grid[ind], states[ind]))
#             _, x1 = default_run(grid[ind], stepsize)
#             print("new_point: ", x1)
#             print("going to index:", paths[1][ind])
#             if paths[1][ind] >=0:
#                 starting_indices.append(paths[1][ind])
#             elif FOLLOW:
#                 _, index = tree.query(x1)
#                 print("going to index (tree):", index)
#                 starting_indices.append(index)
#         if np.all(is_inside([x0, x1], args.plot_boundaries)):
#             traj = list(zip(x0, x1))
#             ax3d.plot3D(xs=traj[0], ys=traj[1], zs=traj[2],
#                         color="lightblue" if paths[2][ind] == 0 else "black")
#             starting_indices.append(paths[1][ind])
#     print("done\n")


def get_parameter_order(func):
    args, _, _, defaults = inspect.getargspec(func)
    assert len(args) >= 2, "your rhs function takes only %i arguments, but it "\
        "should take at least x0 and t for odeint to work with it" % len(args)
    return args[2:]


def get_ordered_parameters(func, parameter_dict):
    ordered_parameters = get_parameter_order(func)
    assert set(ordered_parameters).issubset(parameter_dict), "you did not " \
        "provide all parameters"
    return tuple([parameter_dict[par] for par in ordered_parameters])


# def fileno(file_or_fd):
#     fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
#     if not isinstance(fd, int):
#         raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
#     return fd


# @ctl.contextmanager
# def stdout_redirected(to=os.devnull, stdout=None):
#     """
#     http://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
#     """
#     if stdout is None:
#         stdout = sys.stdout
#
#     stdout_fd = fileno(stdout)
#     # copy stdout_fd before it is overwritten
#     # NOTE: `copied` is inheritable on Windows when duplicating a standard
#     # stream
#     with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
#         stdout.flush()  # flush library buffers that dup2 knows nothing about
#         try:
#             os.dup2(fileno(to), stdout_fd)  # $ exec >&to
#         except ValueError:  # filename
#             with open(to, 'wb') as to_file:
#                 os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
#         try:
#             yield stdout  # allow code to be run with the redirected stdout
#         finally:
#             # restore stdout to its previous value
#             # NOTE: dup2 makes stdout_fd inheritable unconditionally
#             stdout.flush()
#             os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


# class remembering(object):
#
#     def __init__(self, remember=True):
#         self.remember = remember
#
#     def __call__(self, f):
#         if not self.remember:
#             return f
#
#         global REMEMBERED
#         REMEMBERED[f] = {}
#
#         def remembering_f(p, stepsize):
#             global REMEMBERED
#             p_tuple = tuple(p)
#             if p_tuple in REMEMBERED[f]:
#                 return REMEMBERED[f][p_tuple]
#             p2 = f(p, stepsize)
#             REMEMBERED[f][p_tuple] = p2
#             return p2
#
#         return remembering_f
