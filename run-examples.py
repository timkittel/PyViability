#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from __future__ import division, print_function

# viability imports
import pyviability as viab
from pyviability import helper
from pyviability import libviability as lv
from pyviability import tsm_style as topo

# model imports
import examples.AWModel as awm
import examples.ConsumptionModel as cm
import examples.FiniteTimeLakeModel as ftlm
import examples.FiniteTimeLakeModel2 as ftlm2
import examples.GravityPendulumModel as gpm
import examples.PlantModel as pm
import examples.PopulationAndResourceModel as prm
import examples.SwingEquationModel as sqm
import examples.TechChangeModel as tcm

# other useful stuff
import argparse
try:
    import argcomplete
except ImportError:
    with_argcomplete = False
else:
    with_argcomplete = True
import datetime as dt
import functools as ft
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.optimize as opt
import time
import sys


PRINT_VERBOSITY = 2


def save_figure(filename, fig=None):
    if fig is None:
        fig = plt.gcf()
    print("saving to {!r} ... ".format(filename), end="", flush=True)
    fig.savefig(filename)
    print("done")


def plotPhaseSpace( evol, boundaries, steps = 2000, xlabel = "", ylabel = "", colorbar = True, style = {}, alpha = None , maskByCond = None, invertAxes = False, ax = plt, lwspeed = False):
	# separate the boundaries
	Xmin, Ymin, Xmax, Ymax = boundaries

	# check boundaries sanity
	assert Xmin < Xmax
	assert Ymin < Ymax

	# build the grid
	X = np.linspace(Xmin, Xmax, steps)
	Y = np.linspace(Ymin, Ymax, steps)

	XY = np.array(np.meshgrid(X, Y))

	# if Condition give, set everything to zero that fulfills it
	if maskByCond:
		mask = maskByCond(XY[0], XY[1])
		XY[0] = np.ma.array(XY[0], mask = mask)
		XY[1] = np.ma.array(XY[1], mask = mask)

## 		dummy0 = np.zeros((steps,steps))
## 		XY[0] = np.where(mask, XY[0], dummy0)
## 		XY[1] = np.where(mask, XY[1], dummy0)

	# calculate the changes ... input is numpy array
	dX, dY = evol(XY,0) # that is where deriv from Vera is mapped to

	if invertAxes:
		data = [Y, X, np.transpose(dY), np.transpose(dX)]
	else:
		data = [X, Y, dX, dY]


	# separate linestyle
	linestyle = None
	if type(style) == dict and "linestyle" in style.keys():
		linestyle = style["linestyle"]
		style.pop("linestyle")

	# do the actual plot
	if style == "dx":
		c = ax.streamplot(*data, color=dX, linewidth=5*dX/dX.max(), cmap=plt.cm.autumn)
	elif style:
            speed = np.sqrt(data[2]**2 + data[3]**2)
            if "linewidth" in style and style["linewidth"] and lwspeed:
                style["linewidth"] = style["linewidth"] * speed/np.nanmax(speed)
##             print speed
##             print np.nanmax(speed)
            c = ax.streamplot(*data, **style)
	else:
		# default style formatting
		speed = np.sqrt(dX**2 + dY**2)
		c = ax.streamplot(*data, color=speed, linewidth=5*speed/speed.max(), cmap=plt.cm.autumn)


	# set opacity of the lines
	if alpha:
		c.lines.set_alpha(alpha)

	# set linestyle
	if linestyle:
		c.lines.set_linestyle(linestyle)

	# add labels if given
	if invertAxes:
		temp = xlabel
		xlabel = ylabel
		ylabel = temp
	if xlabel:
		if ax == plt:
			ax.xlabel(xlabel)
		else:
			ax.set_xlabel(xlabel)
	if ylabel:
		if ax == plt:
			ax.ylabel(ylabel)
		else:
			ax.set_ylabel(ylabel)

	# add colorbar
	if colorbar:
		assert not "color" in style.keys(), "you want a colorbar for only one color?"
		ax.colorbar()

def generate_example(default_rhss,
                     management_rhss,
                     sunny_fct,
                     boundaries,
                     default_parameters=[],
                     management_parameters=[],
                     periodicity=[],
                     default_rhssPS=None,
                     management_rhssPS=None,
                     out_of_bounds=True,
                     compute_eddies=False,
                     rescaling_epsilon=1e-6,
                     stepsize=None,
                     xlabel=None,
                     ylabel=None,
                     set_ticks=None,
                     ):
    """Generate the example function for each example.

    :param default_rhss: list of callables
        length 1, right-hand-side function of the default option. For future compatibiility, this was chosen to be a list already.
    :param management_rhss: list of callables
        right-hand-side functions of the management options
    :param sunny_fct: callable
        function that determines whether a point / an array of points is in the sunny region
    :param boundaries: array-like, shape : (dim, 2)
        for each dimension of the model, give the lower and upper boundary
    :param default_parameters: list of dict, optional
        length 1, the dict contains the parameter values for the default option. For future compatibiility, this was chosen to be a list already.
    :param management_parameters: list of dict, optional
        each dict contains the parameter values for the each management option respectively
    :param periodicity: list, optional
        provide the periodicity of the model's phase space
    :param default_rhssPS: list of callables, optional
        if the default_rhss are not callable for arrays (which is necessary for the plotting of the phase space), then provide a corresponding (list of) function(s) here
    :param management_rhssPS:list of callables, optional
        if the management_rhss are not callable for arrays (which is necessary for the plotting of the phase space), then provide a corresponding (list of) function(s) here
    :param out_of_bounds: bool, default : True
        If going out of the bundaries is interpreted as being in the undesirable region.
    :param compute_eddies:
        Should the eddies be computed? (Becaus the computation of Eddies might take long, this is skipped for models where it's know that there are no Eddies.)
    :param stepsize
        step size used during the viability kernel computation
    :param rescaling_epsilon:
        The epsilon for the time homogenization, see https://arxiv.org/abs/1706.04542 for details.
    :param xlabel:
    :param ylabel:
    :param set_ticks:
    :return: callable
        function that when being called computes the specific example
    """

    plotPS = lambda rhs, boundaries, style: plotPhaseSpace(rhs, [boundaries[0][0], boundaries[1][0], boundaries[0][1], boundaries[1][1]], colorbar=False, style=style)

    if not default_parameters:
        default_parameters = [{}] * len(default_rhss)
    if not management_parameters:
        management_parameters = [{}] * len(management_rhss)

    xlim, ylim = boundaries
    if default_rhssPS is None:
        default_rhssPS = default_rhss
    if management_rhssPS is None:
        management_rhssPS = management_rhss

    def example_function(example_name,
                         grid_type="orthogonal",
                         backscaling=True,
                         plotting="points",
                         run_type="integration",
                         save_to="",
                         n0=80,
                         hidpi=False,
                         use_numba=True,
                         stop_when_finished="all",
                         flow_only=False,
                         mark_fp=None,
            ):

        plot_points = (plotting == "points")
        plot_areas = (plotting == "areas")

        grid, scaling_factor,  offset, x_step = viab.generate_grid(boundaries,
                                                        n0,
                                                        grid_type,
                                                        periodicity = periodicity) #noqa
        states = np.zeros(grid.shape[:-1], dtype=np.int16)

        NB_NOPYTHON = False
        default_runs = [viab.make_run_function(
            nb.jit(rhs, nopython=NB_NOPYTHON),
            helper.get_ordered_parameters(rhs, parameters),
            offset,
            scaling_factor,
            returning=run_type,
            rescaling_epsilon=rescaling_epsilon,
            use_numba=use_numba,
            ) for rhs, parameters in zip(default_rhss, default_parameters)] #noqa
        management_runs = [viab.make_run_function(
            nb.jit(rhs, nopython=NB_NOPYTHON),
            helper.get_ordered_parameters(rhs, parameters),
            offset,
            scaling_factor,
            returning=run_type,
            rescaling_epsilon=rescaling_epsilon,
            use_numba=use_numba,
            ) for rhs, parameters in zip(management_rhss, management_parameters)] #noqa

        sunny = viab.scaled_to_one_sunny(sunny_fct, offset, scaling_factor)

        # adding the figure here already in case VERBOSE is set
        # this makes only sense, if backscaling is switched off
        if backscaling:
            figure_size = np.array([7.5, 7.5])
        else:
            figure_size = np.array([7.5, 2.5 * np.sqrt(3) if grid_type == "simplex-based" else 7.5 ])
        if hidpi:
            figure_size = 2 * figure_size

        figure_size = tuple(figure_size.tolist())

        if (not backscaling) and plot_points:
            # figure_size = (15, 5 * np.sqrt(3) if grid_type == "simplex-based" else 15)
            # figure_size = (15, 5 * np.sqrt(3) if grid_type == "simplex-based" else 15)
            fig = plt.figure(example_name, figsize=figure_size, tight_layout=True)

        # print(lv.STEPSIZE)
        # lv.STEPSIZE = 2 * x_step
        if stepsize is None:
            lv.STEPSIZE = 2 * x_step * max([1, np.sqrt( n0 / 30 )])  # prop to 1/sqrt(n0)
        else:
            lv.STEPSIZE = stepsize
        print(lv.STEPSIZE)
        # print(lv.STEPSIZE)
        # assert False
        print("STEPSIZE / x_step = {:5.3f}".format(lv.STEPSIZE / x_step))

        start_time = time.time()
        viab.topology_classification(grid, states, default_runs, management_runs, sunny,
                                     periodic_boundaries = periodicity,
                                     grid_type=grid_type,
                                     compute_eddies=compute_eddies,
                                     out_of_bounds=out_of_bounds,
                                     stop_when_finished=stop_when_finished,
                                     verbosity=PRINT_VERBOSITY,
                                     )
        time_diff = time.time() - start_time

        print("run time: {!s}".format(dt.timedelta(seconds=time_diff)))

        if backscaling:
            grid = viab.backscaling_grid(grid, scaling_factor, offset)

            if plot_points:
                fig = plt.figure(example_name, figsize=figure_size, tight_layout=True)
                # fig = plt.figure(figsize=(15, 15), tight_layout=True)

                if not flow_only:
                    viab.plot_points(grid, states, markersize=30 if hidpi else 15)
                if ARGS.title:
                    plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(ft.partial(rhs, **parameters), boundaries, topo.styleDefault) #noqa
                    for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
                [plotPS(ft.partial(rhs, **parameters), boundaries, style)
                    for rhs, parameters, style in zip(management_rhssPS, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                if set_ticks is not None:
                    set_ticks()
                else:
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                if xlabel is not None:
                    plt.xlabel(xlabel)
                if ylabel is not None:
                    plt.ylabel(ylabel)

                if save_to:
                    save_figure(save_to)


            if plot_areas:
                fig = plt.figure(example_name, figsize=figure_size, tight_layout=True)

                if not flow_only:
                    viab.plot_areas(grid, states)
                if ARGS.title:
                    plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(ft.partial(rhs, **parameters), boundaries, topo.styleDefault) #noqa
                    for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
                [plotPS(ft.partial(rhs, **parameters), boundaries, style)
                    for rhs, parameters, style in zip(management_rhssPS, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                if set_ticks is not None:
                    set_ticks()
                else:
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                if xlabel is not None:
                    plt.xlabel(xlabel)
                if ylabel is not None:
                    plt.ylabel(ylabel)

                if save_to:
                    save_figure(save_to)

        else:
            plot_x_limits = [0, 1.5 if grid_type == "simplex-based" else 1]
            plot_y_limits = [0, np.sqrt(3)/2 if grid_type == "simplex-based" else 1]

            default_PSs = [viab.make_run_function(rhs, helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning="PS") #noqa
                            for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
            management_PSs = [viab.make_run_function(rhs, helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning="PS") #noqa
                                for rhs, parameters in zip(management_rhssPS, management_parameters)] #noqa

            if plot_points:
                # figure already created above

                if not flow_only:
                    viab.plot_points(grid, states, markersize=30 if hidpi else 15)
                if ARGS.title:
                    plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(rhs, [plot_x_limits, plot_y_limits], topo.styleDefault) for rhs, parameters in zip(default_PSs, default_parameters)]
                [plotPS(rhs, [plot_x_limits, plot_y_limits], style) for rhs, parameters, style in zip(management_PSs, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.axis("equal")

                plt.xlim(plot_x_limits)
                plt.ylim(plot_y_limits)

                if save_to:
                    save_figure(save_to)


            if plot_areas:
                fig = plt.figure(example_name, figsize=(15, 15), tight_layout=True)

                if not flow_only:
                    viab.plot_areas(grid, states)
                if ARGS.title:
                    plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(rhs, [plot_x_limits, plot_y_limits], topo.styleDefault) for rhs, parameters in zip(default_PSs, default_parameters)]
                [plotPS(rhs, [plot_x_limits, plot_y_limits], style) for rhs, parameters, style in zip(management_PSs, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.axis("equal")

                plt.xlim(plot_x_limits)
                plt.ylim(plot_y_limits)

                if save_to:
                    save_figure(save_to)

        print()
        viab.print_evaluation(states)

    return example_function


EXAMPLES = {
            "finite-time-lake":
                generate_example([ftlm.rhs_default],
                                 [ftlm.rhs_management],
                                 ftlm.sunny,
                                 [[-5, 5],[-5, 5]],
                                 out_of_bounds=True,
                                 default_rhssPS=[ftlm.rhs_default_PS],
                                 management_rhssPS=[ftlm.rhs_management_PS],
                                 ),
            "finite-time-lake2":
                generate_example(
                                 [ftlm2.rhs_default],
                                 [ftlm2.rhs_management],
                                 ftlm2.sunny,
                                 [[-5, 5],[-5, 5]],
                                 out_of_bounds=True,
                                 xlabel="$x$",
                                 ylabel="$y$",
                                 ),
            "aw-model-dg":
                generate_example([awm.AW_rescaled_rhs],
                                 [awm.AW_rescaled_rhs],
                                 awm.AW_rescaled_sunny,
                                 [[1e-3, 1 - 1e-3],[1e-3, 1 - 1e-3]],
                                 default_parameters=[{"beta":awm.beta_default, "theta":awm.theta_default}],
                                 management_parameters=[{"beta":awm.beta_DG, "theta":awm.theta_default}],
                                 out_of_bounds=False,
                                 xlabel=r"excess atmospheric carbon $A$ [GtC]",
                                 ylabel=r"economic production $Y$ [trillion US\$]",
                                 set_ticks=awm.set_ticks,
                                 stepsize=0.055,
                                 ),
            "aw-model-dg-bifurc":
                generate_example([awm.AW_rescaled_rhs],
                                 [awm.AW_rescaled_rhs],
                                 awm.AW_rescaled_sunny,
                                 [[1e-3, 1 - 1e-3],[1e-3, 1 - 1e-3]],
                                 default_parameters=[{"beta":awm.beta_default, "theta":awm.theta_default}],
                                 management_parameters=[{"beta":0.035, "theta":awm.theta_default}],
                                 out_of_bounds=False,
                                 compute_eddies=True,
                                 xlabel=r"excess atmospheric carbon $A$ [GtC]",
                                 ylabel=r"economic production $Y$ [trillion US\$]",
                                 set_ticks=awm.set_ticks,
                                 ),
            "aw-model-srm":
                generate_example([awm.AW_rescaled_rhs],
                                 [awm.AW_rescaled_rhs],
                                 awm.AW_rescaled_sunny,
                                 [[1e-8, 1 - 1e-8],[1e-8, 1 - 1e-8]],
                                 default_parameters=[{"beta":awm.beta_default, "theta":awm.theta_default}],
                                 management_parameters=[{"beta":awm.beta_default, "theta":awm.theta_SRM}],
                                 out_of_bounds=False,
                                 compute_eddies=True,
                                 ),
            ## The Pendulum example was taken out, because it is hamiltonian, making the whole algorithm getting unstable.
            ## This would be a future task to fix with an algorithm that does not simply linearly approximate.
            # "pendulum":
            #     generate_example([gpm.pendulum_rhs],
            #                      [gpm.pendulum_rhs],
            #                      gpm.pendulum_sunny,
            #                      [[0, 2*np.pi],[-2.2,1.2]],
            #                      default_parameters=[{"a":0.0}],
            #                      management_parameters=[{"a":0.6}],
            #                      periodicity=[1, -1],
            #                      compute_eddies=True,
            #                      rescaling_epsilon=1e-3,
            #                      ),
            "swing-eq":
                generate_example([sqm.swing_rhs],
                                 [sqm.swing_rhs],
                                 sqm.swing_sunny,
                                 [[-0.5*np.pi, 1.5*np.pi],[-1, 1]],
                                 default_parameters=[{"alpha":0.2, "P":0.3, "K":0.5}],
                                 management_parameters=[{"alpha":0.2, "P":0.0, "K":0.5}],
                                 periodicity=[1, -1],
                                 compute_eddies=False,
                                 rescaling_epsilon=1e-3,
                                 out_of_bounds=False, # set because it creates a nice picture for these specific parameters
                                 stepsize=0.035,
                                 ),
            "plants":
                generate_example([pm.plants_rhs],
                                 [pm.plants_rhs]*2,
                                 pm.plants_sunny,
                                 [[0, 1],[0, 1]],
                                 default_parameters=[{"ax":0.2, "ay":0.2, "prod":2}],
                                 management_parameters=[{"ax":0.1, "ay":0.1, "prod":2}, {"ax":2, "ay":0, "prod":2}],
                                 out_of_bounds=False,
                                 stepsize=0.035,
                                 ),
            ## Taken out because it contains a critical point.
            # "tech-change":
            #     generate_example([tcm.techChange_rhs],
            #                      [tcm.techChange_rhs],
            #                      tcm.techChange_sunny,
            #                      [[0, 1], [0, 2]],
            #                      default_parameters=[
            #                          dict(rvar = 1, pBmin = 0.15, pE = 0.3, delta = 0.025, smax = 0.3, sBmax = 0.)],
            #                      management_parameters=[
            #                          dict(rvar = 1, pBmin = 0.15, pE = 0.3, delta = 0.025, smax = 0.3, sBmax = 0.5)],
            #                      management_rhssPS = [tcm.techChange_rhsPS],
            #                      ),
            "easter-a":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=1000, yMinimal=3000),
                                 [[0, 35000],[0, 18000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 4 * 10 ** (-6), delta = -0.1, kappa = 12000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 2.8 * 10 ** (-6), delta = -0.1, kappa = 12000)],
                                 out_of_bounds=[[False, True], [False, True]],
                                 ),
            "easter-b":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=1200, yMinimal=2000),
                                 [[0, 9000], [0, 9000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 13.6 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 ),
            "easter-c":
                generate_example([prm.easter_rhs],
                                [prm.easter_rhs],
                                ft.partial(prm.easter_sunny, xMinimal=4000, yMinimal=3000),
                                [[0, 9000],[0, 9000]],
                                default_parameters=[
                                    dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                management_parameters=[
                                    dict(phi = 4, r = 0.04, gamma = 16 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                compute_eddies=True,
                                ),
            "easter-d":
                generate_example([prm.easter_rhs],
                                 [prm.easter_rhs],
                                 ft.partial(prm.easter_sunny, xMinimal=4000, yMinimal=3000),
                                 [[0, 9000], [0, 9000]],
                                 default_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 8 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 management_parameters=[
                                     dict(phi = 4, r = 0.04, gamma = 11.2 * 10 ** (-6), delta = -0.15, kappa = 6000)],
                                 compute_eddies=True,
                                 ),
            "consum":
                generate_example([],
                                 [cm.consum_rhs]*2,
                                 cm.consum_sunny,
                                 [[0, 2], [0, 3]],
                                 default_parameters = [],
                                 management_parameters = [dict(u = -0.5),
                                                       dict(u = 0.5)],
                                 management_rhssPS = [cm.consum_rhsPS]*2,
                                 ),

}


AVAILABLE_EXAMPLES = sorted(EXAMPLES)


assert not "all" in AVAILABLE_EXAMPLES
MODEL_CHOICES = ["all"] + AVAILABLE_EXAMPLES
GRID_CHOICES = ["orthogonal", "simplex-based"]
PLOT_CHOICES = ["points", "areas"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""A test script for the standard examples
        
        If you would like to know more details on the actual meaning of the examples,
        please contact the author. Generally, you can understand the dynamics of the 
        models by carefully analyzing the flows, that are plotted. The default flow
        is shown with think lines in light blue, the management flows with thin, dark blue,
        dotted (or dashed) lines.
        """
    )

    parser.add_argument("models", metavar="model", nargs="+",
                        choices=MODEL_CHOICES,
                        help="the model to be run or all; space separated list\n"
                        "allowed values are: " + ", ".join(MODEL_CHOICES))

    parser.add_argument("-b", "--no-backscaling", action="store_false", dest="backscaling",
                        help="omit backscaling after the topology/viability computation")
    parser.add_argument("-f", "--force", action="store_true",
                        help="overwrite existing files")
    parser.add_argument("--flow-only", action="store_true",
                        help="plot only the models flow, nothing else")
    parser.add_argument("--follow", nargs=2, metavar=("point", "dist"),
                        help="follow the points that are at most 'dist' away from 'point")
    parser.add_argument("-g", "--grid", choices=GRID_CHOICES, default=GRID_CHOICES[0],
                        help="grid type")
    parser.add_argument("--hidpi", action="store_true",
                        help="fix some things so everything looks okai on Hi-DPI screens")
    parser.add_argument("-i", "--integrate", action="store_const", dest="run_type",
                        const="integration", default="linear",
                        help="integrate instead of using linear approximation")
    parser.add_argument("--mark-fp", nargs=1, metavar="fp-approximation",
                        help="mark the fixed point of the dynamics which is close to 'fp-approximation'")
    parser.add_argument("-n", "--num", type=int, default=80,
                        help="number of points in each dimension")
    parser.add_argument("--no-title", dest="title", action="store_false",
                        help="remove the title from the plot")

    parser.add_argument("--paper", action="store_true",
                        help="create a plot that has been used for the paper")

    parser.add_argument("-p", "--plot", choices=PLOT_CHOICES, default=PLOT_CHOICES[0],
                        help="how to plot the results")
    parser.add_argument("-r", "--remember", action="store_true",
                        help="remember already calculated values in a dict" \
                        " (might be slow for a large grids)")
    parser.add_argument("-s", "--save", metavar="output-file", nargs="?", default="",
                        help="save the picture; if no 'output-file' is given, a name is generated")
    parser.add_argument("--stop-when-finished", default=lv.TOPOLOGY_STEP_LIST[-1], metavar="computation-step",
                        choices=lv.TOPOLOGY_STEP_LIST,
                        help="stop when the computation of 'computation-step' is finished, choose from: " ", ".join(lv.TOPOLOGY_STEP_LIST) ) 
    parser.add_argument("--no-numba", dest="use_numba", action="store_false",
                        help="do not use numba jit-compiling")


    if with_argcomplete:
        # use argcomplete auto-completion
        argcomplete.autocomplete(parser)

    ARGS = parser.parse_args()
    if "all" in ARGS.models:
        ARGS.models = AVAILABLE_EXAMPLES

    if len(ARGS.models) > 1 and ARGS.save:
        parser.error("computing multiple models but giving only one file name " \
                     "where the pictures should be save to doesn't make sense " \
                     "(to me at least)")

    if ARGS.paper:
        ARGS.hidpi = True
        ARGS.title = False
        ARGS.num = 200
        mpl.rcParams["axes.labelsize"] = 36
        mpl.rcParams["xtick.labelsize"] = 32
        mpl.rcParams["ytick.labelsize"] = 32
        ARGS.mark_fp = "[0.5,0.5]"

    if ARGS.mark_fp is not None:
        ARGS.mark_fp = np.array(eval(ARGS.mark_fp))

    for model in ARGS.models:
        save_to = ARGS.save
        if save_to is None:  # -s or --save was set, but no filename was given
            save_to = "_".join([model, ARGS.grid, ARGS.plot]) + ".jpg"

        print()
        print("#"*80)
        print("computing example: " + model)
        print("#"*80)
        EXAMPLES[model](model,
                    grid_type=ARGS.grid,
                    backscaling=ARGS.backscaling,
                    plotting=ARGS.plot,
                    run_type=ARGS.run_type,
                    save_to=save_to,
                    n0=ARGS.num,
                    hidpi=ARGS.hidpi,
                    use_numba=ARGS.use_numba,
                    stop_when_finished=ARGS.stop_when_finished,
                    flow_only=ARGS.flow_only,
        )

    plt.show()

