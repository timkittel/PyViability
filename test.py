
from __future__ import division, print_function


import PyViability as viab
import helper

import PlantModel as pm
import TechChangeModel as tcm
import PopulationAndResourceModel as prm
import GravityPendulumModel as gpm
import ConsumptionModel as cm
import AWModel as awm

import PTopologyL as topo

import myPhaseSpaceL as mPS

import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import functools as ft
import numba as nb


def generate_example(default_rhss,
                     management_rhss,
                     sunny_fct,
                     boundaries,
                     default_parameters=[],
                     management_parameters=[],
                     n0=80,
                     periodicity=[],
                     default_rhssPS=None,
                     management_rhssPS=None,
                     out_of_bounds=True,
                     compute_eddies=False,
                     ):

    plotPS = lambda rhs, boundaries, style: mPS.plotPhaseSpace(rhs, [boundaries[0][0], boundaries[1][0], boundaries[0][1], boundaries[1][1]], colorbar=False, style=style)

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
                         plot_points=True,
                         plot_areas=False,
                         run_type="integration",
                         save=False,
            ):
        grid, scaling_factor,  offset, _ = viab.generate_grid(boundaries,
                                                        n0,
                                                        grid_type,
                                                        periodicity = periodicity) #noqa
        states = np.zeros(grid.shape[:-1], dtype=np.int16)

        NB_NOPYTHON = True
        default_runs = [viab.make_run_function(nb.jit(rhs, nopython=NB_NOPYTHON), helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning=run_type) for rhs, parameters in zip(default_rhss, default_parameters)] #noqa
        management_runs = [viab.make_run_function(nb.jit(rhs, nopython=NB_NOPYTHON), helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning=run_type) for rhs, parameters in zip(management_rhss, management_parameters)] #noqa

        sunny = viab.scaled_to_one_sunny(sunny_fct, offset, scaling_factor)

        # adding the figure here already if VERBOSE is set
        # this makes only sense, if backscaling is switched off
        if (not backscaling) and plot_points:
            figure_size = (15, 5 * np.sqrt(3) if grid_type == "simplex-based" else 15)
            fig = plt.figure(figsize=figure_size, tight_layout=True)

        start_time = time.time()
        viab.topology_classification(grid, states, default_runs, management_runs, sunny,
                                     periodic_boundaries = periodicity,
                                     grid_type=grid_type,
                                     compute_eddies=compute_eddies,
                                     out_of_bounds=out_of_bounds)
        time_diff = time.time() - start_time

        print("run time: {!s} s".format(dt.timedelta(seconds=time_diff)))

        if backscaling:
            grid = viab.backscaling_grid(grid, scaling_factor, offset)

            if plot_points:
                fig = plt.figure(figsize=(15, 15), tight_layout=True)

                viab.plot_points(grid, states)
                plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(ft.partial(rhs, **parameters), boundaries, topo.styleDefault) #noqa
                    for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
                [plotPS(ft.partial(rhs, **parameters), boundaries, style)
                    for rhs, parameters, style in zip(management_rhssPS, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.xlim(xlim)
                plt.ylim(ylim)

                if ARGS.save:
                    fig.savefig(example + "-points.jpg")


            if plot_areas:
                fig = plt.figure(figsize=(15, 15), tight_layout=True)

                viab.plot_areas(grid, states)
                plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(ft.partial(rhs, **parameters), boundaries, topo.styleDefault) #noqa
                    for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
                [plotPS(ft.partial(rhs, **parameters), boundaries, style)
                    for rhs, parameters, style in zip(management_rhssPS, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.xlim(xlim)
                plt.ylim(ylim)

                if ARGS.save:
                    fig.savefig(example + "-areas.jpg")

        else:
            plot_x_limits = [0, 1.5 if grid_type == "simplex-based" else 1]
            plot_y_limits = [0, np.sqrt(3)/2 if grid_type == "simplex-based" else 1]

            default_PSs = [viab.make_run_function(rhs, helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning="PS") #noqa
                            for rhs, parameters in zip(default_rhssPS, default_parameters)] #noqa
            management_PSs = [viab.make_run_function(rhs, helper.get_ordered_parameters(rhs, parameters), offset, scaling_factor, returning="PS") #noqa
                                for rhs, parameters in zip(management_rhssPS, management_parameters)] #noqa

            if plot_points:
                # figure already created above

                viab.plot_points(grid, states)
                plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(rhs, [plot_x_limits, plot_y_limits], topo.styleDefault) for rhs, parameters in zip(default_PSs, default_parameters)]
                [plotPS(rhs, [plot_x_limits, plot_y_limits], style) for rhs, parameters, style in zip(management_PSs, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.axis("equal")

                plt.xlim(plot_x_limits)
                plt.ylim(plot_y_limits)

                if ARGS.save:
                    fig.savefig(example + "-points.jpg")


            if plot_areas:
                fig = plt.figure(figsize=(15, 15), tight_layout=True)

                viab.plot_areas(grid, states)
                plt.gca().set_title('example: ' + example_name, fontsize=20)

                [plotPS(rhs, [plot_x_limits, plot_y_limits], topo.styleDefault) for rhs, parameters in zip(default_PSs, default_parameters)]
                [plotPS(rhs, [plot_x_limits, plot_y_limits], style) for rhs, parameters, style in zip(management_PSs, management_parameters, [topo.styleMod1, topo.styleMod2])] #noqa

                plt.axis("equal")

                plt.xlim(plot_x_limits)
                plt.ylim(plot_y_limits)

                if ARGS.save:
                    fig.savefig(example + "-areas.jpg")

        num = states.size
        num_len = len(str(num))
        for region in ["UNSET", "SHELTER", "GLADE", "LAKE", "SUNNY_UP", "DARK_UP", "BACKWATERS", "SUNNY_DOWN", "DARK_DOWN", "SUNNY_EDDIES", "DARK_EDDIES", "SUNNY_ABYSS", "DARK_ABYSS", "TRENCH"]:
            print(("{:<15}: {:>6.2f}%").format(region, np.count_nonzero(states == getattr(viab, region)) / num * 100))

    return example_function


EXAMPLES = {
            "aw-model":
                generate_example([awm.AW_rescaled_rhs],
                                 [awm.AW_rescaled_rhs],
                                 awm.AW_rescaled_sunny,
                                 [[1e-8, 1 - 1e-8],[1e-8, 1 - 1e-8]],
                                 default_parameters=[{"beta":awm.beta_default, "theta":awm.theta_default}],
                                 management_parameters=[{"beta":awm.beta_DG, "theta":awm.theta_default}],
                                 out_of_bounds=False,
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
            "pendulum":
                generate_example([gpm.pendulum_rhs],
                                 [gpm.pendulum_rhs],
                                 gpm.pendulum_sunny,
                                 [[0, 2*np.pi],[-2.2,1.2]],
                                 default_parameters=[{"a":0.0}],
                                 management_parameters=[{"a":0.6}],
                                 periodicity=[1, -1],
                                 compute_eddies=True,
                                 ),
            "plants":
                generate_example([pm.plants_rhs],
                                 [pm.plants_rhs]*2,
                                 pm.plants_sunny,
                                 [[0, 1],[0, 1]],
                                 default_parameters=[{"ax":0.2, "ay":0.2, "prod":2}],
                                 management_parameters=[{"ax":0.1, "ay":0.1, "prod":2}, {"ax":2, "ay":0, "prod":2}],
                                 out_of_bounds=False,
                                 ),
            "tech-change":
                generate_example([tcm.techChange_rhs],
                                 [tcm.techChange_rhs],
                                 tcm.techChange_sunny,
                                 [[0, 1], [0, 2]],
                                 default_parameters=[
                                     dict(rvar = 1, pBmin = 0.15, pE = 0.3, delta = 0.025, smax = 0.3, sBmax = None)],
                                 management_parameters=[
                                     dict(rvar = 1, pBmin = 0.15, pE = 0.3, delta = 0.025, smax = 0.3, sBmax = 0.5)],
                                 management_rhssPS = [tcm.techChange_rhsPS],
                                 ),
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

    parser = argparse.ArgumentParser(description="test script for the standard examples")

    parser.add_argument("models", metavar="model", nargs="+",
                        choices=MODEL_CHOICES,
                        help="the model to be run or all; space separated list\n"
                        "allowed values are: " + ", ".join(MODEL_CHOICES))

    parser.add_argument("-b", "--no-backscaling", action="store_false", dest="backscaling",
                        help="omit backscaling after the topology/viability computation")
    parser.add_argument("-f", "--force", action="store_true",
                        help="overwrite existing files")
    parser.add_argument("-g", "--grid", choices=GRID_CHOICES, default=GRID_CHOICES[0],
                        help="grid type")
    parser.add_argument("-i", "--integrate", action="store_const", dest="run_type",
                        const="integration", default="linear",
                        help="integrate instead of using linear approximation")
    parser.add_argument("-p", "--plot", metavar="plot-style", nargs="+", choices=PLOT_CHOICES, default=PLOT_CHOICES[:1],
                        help="how to plot the results: "+", ".join(PLOT_CHOICES))
    parser.add_argument("-r", "--remember", action="store_true",
                        help="remember already calculated values in a dict" \
                        " (might be slow for a large grids)")
    parser.add_argument("-s", "--save", metavar="output-file", nargs="?", default="",
                        help="save the picture; if no 'output-file' is given, a name is generated")


    ARGS = parser.parse_args()
    print(ARGS)

    for model in ARGS.models:
        print()
        print("#"*40)
        print("computing example: " + model)
        print("#"*40)
        EXAMPLES[model](model,
                    grid_type=ARGS.grid,
                    backscaling=ARGS.backscaling,
                    plot_points=("points" in ARGS.plot),
                    plot_areas=("areas" in ARGS.plot),
                    run_type=ARGS.run_type,
        )

    plt.show()

