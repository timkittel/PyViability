# please add documentation!
# and add which pep8 stuff we want to ignore

import pkg_resources

version = __version__ = pkg_resources.get_distribution(__package__).version
"version string"
version_info = __version_info__ = tuple(map(int, __version__.split(".")))
"version tuple"

del pkg_resources

from .libviability import backscaling_grid, \
    generate_grid, \
    make_run_function, \
    plot_points, \
    plot_areas, \
    print_evaluation, \
    topology_classification, \
    scaled_to_one_sunny


