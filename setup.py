
# run 'pip install -e .' in the command line for installation

from __future__ import print_function  # just for printing the error if python2.x is used

from setuptools import setup

required_python_version = (3, 5)
required_python_string=">=" + ".".join(map(str, required_python_version))


################################################################################
# small hack as the 'python_requires' keyword argument of 'setup' seems to be
# ignored

import sys

# check that at least python 'required_python_version' is used
if sys.version_info < required_python_version:
    print("Please use at least Python " + ".".join(map(str, required_python_version)),
          " (not '" + sys.version + "')")
    sys.exit(1)
################################################################################


setup(name="pyviability",
      version="0.2.1",
      description="a library for computation of viabilty sets and TSM sets",
      url="https://timkittel.github.io/PyViability/",
      author="Tim Kittel",
      author_email="Tim.Kittel@pik-potsdam.de",
      license="BSD 2-Clause",
      packages=["pyviability"],
      python_requires=required_python_string,
      install_requires=[
          "numpy>=1.11.0",
          "numba>=0.28.1",
          "scipy>=0.17.0",
          "matplotlib>=1.5.1", # for plotting the examples only
          "pillow>=4.1.1",     # for saving the example plots only
          "argcomplete>=1.0.0",
      ],
      package_data={
          'colorfile': ['colors.soc'],
      },
      zip_safe=False)


