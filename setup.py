
# run 'pip install -e .' in the command line for installation

from setuptools import setup


required_python_version = (3, 6)
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
      version="0.11",
      description="a library for computation of viabilty sets and TSM sets",
      url="not yet there",
      author="Tim Kittel",
      author_email="Tim.Kittel@pik-potsdam.de",
      license="whatever for now",
      packages=["pyviability"],
      python_requires=required_python_string,
      install_requires=[
          # "python_version>=3.5.0",
          "numpy>=1.11.0",
          "numba>=0.28.1",
          "scipy>=0.17.0",
          "matplotlib>=1.5.1",
      ],
      zip_safe=False)




