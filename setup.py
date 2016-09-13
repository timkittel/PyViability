from setuptools import setup



setup(name="pyviability",
      version="0.11",
      description="a library for computation of viabilty sets and TSM sets",
      url="not yet there",
      author="Tim Kittel",
      author_email="Tim.Kittel@pik-potsdam.de",
      license="whatever for now",
      packages=["pyviability"],
      install_requires=[
          # "python>=3.5.1",
          "numpy>=1.11.0",
          "numba>=0.28.1",
          "scipy>=0.17.0",
          "matplotlib>=1.5.1",
      ],
      zip_safe=False)




