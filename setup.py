from setuptools import setup
from Cython.Build import cythonize

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

setup(
    package_dir={'core': 'src/core'},
    ext_modules=cythonize(
        ["src/core/voronoi_cython.pyx"]),
)
