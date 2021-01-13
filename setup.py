from setuptools import setup
from Cython.Build import cythonize

setup(
    package_dir={'core': 'src/core'},
    ext_modules=cythonize(["src/core/*.pyx"],  language_level = "3"),
)
