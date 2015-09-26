#!/usr/bin/env python
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
import numpy as np

# Check if cython exists, then use it. Otherwise compile already cythonized cpp file
have_cython = False
try:
    from Cython.Build import cythonize
    have_cython = True
except ImportError:
    pass

if have_cython:
    cpp_extension = cythonize(Extension('pixel_clusterizer.clusterizer', ['pixel_clusterizer/clusterizer.pyx', 'pixel_clusterizer/cpp/Clusterizer.cpp', 'pixel_clusterizer/cpp/Basis.cpp']))
else:
    cpp_extension = [Extension('pixel_clusterizer.clusterizer',
                               sources=['pixel_clusterizer/cpp/clusterizer.cpp'],
                               language="c++")]

version = '1.0.3'
author = 'David-Leon Pohl'
author_email = 'pohl@physik.uni-bonn.de'

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='pixel_clusterizer',
    version=version,
    description='A clusterizer to cluster hits of a pixel detector with Python. The clustering happens in C++ on numpy arrays to increase the speed.',
    url='https://github.com/SiLab-Bonn/pixel_clusterizer',
    license='GNU LESSER GENERAL PUBLIC LICENSE Version 2.1',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cpp_extension,
    include_dirs=[np.get_include()],
    keywords=['cluster', 'clusterizer', 'pixel'],
    platforms='any'
)
