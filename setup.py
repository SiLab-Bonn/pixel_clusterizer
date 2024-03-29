#!/usr/bin/env python
from setuptools import setup, find_packages  # This setup relies on setuptools since distutils is insufficient and badly hacked code

version = '3.1.7'
author = 'Yannick Dieter, David-Leon Pohl, Jens Janssen'
author_email = 'dieter@physik.uni-bonn.de, pohl@physik.uni-bonn.de, janssen@physik.uni-bonn.de'

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='pixel_clusterizer',
    version=version,
    description='A fast, generic, and easy to use clusterizer to cluster hits of a pixel matrix in Python.',
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
    keywords=['cluster', 'clusterizer', 'pixel'],
    python_requires='>=2.7',
    platforms='any'
)
