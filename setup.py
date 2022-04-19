# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='bettermdptoolbox',
    version='1.0',
    platforms=['Any'],
    license='New BSD',
    packages=[''],
    install_requires=['numpy', 'scipy', 'gym', 'ipython'],
    setup_requires=['pytest-runner'],
)
