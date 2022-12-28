# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

"""
"pip install pygame" error warning:
if you're facing trouble installing pygame with Python 3.11, you need to install with following command:
'pip install pygame --pre'

More discussion:
https://stackoverflow.com/questions/74188013/python-pygame-not-installing
"""

setup(
    name='bettermdptoolbox_v2',
    version='1.0',
    platforms=['Any'],
    license='New BSD',
    packages=[''],
    install_requires=['gym>=0.26, <=0.26.2', 'pygame', 'numpy', 'tqdm'],
)
