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
    name='bettermdptools',
    url='https://github.com/jlm429/bettermdptools',
    version='0.5.1',
    platforms=['Any'],
    license='New BSD',
    author='John Mansfield',
    author_email='jlm429@gmail.com',
    packages=["algorithms", "utils"],
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['gymnasium>=0.26, <=0.26.3', 'pygame', 'numpy', 'tqdm', 'pandas', 'seaborn', 'matplotlib'],
)
