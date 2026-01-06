# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

# NOTE:
# pygame may require `pip install pygame --pre` on Python 3.11+
# See README for details

setup(
    name="bettermdptools",
    url="https://github.com/jlm429/bettermdptools",
    version="0.8.2",
    platforms=["Any"],
    license="New BSD",
    author="John Mansfield",
    author_email="jlm429@gmail.com",
    description="so much room for activities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gymnasium>=0.26,<0.27",
        "pygame",
        "numpy",
        "tqdm",
        "pandas",
        "seaborn",
        "matplotlib>=3.7.0,<=3.8.0",
    ],
)
