# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

# NOTE:
# pygame may require `pip install pygame --pre` on Python 3.11+
# See README for details

setup(
    name="bettermdptools",
    url="https://github.com/jlm429/bettermdptools",
    version="0.8.6",
    platforms=["Any"],
    license="New BSD",
    author="John Mansfield",
    author_email="jlm429@gmail.com",
    description="so much room for activities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,

    # --- added ---
    python_requires=">=3.10,<4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    # -------------

    install_requires=[
        "gymnasium>=0.26,<0.27",
        "pygame",
        "numpy<2",
        "tqdm",
        "pandas",
        "seaborn",
        "matplotlib",
    ],
)
