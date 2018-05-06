import os
from setuptools import setup, find_packages


setup(
    name="opf_split",
    version="0.1",
    author="Luiz Felix",
    author_email="lzcfelix@gmail.com",
    description=("Utility to sync numpy and OPF folds to train classifiers."),
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
	'plac==0.9.6',
	'numpy==1.14.2'
    ]
)

