'''
This file is used to install the package using pip.
'''

from setuptools import setup, find_packages

setup(
    name='crps_neighboorhood_based',
    version='0.0.1',
    description='A package to calculate the neighborhood-based CRPS: Stein and Stoop (2022)',
    author='Alireza Amani',
    author_email='alireza.amani101@gmail.com',
    packages=find_packages(),  # Finds packages automatically
    install_requires=[  # List any dependencies here
        'numpy',

    ],
)
