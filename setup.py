# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requires = open('requirements.txt').read().strip().split('\n')

setup(
    name='intake-mnist-data',
    version='0.0.1',
    description='MNIST digits data and plugins for Intake',
    url='https://github.com/ContinuumIO/intake-mnist-data',
    maintainer='Martin Durant',
    maintainer_email='mdurant@anaconda.com',
    license='BSD',
    py_modules=['intake_mnist'],
    packages=find_packages(),
    package_data={'': ['*.csv', '*.yml', '*.yaml']},
    include_package_data=True,
    install_requires=requires,
    long_description=open('README.rst').read(),
    zip_safe=False, )
