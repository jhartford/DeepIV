#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "keras==2.3.1",
    "tensorflow==2.5",
    "sklearn",  # required for comparing to linear
    "h5py"  # required for saving models
]

optimal_packages = {
    "nonpar": ["rpy2"]
}

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='deepiv',
    version='0.1.2',
    description="A package for counterfactual prediction using deep instrument variable methods",
    long_description=readme + '\n\n' + history,
    author="Jason Hartford, Xingrui Wang",
    author_email='jasonhar@cs.ubc.ca',
    url='https://github.com/jhartford/deepiv',
    packages=[
        'deepiv',
    ],
    package_dir={'deepiv':
                 'deepiv'},
    include_package_data=True,
    install_requires=requirements,
    extras_require=optimal_packages,
    license="MIT license",
    zip_safe=False,
    keywords='deepiv',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
