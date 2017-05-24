#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "keras==1.1.2",
    "theano"
]

requirements = [
    "keras",
    "tensorflow"
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='deepiv',
    version='0.1.0',
    description="A package for counter factual prediction using deep instrument variable methods",
    long_description=readme + '\n\n' + history,
    author="Jason Hartford",
    author_email='jasonhar@cs.ubc.ca',
    url='https://github.com/jhartford/deepiv',
    packages=[
        'deepiv',
    ],
    package_dir={'deepiv':
                 'deepiv'},
    include_package_data=True,
    install_requires=requirements,
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
    ],
    test_suite='tests',
    tests_require=test_requirements
)
