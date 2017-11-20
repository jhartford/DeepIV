======
DeepIV
======


.. image:: https://img.shields.io/pypi/v/deepiv.svg
        :target: https://pypi.python.org/pypi/deepiv

.. image:: https://img.shields.io/travis/jhartford/deepiv.svg
        :target: https://travis-ci.org/jhartford/deepiv

.. image:: https://readthedocs.org/projects/deepiv/badge/?version=latest
        :target: https://deepiv.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/jhartford/deepiv/shield.svg
     :target: https://pyup.io/repos/github/jhartford/deepiv/
     :alt: Updates


IMPORTANT: Newer versions of Keras have broken this implementation. This code currently only support Keras 2.0.6 (which is what will be installed if you use the pip install instructions described below).

A package for counterfactual prediction using deep instrument variable methods that builds on Keras_. 

You can read how the method works in our DeepIV_ paper.

If you use this package in your research, please cite it as::

        @inproceedings{Hartford17,
        author    = {Jason Hartford and
                Greg Lewis and
                Kevin Leyton-Brown and
                Matt Taddy},
        title     = {Deep IV: A Flexible Approach for Counterfactual Prediction},
        booktitle = {Proceedings of the 34th International Conference on Machine Learning,
                {ICML} 2017, Sydney, Australia, 6-11 August 2017},
        pages     = {1--9},
        year      = {2017}
        }


* Free software: MIT license
* Documentation: https://deepiv.readthedocs.io.


Installation
--------
To use DeepIV, you can simply naviage to to the DeepIV directory on your machine and run:

        pip install .

You can then use the package by simply running: import deepiv in python. See the examples directory for example usage.

The package is currently under active development, so you may want to install it using the following command:

        pip install -e .

By doing this, every time you git pull an update, it will be reflected in your installation.


Usage
--------
The DeepIV package is simply a subclass of the Keras Model class that provides the necessary functions for fitting Deep instrumental variable models. Because of this, you can think of it as a drop-in replacement of the Keras Model object.
The DeepIV procedure consists of two stages: 
1. Fit the Treatment model.
2. Fit the Response model that takes the fitted Treatment model as input. 

Example usage is shown in the experiments directory. 

``demand_simulation.py`` gives a simple example using a feedforward network for both the treatment and the response models.

``demand_simulation_mnist.py`` is a little more complicated: it uses a convolutation network to fit an image embedding and then concatinates the embedding with other features to fit the network. 

Both those examples use simulated data where ground truth is known, so they can report the causal mean squared error. On real data this isn't possible, so we advise that you use a holdout set to tune hyperparameters of the network (or cross validation in the case of small networks). You can choose hyperparameters based on the losses returned at each stage (see the paper for details on why this works).

DeepIV should be compatable with all Keras layers, so the Keras_ documentation is a good place to learn about designing network architectures. Feel free to file a bug report if something doesn't work.


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _DeepIV: http://proceedings.mlr.press/v70/hartford17a.html
.. _Keras: https://keras.io
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

