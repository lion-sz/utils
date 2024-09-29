Hyperparameter Tuning
"""""""""""""""""""""

There are two ways to use the optimization framework,
either through the cli or the :func:`hp_tune.tune` method.

The CLI implements the `tune` and `visualize` methods.

.. typer:: hp_tune:app
    :width: 100
    :show-nested:


This module contains a function to tune hyperparameters.
The primary entry point is the :func:`hp_tune.tune` function.

.. autofunction:: hp_tune.tune


Tuning Methods
''''''''''''''

For now I've implemented primarily hyperband based methods and random search.

.. autofunction:: hp_tune.models.random_search
.. autofunction:: hp_tune.models.hyperband
.. autofunction:: hp_tune.models.bohb
.. autofunction:: hp_tune.models.hyperband_est_resources
