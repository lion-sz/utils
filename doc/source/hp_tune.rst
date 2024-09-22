Hyperparameter Tuning
"""""""""""""""""""""

This module contains a function to tune hyperparameters.
The primary entry point is the :func:`hp_tune.tune` function.

.. autofunction:: hp_tune.tune

For now I've only implemented the hyperband algorithm.

.. autofunction:: hp_tune.hyperband
.. autofunction:: hp_tune.hyperband_est_resources


