Discretize the true problem
===========================

When the true problem is stage-wise independent continuous or Markovian continuous
, it needs to be discretized.

Stage-wise independent continuous problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stage-wise independent continuous demand process can be discretized simply
by specifying the sample size and the random seed.

.. ipython:: python

    from msppy.utils.examples import construct_nvic
    nvic = construct_nvic()
    nvic.discretize(random_state=888, n_samples=100)

Markovian continuous problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following snippet discretizes the Markovian uncertainty by a ten dimensional
Markov chain using stochastic approximation method (stochastic gradient descent)
with 1000 iterations.

.. ipython:: python

    from msppy.utils.examples import construct_nvm
    nvm = construct_nvm()
    nvm.discretize(n_Markov_states=10, n_sample_paths=1000, method='SA');

Module Reference
~~~~~~~~~~~~~~~~
.. currentmodule:: msppy.msp.MSLP
.. autofunction:: discretize
