#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import numpy
from scipy import stats
import numbers
from collections import abc
import gurobipy


def compute_CI(array, percentile):
    """Compute percentile% CI for the given array."""
    if len(array) == 1:
        raise NotImplementedError
    mean = numpy.mean(array)
    # standard error
    se = numpy.std(array, ddof=1) / numpy.sqrt(len(array))
    # critical value
    cv = (
        stats.t.ppf(1 -(1-percentile/100)/2, len(array)-1)
        if len(array) != 1
        else 0
    )
    return mean - cv * se, mean + cv * se

def MA(array, window):
    weights = numpy.repeat(1, window)/window
    return numpy.convolve(array,weights,'valid')

def exp_MA(array, window):
    weights = numpy.exp(numpy.linspace(-1,0,window))
    weights /= sum(weights)
    return numpy.convolve(array,weights,'valid')

def rand_int(k, random_state, probability=None, size=None, replace=None):
    """Randomly generate certain numbers of sample from range(k) with given
    probability with/without replacement"""
    if probability is None and replace is None:
        return random_state.randint(low=0, high=k, size=size)
    else:
        return random_state.choice(a=k, p=probability, size=size, replace=replace)

def check_random_state(seed):
    """Turn the seed into a RandomState instance.

    Parameters & Returns
    --------------------
    seed : None, numpy.random, int, instance of RandomState
        If None, return numpy.random.
        If int, return a new RandomState instance with seed.
        Otherwise raise ValueError.
    """
    if seed in [None, numpy.random]:
        return numpy.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, numpy.integer)):
        return numpy.random.RandomState(seed)
    if isinstance(seed, numpy.random.RandomState):
        return seed
    raise ValueError(
        "{%r} cannot be used to seed a numpy.random.RandomState instance"
            .format(seed)
    )

def check_Markov_states_and_transition_matrix(
        Markov_states,
        transition_matrix,
        T):
    """Check Markov states and transition matrix are in the right form. Return
    the dimension of MC and the number of Markov states."""
    n_Markov_states = []
    dim_Markov_states = []
    if len(transition_matrix) != T:
        raise ValueError(
            "The transition_matrix is of length {}, expecting of length {}!"
            .format(len(transition_matrix), T)
        )
    if len(Markov_states) != T:
        raise ValueError(
            "The Markov_states is of length {}, expecting of length {}!"
            .format(len(Markov_states), T)
        )
    a = 1
    for t, item in enumerate(transition_matrix):
        if a != numpy.array(item).shape[0]:
            raise ValueError("Invalid transition_matrix!")
        else:
            a = numpy.array(item).shape[1]
            n_Markov_states.append(a)
        for single in item:
            if round(sum(single),4) != 1:
                raise ValueError("Probability does not sum to one!")
    for t, item in enumerate(Markov_states):
        shape = numpy.array(item).shape
        if shape[0] != n_Markov_states[t]:
            raise ValueError(
                "The dimension of Markov_states is not compatible with \
                the dimension of transition_matrix!"
            )
        dim_Markov_states.append(shape[1])
    return dim_Markov_states, n_Markov_states

def check_Markovian_uncertainty(Markovian_uncertainty, T):
    """Check Markovian uncertainty is in the right form. Return
    the dimension of MC."""
    dim_Markov_states = []
    if not callable(Markovian_uncertainty):
        raise ValueError("Markovian uncertainty must be callable!")
    try:
        initial = Markovian_uncertainty(numpy.random, 2)
    except TypeError:
        raise TypeError("Sample path generator should always take "
            + "numpy.random.RandomState and size as its arguments!")
    if not isinstance(initial, numpy.ndarray) or initial.ndim != 3:
        raise ValueError("Sample path generator should always return a three "
            + "dimensional numpy array!")
    if initial.shape[1] != T:
        raise ValueError("Second dimension of sample path generator expectes "
            + "to be {} rather than {}!".format(T, initial.shape[1]))
    for t in range(T):
        dim_Markov_states.append(initial.shape[2])
    return dim_Markov_states

def allocate_jobs(n_forward_samples, n_processes):
    if n_forward_samples - n_processes == 1:
        return [[i] for i in range(n_processes-1)] + [range(n_processes-1,n_forward_samples)]
    chunk = (
        int(n_forward_samples / n_processes)
        if n_forward_samples % n_processes == 0
        else int(n_forward_samples / n_processes) + 1
    )
    division = list(range(0, n_forward_samples, chunk))
    division.append(n_forward_samples)
    return [range(division[p], division[p + 1]) for p in range(n_processes)]

def fit(array, convex=1):
    """Fit a smooth line to the given time-series data"""
    N = len(array)
    m = gurobipy.Model()
    fv = m.addVars(N)
    if convex == 1:
        m.addConstrs(fv[i] <= fv[i-1] for i in range(1,N))
        m.addConstrs(fv[i] + fv[i-2] >= 2*fv[i-1] for i in range(2,N))
    else:
        m.addConstrs(fv[i] >= fv[i-1] for i in range(1,N))
        m.addConstrs(fv[i] + fv[i-2] <= 2*fv[i-1] for i in range(2,N))
    m.setObjective(
        gurobipy.quicksum([fv[i] * fv[i] for i in range(N)])
        - 2 * gurobipy.LinExpr(array,fv.values())
    )
    m.Params.outputFlag = 0
    m.optimize()
    return [fv[i].X for i in range(N)]
