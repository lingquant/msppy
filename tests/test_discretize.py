#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import numpy
import pytest
from msppy.discretize import Markovian
T = 4
n_Markov_states=[1,3,4,3]
def f(random_state, size):
    a = numpy.empty([size,T,1])
    a[:,0,:] = 0.2
    for t in range(1,T):
        noise = random_state.normal(size=size)
        a[:,t,:] = 0.5*a[:,t-1,:] + noise[:,numpy.newaxis]
    return a
def g(random_state, size):
    a = numpy.empty([size,T,2])
    a[:,0,:] = [0.2,0.3]
    for t in range(1,T):
        noise = random_state.multivariate_normal(
            mean=[0,0],
            cov=[[1,0],
                 [0,1]],
            size=size,
        )
        a[:,t,:] = 0.5 * a[:,t-1,:] + noise
    return a

def test_RSA():
    m = Markovian(f,n_Markov_states,5)
    n = Markovian(g,n_Markov_states,5)
    m.RSA()
    n.RSA()

def test_SA():
    m = Markovian(f,n_Markov_states,5)
    n = Markovian(g,n_Markov_states,5)
    m.SA()
    n.SA()
#
def test_SAA():
    m = Markovian(f,n_Markov_states,5)
    n = Markovian(g,n_Markov_states,5)
    m.SAA()
    n.SAA()
