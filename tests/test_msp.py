#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
from msppy.msp import MSLP
import pytest
from msppy.utils.exception import MarkovianDimensionError
import numpy
# MC uncertainty
Markov_states = [
    [[6000]],
    [[5000],[7000]],
    [[6000],[8000]],
]
transition_matrix = [
    [[1]],
    [[0.4,0.6]],
    [[0.2,0.8],[0.3,0.7]]
]
Invalid_Markov_states_1 = [
    [[6000],[8000]],
    [[5000],[7000]],
    [[6000],[8000]],
]
invalid_transition_matrix = [None]*3
invalid_transition_matrix[0] = [
    [[0.4,0.6]],
    [[0.4,0.6]],
    [[0.2,0.8],[0.3,0.7]]
]
invalid_transition_matrix[1] = [
    [[1]],
    [[0.3,0.6]],
    [[0.2,0.8],[0.3,0.7]]
]
invalid_transition_matrix[2] = [
    [[1]],
    [[0.3,0.6]],
    [[0.2,0.8],[0.3,0.7],[0.4,0.6]]
]
def sample_path_generator(random_state,size):
    a = numpy.zeros([size,3,1])
    a[:,0,:] = 0.2
    for t in range(1,3):
        a[:,t,:] = 0.5*a[:,t-1,:] + random_state.lognormal(0,1,size=[size,1])
    return a

def sample_path_generator_2(random_state,size):
    a = numpy.zeros([size,3,2])
    a[:,0,:] = [0.2,0.3]
    for t in range(1,3):
        a[:,t,:] = 0.5*a[:,t-1,:] + random_state.multivariate_normal(
            mean=[0,0],cov=numpy.eye(2),size=size)
    return a

def invalid_sample_path_generator_1(random_state,size):
    a = numpy.zeros([size,3])
    for t in range(1,3):
        a[:,t] = 0.5*a[:,t-1] + random_state.lognormal(0,1,size=size)
    return a

def invalid_sample_path_generator_2(size):
    a = numpy.zeros([size,3,1])
    for t in range(1,3):
        a[:,t,:] = 0.5*a[:,t-1,:] + numpy.random.lognormal(0,1,size=size)
    return a
# stage-wise independent uncertainty
samples_1 = [2,3,4]
samples_2 = [5,6,7,8]

def sample_generator_1(random_state):
    return random_state.normal()

def sample_generator_2(random_state):
    return random_state.lognormal()

class TestMCMSLP(object):
    MSP = MSLP(T=3)

    def test_MC_uncertainty(self):
        with pytest.raises(ValueError):
            self.MSP.add_MC_uncertainty(
                Markov_states=Invalid_Markov_states_1,
                transition_matrix=transition_matrix,
            )
        for i in range(3):
            with pytest.raises(ValueError):
                self.MSP.add_MC_uncertainty(
                    Markov_states=Markov_states,
                    transition_matrix=invalid_transition_matrix[i],
                )
        self.MSP.add_MC_uncertainty(
            Markov_states=Markov_states,
            transition_matrix=transition_matrix,
        )
        with pytest.raises(ValueError):
            self.MSP.add_Markovian_uncertainty(sample_path_generator)

        self.MSP[0].addVar(uncertainty_dependent=0)
        a = self.MSP[1].addVars(2)
        self.MSP[1].addConstr(
            a[0] + a[1] == 10,
            uncertainty = {a[0]:[2,3,4]},
            uncertainty_dependent = {'rhs':0}
        )
        a = self.MSP[2].addVars(2)
        self.MSP[2].addConstr(
            a[0] + a[1] == 10,
            uncertainty = {a[1]:[4,5,6,7]},
            uncertainty_dependent = {a[0]:0}
        )
        self.MSP._check_inidividual_Markovian_index()
        # state variables must be added
        with pytest.raises(Exception):
            self.MSP._check_individual_stage_models()
        for t in range(3):
            self.MSP[t].addStateVar()
        self.MSP._check_individual_stage_models()
        self.MSP._check_multistage_model()
        self.MSP[1][1].addVar(uncertainty_dependent=1)
        with pytest.raises(MarkovianDimensionError):
            self.MSP._check_inidividual_Markovian_index()

class TestMarkovianMSLP(object):
    MSP = MSLP(T=3)

    def test_Markovian_uncertainty(self):
        with pytest.raises(ValueError):
            self.MSP.add_Markovian_uncertainty(invalid_sample_path_generator_1)
        with pytest.raises(TypeError):
            self.MSP.add_Markovian_uncertainty(invalid_sample_path_generator_2)
        self.MSP.add_Markovian_uncertainty(sample_path_generator)
        self.MSP[0].addVar(uncertainty_dependent=0)
        a = self.MSP[1].addVars(2)
        self.MSP[1].addConstr(
            a[0] + a[1] == 10,
            uncertainty = {a[0]:sample_generator_1},
            uncertainty_dependent = {'rhs':0}
        )
        a = self.MSP[2].addVars(2)
        self.MSP[2].addConstr(
            a[0] + a[1] == 10,
            uncertainty = {a[1]:sample_generator_2},
            uncertainty_dependent = {a[0]:0}
        )
        for t in range(3):
            self.MSP[t].addStateVar()
        with pytest.raises(Exception):
            self.MSP._check_multistage_model()
        with pytest.raises(Exception):
            self.MSP._check_individual_stage_models()
        self.MSP.discretize(n_Markov_states=2,n_sample_paths=5)
        with pytest.raises(Exception):
            self.MSP._check_individual_stage_models()
        self.MSP.discretize(n_samples=3)
        self.MSP._check_individual_stage_models()
        self.MSP._check_multistage_model()
