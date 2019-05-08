#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
from msppy.sp import StochasticModel
from msppy.utils.exception import (SampleSizeError, DistributionError)
import numpy
import pytest

def univariate_generator(random_state):
    return random_state.randint(10)

def multivariate_generator_3(random_state):
    return random_state.multivariate_normal(
        mean = numpy.zeros(3),
        cov = numpy.eye(3)
    )

def univariate_generator_wrong(random_state):
    return 'a'

def multivariate_generator_2(random_state):
    return random_state.multivariate_normal(
        mean = numpy.zeros(2),
        cov = numpy.eye(2),
    )

class TestFiniteStochasticModel(object):

    m = StochasticModel()
    a = m.addVars(3)
    m.update()

    def test_independent_univariate_uncertainty(self):
        # uncertainty exception hierarchy:
        # TypeError->ValueError->SampleSizeError
        self.m.addVar(uncertainty = range(10))
        self.m.addConstr(
            self.a[0] == 0,
            uncertainty = {'rhs': range(10), self.a[1]: range(10)}
        )
        # test dimensionality error is catched
        with pytest.raises(SampleSizeError):
            self.m.addStateVar(uncertainty = range(11))
        with pytest.raises(SampleSizeError):
            self.m.addVar(uncertainty = univariate_generator)
        with pytest.raises(SampleSizeError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs': range(11)}
            )
        with pytest.raises(SampleSizeError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {self.a[1]: univariate_generator}
            )
        # test non-numeric scenarios
        with pytest.raises(ValueError):
            self.m.addStateVar(uncertainty = ['a'])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs': ['a']}
            )
        # test uncertainty having different dimension with object
        with pytest.raises(ValueError):
            self.m.addVar(uncertainty = [range(10)])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs': [range(10)]*2}
            )

    def test_independent_multivariate_uncertainty(self):
        # test uncertain objective coefficient
        self.m.addVars(3, uncertainty = [range(3)]*10)
        # test constraint coefficient
        self.m.addConstrs(
            (self.a[i] == 0 for i in range(3)),
            uncertainty = [range(3)]*10
        )
        # test dimensionality error is catched
        with pytest.raises(SampleSizeError):
            self.m.addStateVars(3, uncertainty = [range(3)]*11)
        with pytest.raises(SampleSizeError):
            self.m.addVars(3, uncertainty = multivariate_generator_3)
        with pytest.raises(SampleSizeError):
            self.m.addConstrs(
                (self.a[i] == 0 for i in range(3)),
                uncertainty = [range(3)]*11
            )
        with pytest.raises(SampleSizeError):
            self.m.addConstrs(
                (self.a[i] == 0 for i in range(3)),
                uncertainty = multivariate_generator_3
            )
        # test non-numeric scenarios
        with pytest.raises(ValueError):
            self.m.addStateVars(2, uncertainty = ['a','b'])
        with pytest.raises(ValueError):
            self.m.addConstrs(
                (self.a[i] == 0 for i in range(2)),
                uncertainty = [[1,2],['a','b']]
            )
        # test uncertainty having different dimension with object
        with pytest.raises(ValueError):
            self.m.addVars(3, uncertainty = [range(2),range(3)])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs': [range(2),range(3)]}
            )

    def test_dependent_univariate_uncertainty(self):
        # test uncertain objective coefficient
        self.m.addVar(uncertainty_dependent = 0)
        # test constraint coefficient
        self.m.addConstr(
            self.a[0] == 0,
            uncertainty_dependent = {'rhs': 1, self.a[1]: 2}
        )
        # test invalid scenarios
        with pytest.raises(ValueError):
            self.m.addStateVar(uncertainty_dependent = [1,2])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty_dependent = {'rhs': [1,2,3]}
            )
        with pytest.raises(ValueError):
            self.m.addStateVar(uncertainty_dependent = [1,2])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty_dependent = {'rhs': [1,2,3]}
            )

    def test_dependent_multivariate_uncertainty(self):
        # test uncertain objective coefficient
        self.m.addVars(3, uncertainty_dependent = [4,5,6])
        # test constraint coefficient
        self.m.addConstrs(
            (self.a[i] == 0 for i in range(3)),
            uncertainty_dependent = [7,8,9]
        )
        # test invalid scenarios
        with pytest.raises(ValueError):
            self.m.addStateVars(3, uncertainty_dependent = [1,2])
        with pytest.raises(ValueError):
            self.m.addConstrs(
                (self.a[i] == 0 for i in range(3)),
                uncertainty_dependent = 1
            )
        with pytest.raises(ValueError):
            self.m.addStateVars(3, uncertainty_dependent = 1)
        with pytest.raises(ValueError):
            self.m.addConstrs(
                (self.a[i] == 0 for i in range(3)),
                uncertainty_dependent = [1,2,3,4]
            )
        with pytest.raises(TypeError):
            self.m.addConstrs(
                (self.a[i] == 0 for i in range(3)),
                uncertainty_dependent = {'rhs': 1}
            )

    def test_update_uncertainty(self):
        for j in range(self.m.n_samples):
            self.m._update_uncertainty(j)

    def test_set_probability(self):
        self.m.set_probability([
        1/self.m.n_samples for _ in range(self.m.n_samples)
        ])
        with pytest.raises(ValueError):
            self.m.set_probability([
                1/self.m.n_samples for _ in range(self.m.n_samples+1)
            ])

    def test_update_uncertainty_dependent(self):
        self.m._update_uncertainty_dependent(range(10))

    def test_controls(self):
        assert(len(self.m.controls)
            == self.m.numVars - len(self.m.local_copies) - len(self.m.states)
        )

class TestContinuousStochasticModel():

    m = StochasticModel()
    a = m.addVars(3)
    m.update()

    def test_independent_univariate_uncertainty(self):
        # test uncertain objective coefficient
        self.m.addVar(uncertainty = univariate_generator)
        # test constraint coefficient
        self.m.addConstr(
            self.a[0] == 0,
            uncertainty = {'rhs': univariate_generator}
        )
        # test dimensionality error is catched
        with pytest.raises(SampleSizeError):
            self.m.addStateVar(uncertainty = [1,2,3])
        with pytest.raises(SampleSizeError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs':[1,2,3]}
            )
        # test non-numeric scenarios
        with pytest.raises(ValueError):
            self.m.addStateVar(uncertainty = ['a'])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs':['a']}
            )
        # test invalid distribution
        with pytest.raises(DistributionError):
            self.m.addVar(uncertainty = univariate_generator_wrong)
        with pytest.raises(DistributionError):
            self.m.addVar(uncertainty = multivariate_generator_3)
        with pytest.raises(DistributionError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs': multivariate_generator_2}
            )

    def test_dependent_univariate_uncertainty(self):
        # test uncertain objective coefficient
        self.m.addVar(uncertainty = univariate_generator)
        # test constraint coefficient
        self.m.addConstr(
            self.a[0] == 0,
            uncertainty = {'rhs': univariate_generator}
        )
        # test dimensionality error is catched
        with pytest.raises(SampleSizeError):
            self.m.addStateVar(uncertainty = [1,2,3])
        with pytest.raises(SampleSizeError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs':[1,2,3]}
            )
        # test non-numeric scenarios
        with pytest.raises(ValueError):
            self.m.addStateVar(uncertainty = ['a'])
        with pytest.raises(ValueError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs':['a']}
            )
        # test invalid distribution
        with pytest.raises(DistributionError):
            self.m.addVar(uncertainty = univariate_generator_wrong)
        with pytest.raises(DistributionError):
            self.m.addVar(uncertainty = multivariate_generator_3)
        with pytest.raises(DistributionError):
            self.m.addConstr(
                self.a[0] == 0,
                uncertainty = {'rhs': multivariate_generator_2}
            )

    def test_sample_uncertainty(self):
        self.m._sample_uncertainty(random_state=7)
