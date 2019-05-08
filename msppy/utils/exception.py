#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
class SampleSizeError(Exception):
    """Exception class to raise if uncertainty of different sample sizes are
    added to the model."""
    def __init__(self, modelName, dimensionality, uncertainty, dimension):
        Exception.__init__(
            self,
            "Dimensionality of stochasticModel {} is {}\
            but dimension of the uncertainty {} is {}".format(
                modelName, dimensionality, uncertainty, dimension
            ),
        )


class DistributionError(Exception):
    """Exception class to raise if continuous distribution is not added
    properly"""
    def __init__(self, arg=True, ret=True):
        if arg == False:
            Exception.__init__(
                self,
                "Continuous distribution should always take \
                numpy.random.RandomState as its single argument.",
            )
        if ret == False:
            Exception.__init__(
                self,
                "Univariate distribution should always return a number; \
                Multivariate distribution should always return an array-like.",
            )

class MarkovianDimensionError(Exception):
    """Exception class to raise if dim index is not specified
    properly"""
    def __init__(self):
        Exception.__init__(
            self,
            "Dimension indices of Markovian uncertainties not set properly."
        )
