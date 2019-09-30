#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import numpy

def Expectation(obj, grad, p, sense):
    if p:
        return (numpy.dot(p, obj),numpy.dot(p, grad))
    else:
        return (numpy.mean(obj),numpy.mean(grad, axis=0))

def Expectation_AVaR(obj, grad, p, alpha_, lambda_, sense):
    n_samples, n_states = grad.shape
    if not p:
        p = numpy.ones(n_samples)/n_samples
    objAvg = numpy.dot(p, obj)
    gradAvg = numpy.dot(p, grad)
#    assert(type(gradAvg) == list and len(gradAvg) == len(p))
    objSortedIndex = numpy.argsort(obj)
    if sense == -1:
        objSortedIndex = objSortedIndex[::-1]
    ## store the index of 1-alpha percentile ##
    tempSum = 0
    for index in objSortedIndex:
        tempSum += p[index]
        if tempSum >= 1 - alpha_:
            kappa = index
            break
#    kappa = objSortedIndex[int((1 - alpha_) * sampleSize)]
    ## obj=(1-lambda)*objAvg+lambda(obj_kappa+1/alpha*avg((obj_kappa - obj_l)+))
    objLP = (1 - lambda_) * objAvg + lambda_ * obj[kappa]
    ## grad=(1-lambda)*gradAvg+lambda(grad_kappa+1/alpha*avg((pos))
    gradLP = (1 - lambda_) * gradAvg + lambda_ * grad[kappa]

    gradTerm = numpy.zeros((n_samples, n_states))
    objTerm = numpy.zeros(n_samples)
    for j in range(n_samples):
        if sense*(obj[j] - obj[kappa]) >= 0:
            gradTerm[j] = sense * (grad[j] - grad[kappa])
            objTerm[j] = sense * (obj[j] - obj[kappa])
    objLP += sense * lambda_ * numpy.dot(p, objTerm) / alpha_
    gradLP += sense * lambda_ * numpy.dot(p, gradTerm) / alpha_
    return (objLP, gradLP)
