#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
from msppy.msp import MSIP
from msppy.solver import SDDiP
from msppy.evaluation import EvaluationTrue
import numpy
import gurobipy
numpy.random.seed(2)

T = 5
numSeats = 5
numScen = 5
offer = numpy.random.randint(numSeats, size=(T, numScen, numSeats))
price = [[3+t for _ in range(20)] for t in range(T)]
selling = MSIP(T=T, sense=-1, bound=100)
for t in range(T):
    m = selling.models[t]
    now, past = m.addStateVars(numSeats, vtype='B', name="seatCondition")
    if t == 0:
        m.addConstrs(past[i] == 1 for i in range(numSeats))
    else:
        accept_offer = m.addVars(numSeats, vtype='B', obj=price[t], name="acceptOffer")
        m.addConstrs(
            (accept_offer[i] <= 0 for i in range(numSeats) ),
            uncertainty = offer[t],
        )
        m.addConstrs(now[i] == past[i] - accept_offer[i] for i in range(numSeats))
SDDiP(selling).solve(max_iterations=20, cuts=['LG'])
resultTrue = EvaluationTrue(selling)
resultTrue.run(n_simulations=100)
