#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:32:42 2018

News vendor problem. Minimize cost.
risk neutral optimum is -97.9
risk averse optimum (alpha_ = 0.6, lambda_ = 0.5) is -91.6

Uncertainties on the right hand side (demand), markovian purchase price
stage-wise independent discrete uncertainties & stage-wise dependent discrete uncertainties
initial inventory is 5, full capacity is 100.
selling amount is restricted between half of demand and full demand

@author: lingquan
"""
from msppy.msp import MSLP
from msppy.solver import Extensive,SDDP
import gurobipy
T = 4
PurchasePrice = [5.0, 8.0]
Demand = [[10.0, 15.0], [12.0, 20.0], [8.0, 20.0]]
RetailPrice = 7.0

newsVendor = MSLP(T=T, sense=1, bound=-1000)
newsVendor.add_MC_uncertainty(
    Markov_states = [
        [[5.0]],
        [[5.0]],
        [[5.0],[8.0]],
        [[5.0],[8.0]]
    ],
    transition_matrix = [
        [[1]],
        [[1]],
        [[0.6,0.4]],
        [[0.3,0.7],[0.3,0.7]]
    ]
)
for t in range(T):
    m = newsVendor[t]
    now, past = m.addStateVar(ub=100, name="stock")
    if t > 0:
        buy = m.addVar(name="buy", uncertainty_dependent=0)
        sell = m.addVar(name="sell", obj=- RetailPrice)
        m.addConstr(now == past + buy - sell)
        random = m.addVar(lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, name="demand")
        m.addConstr(random == 20, uncertainty={'rhs': Demand[t-1]})
        m.addConstr(sell <= random)
        m.addConstr(sell >= 0.5 * random)
    if t == 0:
        m.addConstr(now == 5.0)
Extensive(newsVendor).solve()
newsVendor.set_AVaR(a=0.6, l=0.5)
Extensive(newsVendor).solve()
SDDP(newsVendor).solve(max_iterations=100)
