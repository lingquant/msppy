#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
'''
The data process is stage-wise independent and on the RHS.
It was originally from http://www.optimization-online.org/DB_FILE/2017/12/6388.pdf.
Verified optimal value by extensive solver is 68200.
SDDP solver also obtains the same optimal value.
d = (
    100 w.p. 0.4,
    300 w.p. 0.6
)
The first stage:
min x + 3w + 0.5y
     x <= 2
     x + w - y = 1

The second stage:
min x + 3w + 0.5y
     x <= 2
     x + y_past + w - y_now = d

The third stage:

min x + 3w + 0.5y
     x <= 2
     x + y_past + w - y_now = d

'''
from msppy.msp import MSIP
from msppy.solver import SDDiP,Extensive
import gurobipy
T = 3
A = [100,300]
airConditioners = MSIP(T=T, bound=0)
for t in range(T):
    m = airConditioners[t]
    storage_now, storage_past = m.addStateVar(vtype='I', obj=50)
    produce = m.addVar(ub=200, vtype='I', obj=100)
    overtime = m.addVar(vtype='I', obj=300)
    m.update()
    if t == 0:
        m.addConstr(produce + overtime - storage_now == 100)
    else:
        m.addConstr(
            storage_past + produce + overtime - storage_now == 0,
            uncertainty={'rhs': A}
        )
        m.set_probability([0.4,0.6])
Extensive(airConditioners).solve()
SDDiP(airConditioners).solve(cuts=['B'], max_iterations=10)
