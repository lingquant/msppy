#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
Minimize the absolute value
A stage-wise independent finite discrete MSLP.
The data process is stage-wise independent and on the RHS.
This is a problem originally from Bernado.
"""
from msppy.msp import MSIP
from msppy.solver import Extensive, SDDiP
import numpy
numpy.random.seed(3)
T = 4
precision = 1
rhs = [numpy.random.normal(-1,1,size=10) for _ in range(T-1)]
absolute = MSIP(T=T, bound=0)
for t in range(T):
    m = absolute[t]
    x_now, x_past = m.addStateVar(lb=-100, ub=100, name='x')
    if t == 0:
        m.addConstr(x_now == 0)
    else:
        y = m.addVar(name='y', obj=1)
        c = m.addVar(vtype='B', name='c')
        slack = m.addVar(ub=1/10**(precision), name='slack')
        m.addConstr(y >= -x_now)
        m.addConstr(y >= x_now)
        m.addConstr(
            x_now - x_past - 2*c  + slack == -1,
            uncertainty={'rhs': rhs[t-1]}
        )
absolute.binarize(bin_stage=T, precision=precision)
Extensive(absolute).solve(outputFlag=0)
absolute_sddip = SDDiP(absolute)
absolute_sddip.solve(cuts=['LG'], n_steps=3, n_processes=3, max_iterations=30)

#absolute.extensive(outputflag=0)
