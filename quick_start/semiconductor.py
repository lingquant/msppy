"""
Created on Wed Nov 09 10:43:35 2018

@author: lingquan

    Special thanks to Ian to prepared the data and modeling
    The formulation is from Shabbir Ahmed, Semiconductor Tool Planning Via Multistage Stochastic programming
    A wafer fab consisting of M tool types that process N types of wafers. Each product goes through K processing steps, each of which can be performed on one or more tool types.
    I: tool type
    J: wafer type
    K: processing step
    x_it: number of tool type i purchased in period t
    u_jt: the shortage of wafer type j in period t
    v_jkt: the allocation of processing step k of wafer type j to tool type i in period t
    w_jt: the production of wafer type j in period t
    a_i,j,k: time requied (in hrs) by processing step k on wafer type j on tool type i
    alpha_it: cost of tool type i in period t
    beta_jt: penalty cost of unit shortage in wafer type j in period t
    c_i: the capacity of tool type i
    d_jt: demand of wafer type j in period t
    X_it: accumulated number of tool type i purchased till period t
"""

import numpy
from msppy.solver import SDDP
from msppy.msp import MSLP
from msppy.evaluation import EvaluationTrue
import gurobipy

# INPUT
T = 4
I = 10
J = 10
K = 2

alpha_0 =  [686, 784, 540, 641, 1073, 1388, 1727, 1469, 586, 515]
beta_0 = [174, 115, 92, 116, 93, 164, 190, 174, 190, 200]
c = [7,17,11,16,18,7,7,9,8,14]
d_0 = [607,943,732,1279,434,378,1964,430, 410, 525]

a = numpy.random.randint(5,10,size = [I, J, K])

d_perturb =  [0.0422902245,
               0.0549456137,
               0.0868569685,
               0.0950609064,
               0.0538731273,
               0.0917075818,
               0.0673065114,
               0.0594680277,
               0.0544299191,
               0.0782010312]

beta_perturb =  [0.0129739644,
                0.063853852,
                0.0925580104,
                0.0766634092,
                0.0953244752,
                0.0563760149,
                0.075759652,
                0.0583249427,
                0.0324810132,
                0.0694020021]

alpha_perturb =  [0.0638533975,
                 0.068050401,
                 0.0747693903,
                 0.0514849591,
                 0.0323470258,
                 0.0480910211,
                 0.0304004586,
                 0.0976094813,
                 0.0694752024,
                 0.0703992735,
                 0.0775236862]

# d_jt #
def d_generator(random_state):
    return [round(numpy.dot(random_state.normal(1, d_perturb[j]), d_0[j])) for j in range(J)]
# alpha_it #
def alpha_generator(random_state):
    return [round(numpy.dot(random_state.normal(1, alpha_perturb[i]), alpha_0[i])) for i in range(I)]
# beta_jt #
def beta_generator(random_state):
    return [round(numpy.dot(random_state.normal(1, beta_perturb[j]), beta_0[j])) for j in range(J)]
semiconductor = MSLP(T=T, bound=0)
for t in range(T):
    m = semiconductor[t]
    # X_it #
    X_now, X_past = m.addStateVars(I, name='accumulated purchase')
    if t == 0:
        m.addConstrs(X_now[j] == 0 for j in range(J))
    else:
        # u_jt #
        u = m.addVars(J, name='shortage', uncertainty=beta_generator)
        # v_ijkt #
        v = m.addVars(I,J,K, name='allocation')
        # w_jt #
        w = m.addVars(J, name='production')
        # x_it #
        x = m.addVars(I, name='purchase', uncertainty=alpha_generator)
        ## accumulated number of purchased tools updated ##
        m.addConstrs(X_now[i] == X_past[i] + x[i] for i in range(I))
        # time allocation constraint
        m.addConstrs(gurobipy.quicksum( gurobipy.quicksum( a[i][j][k] * v[(i,j,k)] for k in range(K) ) for j in range(J)) <= c[i] * X_now[i] for i in range(I))
        # production allocation constraint
        m.addConstrs(gurobipy.quicksum( v[(i,j,k)] for i in range(I) ) >= w[j] for j in range(J) for k in range(K) )
        # demand constraint
        m.addConstrs( (w[j] + u[j] >= 0 for j in range(J)), uncertainty=d_generator )
semiconductor.discretize(n_samples=20)
SDDP(semiconductor).solve(max_iterations=10)
result = EvaluationTrue(semiconductor)
result.run(n_simulations=1000)
