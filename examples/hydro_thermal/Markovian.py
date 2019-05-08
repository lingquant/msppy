#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import pandas
import numpy
import gurobipy
from msppy.msp import MSLP
from msppy.solver import SDDP
from msppy.evaluation import Evaluation,EvaluationTrue
import time
import sys

method = sys.argv[1]
n_Markov_states = int(sys.argv[2])
n_sample_paths = int(sys.argv[3])
n_iterations = int(sys.argv[4])

gamma = numpy.array(pandas.read_csv(
    "./data/gamma.csv",
    names=[0,1,2,3],
    index_col=0,
    skiprows=1,
))
sigma = [
    numpy.array(pandas.read_csv(
        "./data/sigma_{}.csv".format(i),
        names=[0,1,2,3],
        index_col=0,
        skiprows=1,
    )) for i in range(12)
]
exp_mu = numpy.array(pandas.read_csv(
    "./data/exp_mu.csv",
    names=[0,1,2,3],
    index_col=0,
    skiprows=1,
))
hydro_ = pandas.read_csv("./data/hydro.csv", index_col=0)
demand = pandas.read_csv("./data/demand.csv", index_col=0)
deficit_ = pandas.read_csv("./data/deficit.csv", index_col=0)
exchange_ub = pandas.read_csv("./data/exchange.csv", index_col=0)
exchange_cost = pandas.read_csv("./data/exchange_cost.csv", index_col=0)
thermal_ = [pandas.read_csv("./data/thermal_{}.csv".format(i), index_col=0) for i in range(4)]
hist = [pandas.read_csv("./data/hist_{}.csv".format(i), index_col=0, sep=';') for i in range(4)]
stored_initial = numpy.array(hydro_['INITIAL'][:4])
inflow_initial = numpy.array(hydro_['INITIAL'][4:8])

T = 120
def generator(random_state,size):
    inflow = numpy.empty([size,T,4])
    inflow[:,0,:] = inflow_initial[numpy.newaxis:,]
    for t in range(T-1):
        noise = numpy.exp(random_state.multivariate_normal(mean=[0]*4, cov=sigma[t%12],size=size))
        inflow[:,t+1,:] = noise * (
            (1-gamma[t%12]) * exp_mu[t%12]
            + gamma[t%12] * exp_mu[t%12]/exp_mu[(t-1)%12] * inflow[:,t,:]
        )
    return inflow

HydroThermal = MSLP(T=T, bound=0, discount=0.9906)
HydroThermal.add_Markovian_uncertainty(generator)
for t in range(T):
    m = HydroThermal[t]
    stored_now, stored_past = m.addStateVars(4, ub=hydro_['UB'][:4], name="stored")
    spill = m.addVars(4, obj=0.001, name="spill")
    hydro = m.addVars(4, ub=hydro_['UB'][-4:], name="hydro")
    deficit = m.addVars(
        [(i,j) for i in range(4) for j in range(4)],
        ub = [
            demand.iloc[t%12][i] * deficit_['DEPTH'][j]
            for i in range(4) for j in range(4)
        ],
        obj = [
            deficit_['OBJ'][j]
            for i in range(4) for j in range(4)
        ],
        name = "deficit")
    thermal = [None] * 4
    for i in range(4):
        thermal[i] = m.addVars(
            len(thermal_[i]),
            ub=thermal_[i]['UB'],
            lb=thermal_[i]['LB'],
            obj=thermal_[i]['OBJ'],
            name="thermal_{}".format(i)
        )
    exchange = m.addVars(5,5, obj=exchange_cost.values.flatten(),
        ub=exchange_ub.values.flatten(), name="exchange")
    thermal_sum = m.addVars(4, name="thermal_sum")
    m.addConstrs(
        thermal_sum[i] == gurobipy.quicksum(thermal[i].values()) for i in range(4)
    )
    for i in range(4):
        m.addConstr(
            thermal_sum[i]
            + gurobipy.quicksum(deficit[(i,j)] for j in range(4))
            + hydro[i]
            - gurobipy.quicksum(exchange[(i,j)] for j in range(5))
            + gurobipy.quicksum(exchange[(j,i)] for j in range(5))
            == demand.iloc[t%12][i]
        )
    m.addConstr(
        gurobipy.quicksum(exchange[(j,4)] for j in range(5))
        - gurobipy.quicksum(exchange[(4,j)] for j in range(5))
        == 0
    )
    m.addConstrs(
        (stored_now[i] + spill[i] + hydro[i] - stored_past[i] == 0 for i in range(4)),
        uncertainty_dependent=[0,1,2,3],
    )
    if t == 0:
        m.addConstrs(stored_past[i] == stored_initial[i] for i in range(4))
start = time.time()
HydroThermal.discretize(
    method=method,
    n_Markov_states=n_Markov_states,
    n_sample_paths=n_sample_paths,
)
time_MCA = time.time() - start
HT_sddp = SDDP(HydroThermal)
HT_sddp.solve(
    n_processes=6,
    n_steps=6,
    max_iterations=n_iterations,
)
result = Evaluation(HydroThermal)
result.run(random_state=666, n_simulations=3000)
resultTrue = EvaluationTrue(HydroThermal)
resultTrue.run(random_state=666, n_simulations=3000)
model = method + str(n_Markov_states)
if model == 'SA100':
    fig = HT_sddp.plot_bounds(window=1,smooth=1)
    fig.tight_layout()
    fig.savefig("./result/{}_bounds.png".format(model), dpi=1200)
pandas.DataFrame({'pv':result.pv}).to_csv("./result/{}_result.csv".format(model))
pandas.DataFrame({'pv':resultTrue.pv}).to_csv("./result/{}_resultTrue.csv".format(model))
print('model',model)
print('iter_MCA',n_sample_paths)
print('time_MCA',time_MCA)
print('iter_SDDP',n_iterations)
print('time_SDDP',HT_sddp.total_time)
print('time_total',time_MCA+HT_sddp.total_time)
print('gap',result.gap)
print('bound',result.db)
print('CI for approx. model',result.CI)
print('CI for true problem', resultTrue.CI)
