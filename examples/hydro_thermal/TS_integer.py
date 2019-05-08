#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import pandas
import numpy
import gurobipy
from msppy.msp import MSIP
from msppy.solver import Extensive
from msppy.solver import SDDiP
from msppy.evaluation import Evaluation,EvaluationTrue
import sys

T = int(sys.argv[1])
solver = sys.argv[2]
if solver == 'SDDiP':
    cut = sys.argv[3]
    n_iterations = int(sys.argv[4])
else:
    if sys.argv[3] == 'None':
        time_limit = float('inf')
    else:
        time_limit = float(sys.argv[3])

seed = 888

gamma = pandas.read_csv(
    "./data/gamma.csv",
    names=[0,1,2,3],
    index_col=0,
    skiprows=1,
)
sigma = [
    pandas.read_csv(
        "./data/sigma_{}.csv".format(i),
        names=[0,1,2,3],
        index_col=0,
        skiprows=1,
    ) for i in range(12)
]
exp_mu = pandas.read_csv(
    "./data/exp_mu.csv",
    names=[0,1,2,3],
    index_col=0,
    skiprows=1,
)
hydro_ = pandas.read_csv("./data/hydro.csv", index_col=0)
demand = pandas.read_csv("./data/demand.csv", index_col=0)
deficit_ = pandas.read_csv("./data/deficit.csv", index_col=0)
exchange_ub = pandas.read_csv("./data/exchange.csv", index_col=0)
exchange_cost = pandas.read_csv("./data/exchange_cost.csv", index_col=0)
thermal_ = [pandas.read_csv("./data/thermal_{}.csv".format(i), index_col=0) for i in range(4)]
stored_initial = hydro_['INITIAL'][:4]
inflow_initial = hydro_['INITIAL'][4:8]
thermal_mid_level = [(thermal_[i].sum()['UB'] + thermal_[i].sum()['LB'])/2 for i in range(4)]

def sampler(t):
    def inner(random_state):
        noise = numpy.exp(random_state.multivariate_normal(mean=[0]*4, cov=sigma[t%12]))
        coef = [None]*4
        rhs = [None]*4
        for i in range(4):
            coef[i] = -noise[i]*gamma.iloc[t%12][i]*exp_mu[i][t%12]/exp_mu[i][(t-1)%12]
            rhs[i] = noise[i]*(1-gamma.iloc[t%12][i])*exp_mu[i][t%12]
        return (coef+rhs)
    return inner

HydroThermal = MSIP(T=T, bound=0, discount=0.9906)
for t in range(T):
    m = HydroThermal[t]
    stored_now, stored_past = m.addStateVars(4, ub=hydro_['UB'][:4], name="stored")
    inflow_now, inflow_past = m.addStateVars(4, name="inflow")
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
    m.addConstrs(thermal_sum[i] == gurobipy.quicksum(thermal[i].values()) for i in range(4))
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
        stored_now[i] + spill[i] + hydro[i] - stored_past[i] == inflow_now[i]
        for i in range(4)
    )
    if t == 0:
        m.addConstrs(stored_past[i] == stored_initial[i] for i in range(4))
        m.addConstrs(inflow_now[i] == inflow_initial[i] for i in range(4))
    else:
        TS = m.addConstrs(inflow_now[i] + inflow_past[i] == 0 for i in range(4))
        m.add_continuous_uncertainty(
            uncertainty=sampler(t-1),
            locations=(
                [(TS[i],inflow_past[i]) for i in range(4)]
                + [TS[i] for i in range(4)]
            ),
        )
    if t < T/2:
        z = m.addVars(4, vtype='B')
        m.addConstrs(hydro[i] >= z[i] * hydro_['UB'][:4][i]/2 for i in range(4))
        m.addConstrs(thermal_sum[i] >= (1-z[i]) * thermal_mid_level[i] for i in range(4))

HydroThermal.discretize(n_samples=100, random_state=seed)
if solver == 'Extensive':
    HT_extensive = Extensive(HydroThermal)
    HT_extensive.solve(outputFlag=0,TimeLimit=time_limit)
    print('lower bound',HT_extensive.objBound)
    print('upper bound',HT_extensive.objVal)
    print('total time',HT_extensive.total_time)
    print('gap',HT_extensive.MIPGap)
if solver == 'SDDiP':
    HT_sddp = SDDiP(HydroThermal)
    HT_sddp.solve(
       logFile=0,
       logToConsole=0,
       cuts = [cut],
       n_processes=1,
       n_steps=1,
       max_iterations=n_iterations,
    )
    result = Evaluation(HydroThermal)
    if T in [2,3]:
        result.run(random_state=666, n_simulations=-1)
    else:
        result.run(random_state=666, n_simulations=3000)
    resultTrue = EvaluationTrue(HydroThermal)
    resultTrue.run(random_state=666, n_simulations=3000)
    print('lower bound',result.db)
    if T in [2,3]:
        print('upper bound',result.epv)
    else:
        print('CI appr.',result.CI)
    print('gap',result.gap)
    print('CI true',resultTrue.CI)
