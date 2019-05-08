#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import pandas
import numpy
from msppy.msp import MSLP
from msppy.solver import SDDP
import gurobipy
import sys
n_processes = int(sys.argv[1])
hydro_ = pandas.read_csv("./data/hydro.csv", index_col=0)
demand = pandas.read_csv("./data/demand.csv", index_col=0)
deficit_ = pandas.read_csv("./data/deficit.csv", index_col=0)
exchange_ub = pandas.read_csv("./data/exchange.csv", index_col=0)
thermal_ = [
    pandas.read_csv("./data/thermal_{}.csv".format(i), index_col=0)
    for i in range(4)
]
hist = [
    pandas.read_csv("./data/hist_{}.csv".format(i), sep=";")
    for i in range(4)
]
hist = pandas.concat(hist, axis=1)
hist.dropna(inplace=True)
hist.drop(columns='YEAR', inplace=True)
scenarios = [
    hist.iloc[:,12*i:12*(i+1)].transpose().values for i in range(4)
]
T = 120
HydroThermal = MSLP(T=T, bound=0)
for t in range(T):
    m = HydroThermal[t]
    stored_now,stored_past = m.addStateVars(4, ub=hydro_['UB'][:4], name="stored")
    spill = m.addVars(4, name="spill")
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
    exchange = m.addVars(5,5, ub=exchange_ub.values.flatten(), name="exchange")
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
    for i in range(4):
        if t == 0:
            m.addConstr(
                stored_now[i] + spill[i] + hydro[i] - stored_past[i]
                == hydro_['INITIAL'][4:8][i]
            )
        else:
            m.addConstr(
                stored_now[i] + spill[i] + hydro[i] - stored_past[i] == 0,
                uncertainty = {'rhs': scenarios[i][(t-1)%12]}
            )
    if t == 0:
        m.addConstrs(stored_past[i] == hydro_['INITIAL'][:4][i] for i in range(4))
HydroThermal_SDDP = SDDP(HydroThermal)
HydroThermal_SDDP.solve(
    max_time=200,n_processes=n_processes,n_steps=n_processes)
