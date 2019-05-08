#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:36:18 2019

@author: lingquan
"""

import numpy
from msppy.utils.plot import fan_plot
import matplotlib.pyplot as plt
import pandas
from msppy.discretize import Markovian
from scipy import stats
from msppy.msp import MSLP
from msppy.solver import SDDP
from msppy.evaluation import Evaluation, EvaluationTrue
import gurobipy

T = 12
rf = 0.0005
fee = 0.001
params = pandas.read_csv("./data/parameters.csv",index_col=0)
coeffs = pandas.read_csv("./data/coefficients.csv",index_col=0)
mu,phi,omega,alpha_1,alpha_2 = params.iloc[:,0]
alpha = numpy.array(coeffs['alpha'])
beta = numpy.array(coeffs['beta'])
sigma = numpy.array(coeffs['epsilon'])
# stage-wise independent generator
def f(alpha,sigma):
    def inner(random_state):
        return random_state.normal(alpha+1,sigma)
    return inner
# Markovian process generator
def generator(random_state, size):
    # (r_Mt, epsilon_Mt, sigma^2_Mt)
    epsilon = random_state.normal(size=[T,size])
    process = numpy.zeros(shape=[size,T,3])
    process[:,0,0] = -0.006
    process[:,0,2] = omega/(1-alpha_1-alpha_2)
    for t in range(1,T):
        process[:,t,2] = omega + alpha_1*process[:,t-1,1]**2 + alpha_2*process[:,t-1,2]
        process[:,t,1] = numpy.sqrt(process[:,t,2]) * epsilon[t]
        process[:,t,0] = mu + phi*process[:,t-1,0] + process[:,t,1]
    return process
# augmented Markovian process generator
def generator_augmented(random_state, size):
    # (r_it, r_Mt, epsilon_Mt, sigma^2_Mt)
    process = generator(random_state, size)
    market_return = process[:,:,0]
    process_aug = numpy.concatenate(
        (beta*(market_return[:,:,numpy.newaxis]-rf) + rf,process),
        axis=-1,
    )
    return process_aug
# Markov chain discretization
sample_paths = generator(numpy.random.RandomState(0),size=1000)
return_sample_paths = sample_paths[:,:,0]
var_sample_paths = sample_paths[:,:,2]
price_sample_paths = numpy.cumprod(numpy.exp(return_sample_paths),axis=1)
markovian = Markovian(generator,n_Markov_states=[1]+[100]*(T-1),n_sample_paths=1000000)
markovian.SA()
# augment to 103 dimension
Markov_states = [None for _ in range(T)]
transition_matrix = markovian.transition_matrix
for t in range(T):
    market_return = markovian.Markov_states[t][:,0].reshape(-1,1)
    asset_return_market_exposure = beta*(market_return-rf) + rf
    Markov_states[t] = numpy.concatenate(
        (asset_return_market_exposure,markovian.Markov_states[t]), axis=1)
# comparison of the true process vs the Markov chain approximation
fig, ax = plt.subplots(1,2,figsize=(10,5),sharey=True)
fig = fan_plot(return_sample_paths, ax=ax[0])
s = markovian.simulate(1000)
fig = fan_plot(s[:,:,-3], ax=ax[1])
ax[0].set_xlabel("stages")
ax[0].set_ylabel("returns")
ax[0].set_title("simulated S&P500 weekly returns")
ax[1].set_xlabel("stages")
ax[1].set_ylabel("returns")
ax[1].set_title("simulated Markov chain process")
fig.tight_layout()
fig.savefig("./result/asset_MCA.png",dpi=1200)
true = price_sample_paths[:,-1]
simulated = numpy.cumprod(numpy.exp(s[:,:,-3]),axis=1)[:,-1]
result = {
    'mean': numpy.mean(true)*100-100,
    'std':numpy.std(true)*100,
    'VaR':numpy.quantile(true,0.05)*100-100,
    'skewness': stats.skew(true),
    'kurtosis': 3+stats.kurtosis(true),
    'Sharpe Ratio': (numpy.mean(true)*100-100-0.551377065)/numpy.std(true)/100,
}
result_simulated = {
    'mean':numpy.mean(simulated)*100-100,
    'std':numpy.std(simulated)*100,
    'VaR':numpy.quantile(simulated,0.05)*100-100,
    'skewness': stats.skew(simulated),
    'kurtosis': 3+stats.kurtosis(simulated),
    'Sharpe Ratio': (numpy.mean(simulated)*100-100-0.551377065)/numpy.std(simulated)/100,
}
pandas.DataFrame(
    [pandas.Series(result_simulated),pandas.Series(result)],
    index=['Simulated','True']
).to_csv("./result/statistics_MCA.csv")
# evaluation table
evaluation_tab = []
evaluationTrue_tab = []

for lambda_ in [0.0,0.25,0.5,0.75]:
    AssetMgt = MSLP(
        T=T, sense=-1, bound=200
    )
    N = 100
    K = 5
    AssetMgt.add_Markovian_uncertainty(generator_augmented)
    for t in range(T):
        m = AssetMgt[t]
        now, past = m.addStateVars(N+1, lb=0, obj=0, name='asset')
        if t == 0:
            buy = m.addVars(N, name='buy')
            sell = m.addVars(N, name='sell')
            m.addConstrs(now[j] == buy[j] - sell[j] for j in range(N))
            m.addConstr(
                now[N] == 100
                - (1+fee) * gurobipy.quicksum(buy[j] for j in range(N))
                + (1-fee) * gurobipy.quicksum(sell[j] for j in range(N))
            )
        elif t != T-1:
            sell = m.addVars(N, name='sell')
            buy = m.addVars(N, name='buy')
            capm = m.addVars(N, lb = -gurobipy.GRB.INFINITY, name='capm')
            idio = m.addVars(N, name='idio')
            m.addConstr(
                now[N] == (
                    (1+rf) * past[N]
                    - (1+fee) * gurobipy.quicksum(buy[j] for j in range(N))
                    + (1-fee) * gurobipy.quicksum(sell[j] for j in range(N))
                )
            )
            m.addConstrs(
                now[j] == capm[j] + idio[j] + buy[j] - sell[j]
                for j in range(N)
            )
            for j in range(N):
                m.addConstr(past[j] == capm[j], uncertainty_dependent={past[j]:j})
                m.addConstr(past[j] == idio[j], uncertainty={past[j]:f(alpha[j],sigma[j])})
        else:
            v = m.addVar(obj=1, lb=-gurobipy.GRB.INFINITY, name='wealth')
            capm = m.addVars(N, lb = -gurobipy.GRB.INFINITY, name='capm')
            idio = m.addVars(N, name='idio')
            m.addConstr(v == gurobipy.quicksum(now[j] for j in range(N+1)))
            m.addConstrs(
                now[j] == capm[j] + idio[j]
                for j in range(N)
            )
            for j in range(N):
                m.addConstr(past[j] == capm[j], uncertainty_dependent={past[j]:j})
                m.addConstr(past[j] == idio[j], uncertainty={past[j]:f(alpha[j],sigma[j])})
            m.addConstr(now[N] == (1+rf) * past[N])
    AssetMgt.discretize(
        n_samples=100,
        method='input',
        Markov_states=Markov_states,
        transition_matrix=transition_matrix,
        random_state=888,
    )
    AssetMgt.set_AVaR(lambda_=lambda_, alpha_=0.25)
    AssetMgt_SDDP = SDDP(AssetMgt)
    AssetMgt_SDDP.solve(max_iterations=50, n_steps=3, n_processes=3)
    evaluation = Evaluation(AssetMgt)
    evaluation.run(n_simulations=1000, random_state=666)
    evaluationTrue = EvaluationTrue(AssetMgt)
    evaluationTrue.run(n_simulations=1000, random_state=666)
    result = {
        'mean':numpy.mean(evaluation.pv)-100,
        'std':numpy.std(evaluation.pv),
        'VAR': numpy.quantile(evaluation.pv,0.05)-100,
        'skewness': stats.skew(evaluation.pv),
        'kurtosis': 3+stats.kurtosis(evaluation.pv),
        'Sharpe Ratio':
            (numpy.mean(evaluation.pv)-100-0.005513771)/numpy.std(evaluation.pv)
    }
    evaluation_tab.append(pandas.Series(result))
    resultTrue = {
        'mean':numpy.mean(evaluationTrue.pv)-100,
        'std':numpy.std(evaluationTrue.pv),
        'VAR': numpy.quantile(evaluationTrue.pv,0.05)-100,
        'skewness': stats.skew(evaluationTrue.pv),
        'kurtosis': 3+stats.kurtosis(evaluationTrue.pv),
        'Sharpe Ratio':
            (numpy.mean(evaluationTrue.pv)-100-0.005513771)/numpy.std(evaluationTrue.pv)
    }
    evaluationTrue_tab.append(pandas.Series(resultTrue))
index = ['risk neutral','0.25-0.25','0.50-0.25','0.75-0.25']
pandas.DataFrame(evaluation_tab,index=index).to_csv("./result/evaluation.csv")
pandas.DataFrame(evaluationTrue_tab,index=index).to_csv("./result/evaluationTrue.csv")
