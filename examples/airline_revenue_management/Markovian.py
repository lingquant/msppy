"""
@author: lingquan
"""
import numpy, pandas
import matplotlib.pyplot as plt
from scipy.stats import beta
import gurobipy
from msppy.msp import MSIP
from msppy.solver import SDDiP
from msppy.evaluation import Evaluation,EvaluationTrue
from msppy.utils.plot import fan_plot
import time

I = ['AH','HA','BH','HB','CH','HC','AHB','BHA','AHC','CHA','BHC','CHB']
J = ['B1','B2','E1','E2','E3','E4']
L = ['AH','HA','BH','HB','CH','HC']
K = ['B','E']
Rate = {'AH':0.1,'HA':0.1,'BH':0.1,'HB':0.1,'CH':0.05,'HC':0.05,
        'AHB':0.05,'BHA':0.05,'AHC':0,'CHA':0,'BHC':0,'CHB':0}
R = {'B':24, 'E':215}
day = [182,126,84,56,35,21,14,10,7,5,3,2,1,0]
T = 14
airFare = pandas.read_csv("./data/airFare.csv", index_col = [0,1])

def demand_sampler(random_state, size):
    demand = numpy.zeros([size,T,len(I)*len(J)])
    for i_idx,i in enumerate(I):
        for j_idx,j in enumerate(J):
            gamma_k,gamma_theta,beta_a,beta_b = airFare.loc[i,j][:4]
            G = random_state.gamma(gamma_k, gamma_theta, size=size) * (1-Rate[i])
            for t in range(1,T):
                B = (
                    beta.cdf(1-day[t]/day[0], beta_a, beta_b)
                    - beta.cdf(1-day[t-1]/day[0], beta_a, beta_b)
                )
                demand[:,t,6*i_idx+j_idx] = random_state.poisson(G*B)
    return demand
mca = {'model':[],'iter_MCA':[],'time':[]}
for t in range(1,15):
    mca[t] = []
summary = {'model':[],'cut':[],'time':[],'gap':[],'bound':[],
    'CI_approx_lower':[], 'CI_approx_upper':[],
    'CI_true_lower':[], 'CI_true_upper':[]}
for idx,n_Markov_states in enumerate([100,200,400]):
    n_sample_paths = 1500 * n_Markov_states
    airline = MSIP(T, sense=-1, bound=1000000000)
    airline.add_Markovian_uncertainty(demand_sampler)
    for t in range(T):
        m = airline[t]
        ## accumulated fulfilled booking requests ##
        B_now, B_past = m.addStateVars(I,J, vtype='I', name="B")
        ## accumulated cancellation ##
        C_now, C_past = m.addStateVars(I,J, vtype='I', name="C")
        if t == 0:
            m.addConstrs(B_now[(i,j)] == 0 for i in I for j in J)
            m.addConstrs(C_now[(i,j)] == 0 for i in I for j in J)
        else:
            # new fulfilled booking requests
            b = m.addVars(I,J, obj=numpy.array(airFare['fare']), name="b", vtype='I')
            c = m.addVars(I,J, obj=-numpy.array(airFare['fare']), name="c", vtype='I')
            # accumulated fulfilled booking requests updated
            m.addConstrs(B_now[(i,j)] == B_past[(i,j)] + b[(i,j)] for i in I for j in J)
            m.addConstrs(C_now[(i,j)] == C_past[(i,j)] + c[(i,j)] for i in I for j in J)
            # number of cancellation is depending on the cancellation rate
            m.addConstrs(C_now[(i,j)] <= Rate[i] * B_now[(i,j)] + 0.5 for i in I for j in J)
            m.addConstrs(C_now[(i,j)] >= Rate[i] * B_now[(i,j)] - 0.5 for i in I for j in J)
            # capacity constraint
            m.addConstrs(
                gurobipy.quicksum(
                    B_now[(i,j)] - C_now[(i,j)] for i in I for j in J
                    if l in i and j.startswith(k)
                )
                <= R[k] for k in K for l in L
            )
            m.addConstrs(
                (b[(i,j)] <= 0 for i in I for j in J),
                uncertainty_dependent=[
                    6*i+j for i in range(len(I)) for j in range(len(J))
                ]
            )
    start_mca = time.time()
    markovian=airline.discretize(
        method='SA',
        n_Markov_states=n_Markov_states,
        n_sample_paths=n_sample_paths,
        int_flag=1,
    )
    time_mca = time.time() - start_mca
    mca['model'].append(idx)
    mca['iter_MCA'].append(n_sample_paths)
    mca['time'].append(time_mca)
    for t in range(1,15):
        mca[t].append(markovian.n_Markov_states[t-1])
    # comparison visually of the true process and its Markov chain approximation
    if idx == 2:
        simulation = markovian.simulate(1000)
        true = demand_sampler(numpy.random.RandomState(0),1000)
        fig_mca, ax_mca = plt.subplots(6,2,figsize=(10,10),sharey='row',sharex=True)
        fig_mca.text(0.5,0.01,'stage',ha='center')
        fig_mca.text(0,0.5,'demand',va='center',rotation='vertical')
        for j_idx,j in enumerate(J):
            fig_mca = fan_plot(true[:,:,j_idx], ax=ax_mca[j_idx][0])
            fig_mca = fan_plot(simulation[:,:,j_idx], ax=ax_mca[j_idx][1])
        fig_mca.tight_layout()
        fig_mca.savefig("./result/airline_MCA.png",dpi=1200)
        fig_bound,ax_bound = plt.subplots(2,1,figsize=(10,5), sharey=True)
    for cut_index,cut in enumerate(['B','SB']):
        airline_sddp = SDDiP(airline)
        airline_sddp.solve(cuts=[cut], n_processes=1, n_steps=1, max_iterations=100)
        result = Evaluation(airline)
        result.run(random_state=666, n_simulations=3000)
        resultTrue = EvaluationTrue(airline)
        resultTrue.run(random_state=666, n_simulations=3000)
        summary['model'].append(idx)
        summary['cut'].append(cut)
        summary['time'].append(airline_sddp.total_time)
        summary['gap'].append(result.gap)
        summary['bound'].append(result.db)
        summary['CI_approx_lower'].append(result.CI[0])
        summary['CI_approx_upper'].append(result.CI[1])
        summary['CI_true_lower'].append(resultTrue.CI[0])
        summary['CI_true_upper'].append(resultTrue.CI[1])
        if idx == 2:
            fig_bound = airline_sddp.plot_bounds(window=1,smooth=1,
                ax=ax_bound[cut_index])
    if idx == 2:
        fig_bound.tight_layout()
        fig_bound.savefig("./result/airline_bounds.png",dpi=1200)
pandas.DataFrame(summary).to_csv('./result/summary.csv')
pandas.DataFrame(mca).to_csv('./result/mca.csv')
