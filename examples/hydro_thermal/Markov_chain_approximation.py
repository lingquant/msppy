import pandas
import numpy
from msppy.utils.plot import fan_plot
from msppy.discretize import Markovian
import matplotlib.pyplot as plt
import time
import sys

# a script to make Markov chain approximation
n_Markov_states = int(sys.argv[1])
n_sample_paths = int(sys.argv[2])
method = sys.argv[3]

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

T = 120
inflow_initial = numpy.array([[41248.7153,7386.860854,10124.56146,6123.808537]])

def generator(random_state,size):
    inflow = numpy.empty([size,T,4])
    inflow[:,0,:] = inflow_initial
    for t in range(T-1):
        noise = numpy.exp(random_state.multivariate_normal(mean=[0]*4, cov=sigma[t%12],size=size))
        inflow[:,t+1,:] = noise * (
            (1-gamma[t%12]) * exp_mu[t%12]
            + gamma[t%12] * exp_mu[t%12]/exp_mu[(t-1)%12] * inflow[:,t,:]
        )
    return inflow

start = time.time()
markovian = Markovian(
    f=generator,
    n_Markov_states=[1]+[n_Markov_states]*(T-1),
    n_sample_paths=n_sample_paths,
)
Markov_states, transition_matrix = getattr(markovian,method)()
markovian.write("./{}/".format(method+str(n_Markov_states)))
print(time.time()-start)
