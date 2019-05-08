This tutorial deals with the Brazilian interconnected power system. I direct readers to go through discrete.ipynb first, in which I give a brief description of the problem. I simply regard the historical data as the whole sample space. Though it is an unrealistic assumption, it is easier to get you familiar with the problem. A risk averse version of the problem is given in the risk_averse.ipynb. You will see a risk measure can bring higher bias but lower variance.

A step further, a log-normal distribution can be fitted (stage-wise independently) to the inflow energy. Readers can find it in the continuous.ipynb.

The inflow energy really should be a Markovian process. In the TS_modelling.ipynb, I fit a time series model to the historical data. In order to incorporate this model into our optimization problem, two approaches are implemented, SAA and MCA. In the TS.ipynb, I add additional state variables and reformulate the problem into a stage-wise independent pattern. In the Markovian.ipynb, I use Markovian chain approximation. Visual validation of the MCA can be found in Markov_chain_approximation.ipynb.

For illustation purpose, the number of stages is intentionally set small. The full-period problem is solved in the examples folder.
