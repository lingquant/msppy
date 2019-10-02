Hydro-thermal Power System Planning
===================================
This tutorial deals with the Brazilian interconnected power system. I suggest
readers to go through stage-wise independent finite discrete problem first, in
which I give a brief description of the problem and a simple implementation.
I simply regard the historical data as the whole sample space. Though it is an
unrealistic assumption, the purpose of the notebook is to get you familiar with
the problem.

Take a step further. The inflow energy should be a Markovian process. A time series
model can be fitted to the historical data. The coefficients of the time
series equations are output to the data folder. In order to incorporate this
data process into our optimization problem, two approaches are implemented --
time series approach and Markov chain approach.

The time series approach adds additional
state variables and reformulate the problem into a stage-wise independent form.

For Markov chain approaches, the process of discretization is quite involved.
One can certainly make discretization and solve the problem in the same
interface using the msppy package (e.g., the portfolio optimization quick start
example), but I intentionally separate the routine for illustration purpose.

.. toctree::
   :maxdepth: 1

   Introduction <introduction.ipynb>
   Model the random inflow energy <modeling.ipynb>
   Discretize the Markovian continuous uncertainty <discretization.ipynb>
   Solve the problem: TS approach <TS.ipynb>
   Solve the problem: Markovian approach <Markovian.ipynb>
   Solve the problem: Stationary approach <infinity.ipynb>
