Build the true problem
======================
The first step is to construct the true problem. The true problem can be either
stage-wise independent or Markovian.

Stage-wise independent true problem
-------------------------------------------------
Construction of a stage-wise independent true problem has two steps.

First, create an MSLP instance by specifying number of time periods T.
This will create a list of T empty StochasticModel objects.

Second, run a for loop and fill in the T StochasticModel objects.

Stage-wise independent finite discrete problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Suppose we want to purchase
some newspaper today and sell it tomorrow. The demand for newspaper tomorrow is
uniformly distributed in 0 to 10. The retail price, production cost, and
recycled value of one newspaper is $2, $1, $0.5. How many newspaper should we buy
today?

.. code-block:: python

    from msppy.msp import MSLP
    nvid = MSLP(T=2, sense=-1, bound=20)
    for t in range(2):
        m = nvid[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=-1.0)
        else:
            _, buy_past = m.addStateVar(name='bought')
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)

We start with building an MSLP project representing a
two-stage stochastic linear program to be maximized. We then run a
loop to fill in each individual stage problem m. The state variable in this
problem is the number of newspaper to buy. So we add this state variable by
addStateVar to each stage problem. The addStateVar will return a tuple that
represents the added state variable and its local copy variable. In stage one,
we add control variables sold, unsatisfied, and recycled that represent the
number of newspaper to sell, the number of ordered newspaper is unsatisfied,
and the number of newspaper to recycle. We also add the
demand-supply constraint by addConstr. The right hand side of this constraint
representing demand has scenarios from 0 to 10 (by default with equal
probability) and is initialized to be 5 (does not matter since it will
be overwritten by the scenarios) . In the end, we add the recycle equality.

Stage-wise independent continuous problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now suppose the uncertain demand follows a lognormal distribution. The problem
can be constructed simply by replace the scenario list with a continuous function.

.. code-block:: python

    from msppy.msp import MSLP
    import numpy as np
    nvic = MSLP(T=2, sense=-1, bound=100)
    def f(random_state):
        return random_state.lognormal(mean=np.log(4),sigma=2)
    for t in range(2):
        m = nvic[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t == 1:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0, uncertainty={'rhs':f})
            m.addConstr(sold + recycled == buy_past)

Markovian true problems
-----------------------
Construction of a Markovian true problem has three steps.

First, create an MSLP instance by specifying number of time periods T.
This will create a list of T empty StochasticModel objects.

Second, add Markovian uncertainty. The Markovian uncertainty can be a
(non-homogeneous) Markov chain process or a Markovian continuous process.
The Markov chain process should be specified by a Markov state space and a
transition matrix. The Markovian continuous process should be specified by a
sample path generator.

Third, run a for loop and fill in the T StochasticModel objects. Each dimension of the
Markovian uncertainty is added during the process.


Markov chain problems
~~~~~~~~~~~~~~~~~~~~~
Continue with the simple news vendor problem. Suppose now we are dealing with
a three-stage news vendor problem with a Markov chain demand. Specifically, In
the first stage, there is no demand. In the second stage, the initial Markov
state are [4,6] with equal probability. In the third stage, the Markov states
are [4,6] and the transition matrix from stage two to stage three is
[[0.3,0.7],[0.7,0.3]].

.. code-block:: python

    from msppy.msp import MSLP
    nvmc = MSLP(T=3, sense=-1, bound=100)
    nvmc.add_MC_uncertainty(
        Markov_states=[[[0]],[[4],[6]],[[4],[6]]],
        transition_matrix=[
            [[1]],
            [[0.5,0.5]],
            [[0.3,0.7],[0.7,0.3]]
        ]
    )
    for t in range(3):
        m = nvmc[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t != 0:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty_dependent={'rhs':0})
            m.addConstr(sold + recycled == buy_past)


Markovian continuous problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Suppose now the demand follows a time series :math:`d_t=0.5\times d_{t-1}+\epsilon_{t}`,
where :math:`\epsilon_{t}` is i.i.d and follows a lognormal distribution.

.. code-block:: python

    from msppy.msp import MSLP
    import numpy as np
    nvm = MSLP(T=3, sense=-1, bound=500)
    def sample_path_generator(random_state, size):
        a = np.zeros([size,3,1])
        for t in range(1,3):
            a[:,t,:] = (0.5 * a[:,t-1,:]
                + random_state.lognormal(2.5,1,size=[size,1]))
        return a
    nvm.add_Markovian_uncertainty(sample_path_generator)
    for t in range(3):
        m = nvm[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t != 0:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0, uncertainty_dependent={'rhs':0})
            m.addConstr(sold + recycled == buy_past)

Risk averse problem
-------------------
Besides expectation, the package also provides a built-in risk measure, called as
Expectation_AVaR, that is
a linear combination of expectation and average value at risk (AVaR). Risk measures
can be added as a direct method (without adding additional things and directly
changing the coefficients and rhs of cutting planes) or an indirect method
(adding additional state variables to transform the problem into risk neutral).
This two methods are equivalent when the numbers of samples times the parameter
of Value-at-risk happen to be integer for each stage.

The following snippet constructs a stage-wise independent finite discrete
problem in Expectation_AVaR. The parameter l represents the weight given to AVaR
and a represents the parameter of value-of-risk.

.. code-block:: python

    from msppy.msp import MSLP
    import numpy as np
    nvica = MSLP(T=2, sense=-1, bound=100)
    for t in range(2):
        m = nvica[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t == 1:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)
    nvica.set_AVaR(l=0.5, a=0.1, method='direct')

Integer problems
----------------
The variables are in fact integer. This can be achieved by created MSIP instance.
The program can also be binarized (this is useful for tighter cuts).

.. code-block:: python

    from msppy.msp import MSIP
    import numpy as np
    nvidi = MSIP(T=2, sense=-1, bound=100)
    for t in range(2):
        m = nvidi[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0, vtype='I')
        if t == 1:
            sold = m.addVar(name='sold', obj=2, vtype='I')
            unsatisfied = m.addVar(name='unsatisfied', vtype='I')
            recycled = m.addVar(name='recycled', obj=0.5, vtype='I')
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)


Infinite horizon problems
-------------------------
Infinite horizon problem can be built by specifying infinity argument to be True.

.. code-block:: python

    from msppy.msp import MSLP
    rhs = [[0,2,4,6,8],[1,3,5,7,9]]
    nvidinf = MSLP(T=3, discount=0.9, sense=-1, bound=20, infinity=True)
    for t in range(3):
        m = nvidinf[t]
        if t == 0:
            buy_now, _ = m.addStateVar(name='bought', obj=-1.0)
        else:
            _, buy_past = m.addStateVar(name='bought')
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 0,
                uncertainty={'rhs':rhs[t-1]})
            m.addConstr(sold + recycled == buy_past)



Module Reference
----------------
The StochasticModel class
~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: msppy.sp.StochasticModel
.. autofunction:: addStateVars
.. autofunction:: addStateVar
.. autofunction:: addVars
.. autofunction:: addVar
.. autofunction:: addConstrs
.. autofunction:: addConstr
.. autofunction:: set_probability
.. autofunction:: add_continuous_uncertainty

The MSLP class
~~~~~~~~~~~~~~
.. currentmodule:: msppy.msp
.. autoclass:: MSLP
.. currentmodule:: msppy.msp.MSLP

.. autofunction:: add_MC_uncertainty
.. autofunction:: add_Markovian_uncertainty
.. autofunction:: set_AVaR
.. autofunction:: write

The MSIP class
~~~~~~~~~~~~~~
.. currentmodule:: msppy.msp.MSIP
.. autofunction:: binarize
