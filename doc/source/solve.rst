Solve the true problem
======================

Now that we have constructed the true problem and discretize it. The
MSPPy package provides two kinds of solvers, namely SDDP and extensive, to solve the
problem. After solving the problem, the obtained policy can be evaluated both
on the discretization problem and the true problem.

Stage-wise independent discrete finite problem
----------------------------------------------

Both the extensive solver and SDDP solver shows that, the optimum is $3.18 and
an optimal amount of newspaper to buy today is 7 for the true problem

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP, Extensive
    from msppy.evaluation import Evaluation, EvaluationTrue
    nvid = construct_nvid()
    nvid_ext = Extensive(nvid)
    nvid_ext.solve(outputFlag=0)
    nvid_ext.first_stage_solution
    nvid_sddp = SDDP(nvid)
    nvid_sddp.solve(max_iterations=10)
    nvid_sddp.db[-1]
    nvid_sddp.first_stage_solution
    res = EvaluationTrue(nvid)
    res.run(n_simulations=-1)
    res.gap

Stage-wise independent continuous problem
-----------------------------------------

In this case, for both solvers, the problem needs to be discretized. The
following snippet discretizes the data process by 100 number of samples and a
random seed 888. We refer the resulting disretization problem as SAA(1).
Both solvers show that, for SAA(1), the optimum is $2.79 and
an optimal amount of newspaper to buy today is 9.08


.. code-block:: python

    from msppy.utils.examples import construct_nvic
    from msppy.solver import SDDP, Extensive
    from msppy.evaluation import Evaluation, EvaluationTrue
    nvic = construct_nvic()
    nvic.discretize(random_state=1, n_samples=100)
    nvic_ext = Extensive(nvic)
    nvic_ext.solve(outputFlag=0)
    nvic_ext.first_stage_solution
    nvic_sddp = SDDP(nvic)
    nvic_sddp.solve(max_iterations=10)
    nvic_sddp.db[-1]
    nvic_sddp.first_stage_solution
    res = Evaluation(nvic)
    res.run(n_simulations=-1)
    res.gap
    res_true = EvaluationTrue(nvid)
    res_true.run(n_simulations=3000, percentile=95)
    res_true.CI

Our intention is solve the true problem rather than the SAA(1). The
advantage of SDDP solver is that by solving SAA(1), it also offers a policy to
the true problem (though not optimal). By simulation as shown above, the 95%
confidence interval
of the value of the obtained policy on the true problem is [3.10,3.37].
Therefore, we conclude that by solving
SAA(1), the SDDP solver provides you with a policy whose value is within
[3.10,3.37] with 95% confidence for the true problem.

Markov chain problem
--------------------

Both the extensive solver and SDDP solver shows that, the optimum is $9.05
and an optimal amount of newspaper to buy today is 6 for the true problem.

.. code-block:: python

    from msppy.utils.examples import construct_nvmc
    from msppy.solver import SDDP, Extensive
    from msppy.evaluation import Evaluation, EvaluationTrue
    nvmc = construct_nvmc()
    nvmc_ext = Extensive(nvmc)
    nvmc_ext.solve()
    nvmc_ext.first_stage_solution
    nvmc_sddp = SDDP(nvmc)
    nvmc_sddp.solve(max_iterations=10)
    nvmc_sddp.db[-1]
    nvmc_sddp.first_stage_solution
    res = Evaluation(nvmc)
    res.run(n_simulations=-1)
    res.gap

Markovian continuous problem
----------------------------

In this case, for both solvers, the problem needs to be discretized. The
following snippet ten dimensional Markov chain using stochastic approximation
method (stochastic gradient descent) with 1000 iterations. We refer the
resulting disretization problem as MC. Both solvers show that, for MC, the
optimum is $2.67 and an optimal amount of newspaper to buy today is 17.76

.. code-block:: python

    from msppy.utils.examples import construct_nvm
    from msppy.solver import SDDP, Extensive
    from msppy.evaluation import Evaluation, EvaluationTrue
    nvm = construct_nvm()
    nvm.discretize(n_Markov_states=10, n_sample_paths=1000, method='SA');
    nvm_ext = Extensive(nvm)
    nvm_ext.solve(outputFlag=0)
    nvm_ext.first_stage_solution
    nvm_sddp = SDDP(nvm)
    nvm_sddp.solve(max_iterations=100, logToConsole=0)
    nvm_sddp.db[-1]
    nvm_sddp.first_stage_solution
    res = Evaluation(nvm)
    res.run(n_simulations=-1)
    res.gap
    res_true = EvaluationTrue(nvm)
    res_true.run(n_simulations=3000)
    res_true.CI

But our intention is solve the true problem rather than the MC. The advantage of
SDDP solver is that by solving MC, it also offers a policy to the true problem
(though not guaranteed to be optimal). By simulation as above, the 95%
confidence interval of the value of the obtained policy on the true problem is
[26.54,27.99]. Therefore, we conclude that by solving MC, the SDDP solver
provides you with a policy whose value is within [26.54,27.99] with 95%
confidence for the true problem.

Risk averse problem
-------------------
Direct method

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP
    nvid = construct_nvid()
    nvid.set_AVaR(l=0.5, a=0.1, method='direct')
    nvid_sddp = SDDP(nvid)
    nvid_sddp.solve(max_iterations=10)
    nvid_sddp.first_stage_solution

Indirect method

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP
    nvid = construct_nvid()
    nvid.set_AVaR(l=0.5, a=0.1, method='indirect')
    nvid_sddp = SDDP(nvid)
    nvid_sddp.solve(max_iterations=20)
    nvid_sddp.first_stage_solution

Biased sampling
~~~~~~~~~~~~~~~~~~~~~
For both finite-horizon and infinite horizon risk averse problems, we provide a biased sampling scheme to solve the problem. Details can be found in the `paper <https://www.sciencedirect.com/science/article/abs/pii/S0377221720300989>`_. A biased sampling scheme is in essence a change of the refrence probability measure making `bad' scenarios more frequent and is shown to improve the rate of convergence. 

Finite horizon

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP
    nvid = construct_nvid()
    nvid.set_AVaR(l=0.5, a=0.1)
    nvid_sddp = SDDP(nvid, biased_sampling = True)
    nvid_sddp.solve(max_iterations=20)

Inifinite horizon

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP
    nvid = construct_nvid()
    nvid.set_AVaR(l=0.5, a=0.1)
    nvid_sddp = PSDDP(nvid, biased_sampling = True)
    nvid_sddp.solve(max_iterations=20)



Integer problem
---------------

SDDiP solver and Extensive solver can solve MSIP problems. The built-in cuts
provided are Benders' cut, strengthened Benders' cut and Lagaragian cut.

.. code-block:: python

    from msppy.utils.examples import construct_nvidi
    from msppy.solver import SDDiP, Extensive
    nvidi = construct_nvidi()
    nvidi_ext = Extensive(nvidi)
    nvidi_ext.solve(outputFlag=0)
    nvidi_ext.first_stage_solution
    nvidi_sddip = SDDiP(nvidi)
    nvidi_sddip.solve(max_iterations=10, cuts=['SB'])
    nvidi_sddip.db[-1]
    nvidi_sddip.first_stage_solution


Infinite horizon problem
------------------------

PSDDP solver use periodical SDDP algorithm to solve MSLP problems.
Recall that we have constructed a 3-stage MSLP with period of 2 stages.
So the backward pass will add cuts to these 3 stages. In the following snippet,
forward_T is set to be 10, meaning that the forward pass will solve the first
10 stages and select trial solutions.

.. code-block:: python

    from msppy.utils.examples import construct_nvidinf
    from msppy.solver import PSDDP
    nvidinf = construct_nvidinf()
    nvidinf_sddp = PSDDP(nvidinf)
    nvidinf_sddp.solve(max_iterations=10, forward_T=10)
    nvidinf_sddp.db[-1]
    nvidinf_sddp.first_stage_solution
    from msppy.evaluation import Evaluation
    res = Evaluation(nvidinf)
    res.run(n_simulations=1000, T=100, n_processes=3)
    res.gap

Stopping criterion of the SDDP solver
-------------------------------------

SDDP algorithm is an iterative algorithm and thus we need to specify when
to stop the algorithm. Besides simple numeric stopping criterion (max_iterations, max_time,
max_stable_iterations, see documentation below), more sophisticated criterion
are available. Returns back to the stage-wise independent finite discrete
problem introduced at the beginning of this section, in which I specify
max_iterations to be 10. Because this problem is tiny, one can be certain
that after 10 iterations, it converges. When problem is large-scale and we
cannot estimate a good threshold for number of iterations or computational
time. We need something else.

Optimality gap
~~~~~~~~~~~~~~

The following snippet specifies an optimality-gap based stopping criteria.
Specifically, the SDDP solver evaluates the obtained policy
every 1 iteration by compute the exact expected policy value (turning off
simulation). If the gap becomes not larger than 1e-4, the algorithm
will be stopped. We can see that the algorithm stops after 6 iterations.

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP
    nvid = construct_nvid()
    nvid_sddp = SDDP(nvid)
    nvid_sddp.solve(freq_evaluations=1, n_simulations=-1, tol=1e-4)
    nvid_sddp.iteration
    nvid_sddp.db[-1]
    nvid_sddp.first_stage_solution

Stabilization
~~~~~~~~~~~~~

The following snippet specifies a stabilization based stopping criteria.
Specifically, the SDDP solver compares the policy every 1 iteration by computing the
difference of the expected policy values. If the difference becomes
not larger than 1e-4, the algorithm will be stopped. We can see that again, the
algorithm stops after 6 iterations.

.. code-block:: python

    from msppy.utils.examples import construct_nvid
    from msppy.solver import SDDP
    nvid = construct_nvid()
    nvid_sddp = SDDP(nvid)
    nvid_sddp.solve(freq_comparisons=1, n_simulations=-1, tol=1e-4)
    nvid_sddp.iteration
    nvid_sddp.db[-1]
    nvid_sddp.first_stage_solution

Module Reference
----------------
The Solver module
~~~~~~~~~~~~~~~~~

.. automodule:: msppy.solver
   :members:

The Evaluation module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: msppy.evaluation
   :members:
