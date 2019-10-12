Welcome to MSPPy's documentation!
=================================
MSPPy is a Python package to build, solve and
analyze multi-stage stochastic programs. The
package is released under the open source Modified BSD (3-clause) license.

Minimal Example
===============
We first start with a simple news vendor problem. Suppose we want to purchase
some newspaper today and sell it tomorrow. The demand for newspaper tomorrow is
uniformly distributed in 0 to 10. The retail price, production cost, and
recycled value of one newspaper is $2, $1, $0.5. How many newspaper should we buy
today?

.. code-block:: python

    from msppy.msp import MSLP
    from msppy.solver import SDDP
    from msppy.evaluation import Evaluation
    # build the problem
    nvid = MSLP(T=2, sense=-1, bound=10)
    for t in range(2):
        m = nvid[t]
        buy_now, buy_past = m.addStateVar(name='bought', obj=-1.0)
        if t == 1:
            sold = m.addVar(name='sold', obj=2)
            unsatisfied = m.addVar(name='unsatisfied')
            recycled = m.addVar(name='recycled', obj=0.5)
            m.addConstr(sold + unsatisfied == 5, uncertainty={'rhs':range(11)})
            m.addConstr(sold + recycled == buy_past)
    # solve the problem
    sddp = SDDP(nvid)
    sddp.solve(max_iterations=10)
    print(sddp.first_stage_solution)
    # evaluate the solution
    result = Evaluation(nvid)
    result.run(n_simulations=-1)
    print(result.gap)

Get started
===========

.. toctree::
   :maxdepth: 1

   Build the true problem <construct>
   Discretize the true problem <discretize>
   Solve the true problem <solve>
   Real world applications <applications>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
