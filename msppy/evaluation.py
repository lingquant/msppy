#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
from msppy.utils.statistics import rand_int,check_random_state,compute_CI
import pandas
import time
import numpy

class _Evaluation(object):
    """Evaluation base class.

    Parameters
    ----------
    MSP: list
        A multi-stage stochastic program object.

    Attributes
    ----------
    pv: list
        The simulated policy values.

    epv: float
        The exact value of expected policy value (only available for
        approximation model).

    CI: tuple
        The CI of simulated policy values.

    gap: float
        The gap between upper end of the CI and deterministic bound.

    stage_cost: dataframe
        The cost of individual stage models.

    solution: dataframe
        The solution of queried variables.

    Methods
    -------
    run:
        Run simulations on the approximation model.
    """
    def __init__(self, MSP):
        self.MSP = MSP
        self.db = MSP.db
        self.pv = None
        self.CI = None
        self.epv = None
        self.gap = None
        self.stage_cost = None
        self.solution = None

    def _compute_gap(self):
        try:
            MSP = self.MSP
            if self.CI is not None:
                if MSP.sense == 1:
                    self.gap = abs( (self.CI[1]-self.db) / self.db )
                else:
                    self.gap = abs( (self.db-self.CI[0]) / self.db )
            elif self.epv is not None:
                self.gap = abs( (self.epv-self.db) / self.db )
            else:
                self.gap = abs( (self.pv[0]-self.db) / self.db )
        except ZeroDivisionError:
            self.gap = 'nan'

class Evaluation(_Evaluation):
    __doc__ = _Evaluation.__doc__
    def run(
            self,
            n_simulations,
            percentile=95,
            query=None,
            query_stage_cost=False,
            random_state=None,):
        """Run a Monte Carlo simulation to evaluate the policy on the
        approximation model.

        Parameters
        ----------
        n_simulations: int/-1
            If int: the number of simulations;
            If -1: exhuastive evaluation.

        query: list, optional (default=None)
            The names of variables that are intended to query.

        query_stage_cost: bool, optional (default=False)
            Whether to query values of individual stage costs.

        percentile: float, optional (default=95)
            The percentile used to compute the confidence interval.

        random_state: int, RandomState instance or None, optional
            (default=None)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used by numpy.random.
        """
        random_state = check_random_state(random_state)
        query = [] if query is None else list(query)
        MSP = self.MSP
        if n_simulations == -1:
            n_sample_paths, sample_paths = MSP._enumerate_sample_paths(MSP.T-1)
        else:
            n_sample_paths = n_simulations
        ub = [0] * n_sample_paths
        if query_stage_cost:
            stage_cost = [
                [0 for _ in range(n_sample_paths)] for _ in range(MSP.T)
            ]
        solution = {item: [[] for _ in range(MSP.T)] for item in query}
        # forward Sampling
        for j in range(n_sample_paths):
            if n_simulations == -1:
                sample_path = sample_paths[j]
            state = 0
            # time loop
            for t in range(MSP.T):
                if MSP.n_Markov_states == 1:
                    m = MSP.models[t]
                else:
                    if n_simulations == -1:
                        m = MSP.models[t][sample_path[1][t]]
                    else:
                        if t == 0:
                            m = MSP.models[t][0]
                        else:
                            state = random_state.choice(
                                range(MSP.n_Markov_states[t]),
                                p=MSP.transition_matrix[t][state],
                            )
                            m = MSP.models[t][state]
                if t > 0:
                    m._update_link_constrs(forward_solution)
                    if MSP.n_Markov_states == 1:
                        scenario_index = (
                            sample_path[t]
                            if n_simulations == -1
                            else rand_int(
                                m.n_samples, random_state, m.probability
                            )
                        )
                    else:
                        scenario_index = (
                            sample_path[0][t]
                            if n_simulations == -1
                            else rand_int(
                                m.n_samples, random_state, m.probability
                            )
                        )
                    m._update_uncertainty(scenario_index)
                m.optimize()
                if m.status not in [2,11]:
                    m.write_infeasible_model("evaluation_" + str(m.modelName))
                forward_solution = MSP._get_forward_solution(m, t)
                for var in m.getVars():
                    if var.varName in query:
                        solution[var.varName][t].append(var.X)
                if query_stage_cost:
                    stage_cost[t][i] = MSP._get_stage_cost(m, t)
                ub[j] += MSP._get_stage_cost(m, t)
            #! time loop
        #! forward Sampling
        self.pv = ub
        if n_simulations == -1:
            self.epv = numpy.dot(
                ub,
                [
                    MSP._compute_weight_sample_path(sample_paths[j])
                    for j in range(n_sample_paths)
                ],
            )
        if n_simulations not in [-1,1]:
            self.CI = compute_CI(ub, percentile)
        self._compute_gap()
        self.solution = {k: pandas.DataFrame(v) for k, v in solution.items()}
        if query_stage_cost:
            self.stage_cost = pandas.DataFrame(stage_cost)

class EvaluationTrue(Evaluation):
    __doc__ = Evaluation.__doc__
    def run(
            self,
            n_simulations,
            query=None,
            query_stage_cost=False,
            random_state=None,
            percentile=95):
        """Run a Monte Carlo simulation to evaluate a policy on the true problem.

        Parameters
        ----------
        n_simulations: int
            The number of simulations.

        query: list, optional (default=None)
            The names of variables that are intended to query.

        percentile: float, optional (default=95)
            The percentile used to compute the confidence interval.

        query_stage_cost: bool, optional (default=False)
            Whether to query values of individual stage costs.

        random_state: int, RandomState instance or None, optional
            (default=None)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used by numpy.random.
        """
        MSP = self.MSP
        if MSP.__class__.__name__ == 'MSIP':
            MSP._back_binarize()
        # discrete finite model should call evaluate instead
        if (
            MSP._type in ["stage-wise independent", "Markov chain"]
            and MSP._individual_type == "original"
            and not hasattr(MSP,"bin_stage")
        ):
            return super().run(
                n_simulations=n_simulations,
                query=query,
                query_stage_cost=query_stage_cost,
                percentile=percentile,
                random_state=random_state,
            )
        if n_simulations <= 0:
            raise ValueError("number of simulations must be bigger than 0")
        random_state = check_random_state(random_state)
        if MSP._type == "Markovian":
            samples = MSP.Markovian_uncertainty(random_state,n_simulations)
            label_all = numpy.zeros([n_simulations,MSP.T],dtype=int)
            for t in range(1,MSP.T):
                dist = numpy.empty([n_simulations,MSP.n_Markov_states[t]])
                for idx, markov_state in enumerate(MSP.Markov_states[t]):
                    temp = samples[:,t,:] - markov_state
                    dist[:,idx] = numpy.sum(temp**2, axis=1)
                label_all[:,t] = numpy.argmin(dist,axis=1)
        query = [] if query is None else list(query)
        ub = [0] * n_simulations
        if query_stage_cost:
            stage_cost = [[0 for _ in range(n_simulations)] for _ in range(MSP.T)]
        solution = {item: [[] for _ in range(MSP.T)] for item in query}
        # forward Sampling
        for j in range(n_simulations):
            # Markov chain uncertainty state
            if MSP._type == "Markov chain":
                state = 0
            # time loop
            for t in range(MSP.T):
                # sample Markovian uncertainties
                if MSP._type == "Markovian":
                    if t == 0:
                        m = MSP.models[t][0]
                    else:
                        # use the model with the closest markov state
                        m = MSP.models[t][label_all[j][t]]
                        # update Markovian uncertainty
                        m._update_uncertainty_dependent(samples[j][t])
                elif MSP._type == "Markov chain":
                    if t == 0:
                        m = MSP.models[t][0]
                    else:
                        state = random_state.choice(
                            range(MSP.n_Markov_states[t]),
                            p=MSP.transition_matrix[t][state],
                        )
                        m = MSP.models[t][state]
                else:
                    m = MSP.models[t]
                # sample independent uncertainties
                if t > 0:
                    if m._type == "continuous":
                        m._sample_uncertainty(random_state)
                    elif m._flag_discrete == 1:
                        m._update_uncertainty_discrete(
                            rand_int(
                                m.n_samples_discrete,random_state, m.probability)
                        )
                    else:
                        m._update_uncertainty(
                            rand_int(m.n_samples, random_state, m.probability)
                        )
                    m._update_link_constrs(forward_solution)
                m.optimize()
                if m.status not in [2,11]:
                    m.write_infeasible_model("evaluation_true_" + str(m.modelName))
                # get solutions
                forward_solution = MSP._get_forward_solution(m, t)
                for var in m.getVars():
                    if var.varName in query:
                        solution[var.varName][t].append(var.X)
                if query_stage_cost:
                    stage_cost[t].append(MSP._get_stage_cost(m, t))
                ub[j] += MSP._get_stage_cost(m, t)
                if MSP._type == "Markovian":
                    m._update_uncertainty_dependent(
                        MSP.Markov_states[t][label_all[j][t]])
            #! end time loop
        #! forward Sampling
        self.solution = {k: pandas.DataFrame(v) for k, v in solution.items()}
        if query_stage_cost:
            self.stage_cost = pandas.DataFrame(stage_cost)
        self.pv = ub
        if n_simulations != 1:
            self.CI = compute_CI(ub, percentile)
