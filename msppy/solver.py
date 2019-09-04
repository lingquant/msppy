#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
from msppy.utils.logger import LoggerSDDP,LoggerEvaluation,LoggerComparison
from msppy.utils.statistics import check_random_state,rand_int,compute_CI,allocate_jobs
from msppy.evaluation import Evaluation, EvaluationTrue
import time
import numpy
import multiprocessing
import gurobipy
import numbers
from collections import abc
import pandas


class SDDP(object):
    """SDDP solver base class.

    Parameters
    ----------
    MSP: list
        A multi-stage stochastic program object.

    Attributes
    ----------
    db: list
        The deterministic bounds obtained in backward steps.

    pv: list
        The policy values obtained in forward steps.

    bounds: DataFrame
        The DataFrame stores db & pv.

    iteration: int
        The number of iteration in SDDP algorithm.

    percentile: float
        The percentile to make CI.

    n_processes: int
        The number of processes to run in parallel. Run serial SDDP if 1.
        If n_steps is 1, n_processes is coerced to be 1.

    n_steps: int
        The number of forward/backward steps to run in each cut iteration.
        It is coerced to be 1 if n_processes is 1.

    cut_type: list
        It is ["B"] for the base class

    cut_type_list: list
        It is [["B"] for t in range(self.MSP.T-1)] for the bse class

    start: int
        The start stage index to add cut (it should be -1 for SDDP_infinity)

    Methods
    -------
    Solve():
        use SDDP algorithm to solve a MSLP instance.
    """

    def __init__(self, MSP, reset=True):
        self.db = []
        self.pv = []
        self.cut_type = ["B"]
        self.cut_type_list = [["B"] for t in range(MSP.T-1)]
        if reset:
            MSP._reset()
        self.MSP = MSP
        self.iteration = 0
        self.n_processes = 1
        self.n_steps = 1
        self.percentile = 95

    def __repr__(self):
        return (
            "<SDDP solver instance, {} processes, {} steps>"
            .format(self.n_processes, self.n_steps)
        )

    def _forward(
            self,
            random_state=None,
            sample_path_idx=None,
            markovian_idx=None,
            markovian_samples=None,
            solve_true=False,
            query=None,
            query_dual=None,
            query_stage_cost=None):
        """Single forward step. """
        MSP = self.MSP
        forward_solution = [None for _ in range(MSP.T)]
        pv = 0
        query = [] if query is None else list(query)
        query_dual = [] if query_dual is None else list(query_dual)
        solution = {item: numpy.full(MSP.T,numpy.nan) for item in query}
        solution_dual = {item: numpy.full(MSP.T,numpy.nan) for item in query_dual}
        stage_cost = numpy.full(MSP.T,numpy.nan)
        # time loop
        for t in range(MSP.T):
            if MSP._type == "stage-wise independent":
                m = MSP.models[t]
            else:
                if t == 0:
                    m = MSP.models[t][0]
                    state = 0
                else:
                    if sample_path_idx is not None:
                        state = sample_path_idx[1][t]
                    elif markovian_idx is not None:
                        state = markovian_idx[t]
                    else:
                        state = random_state.choice(
                            range(MSP.n_Markov_states[t]),
                            p=MSP.transition_matrix[t][state]
                        )
                    m = MSP.models[t][state]
                    if markovian_idx is not None:
                        m._update_uncertainty_dependent(markovian_samples[t])
            if t > 0:
                m._update_link_constrs(forward_solution[t-1])
                # exhaustive evaluation when the sample paths are given
                if sample_path_idx is not None:
                    if MSP._type == "stage-wise independent":
                        scen = sample_path_idx[t]
                    else:
                        scen = sample_path_idx[0][t]
                    m._update_uncertainty(scen)
                # true stagewise independent randomness is infinite and solve
                # for true
                elif m._type == 'continuous' and solve_true:
                    m._sample_uncertainty(random_state)
                # true stagewise independent randomness is large and solve
                # for true
                elif m._type == 'discrete' and m._flag_discrete == 1 and solve_true:
                    scen = rand_int(
                        k=m.n_samples_discrete,
                        probability=m.probability,
                        random_state=random_state,
                    )
                    m._update_uncertainty(scen)
                # other cases include
                # 1: true stagewise independent randomness is infinite and solve
                # for approximation problem
                # 2: true stagewise independent randomness is large and solve
                # for approximation problem
                # 3: true stagewise independent randomness is small. In this
                # case, true problem and approximation problem are the same.
                else:
                    scen = rand_int(
                        k=m.n_samples,
                        probability=m.probability,
                        random_state=random_state,
                    )
                    m._update_uncertainty(scen)
            m.optimize()
            if m.status not in [2,11]:
                m.write_infeasible_model("backward_" + str(m.modelName))
            forward_solution[t] = MSP._get_forward_solution(m, t)
            for var in m.getVars():
                if var.varName in query:
                    solution[var.varName][t] = var.X
            for constr in m.getConstrs():
                if constr.constrName in query_dual:
                    solution_dual[constr.constrName][t] = constr.PI
            if query_stage_cost:
                stage_cost[t] = MSP._get_stage_cost(m, t)/pow(MSP.discount, t)
            pv += MSP._get_stage_cost(m, t)
            if markovian_idx is not None:
                m._update_uncertainty_dependent(MSP.Markov_states[t][markovian_idx[t]])
        #! time loop
        if query == [] and query_dual == [] and query_stage_cost is None:
            return {
                'forward_solution':forward_solution,
                'pv':pv
            }
        else:
            return {
                'solution':solution,
                'soultion_dual':solution_dual,
                'stage_cost':stage_cost,
                'forward_solution':forward_solution,
                'pv':pv
            }

    def _add_and_store_cuts(
        self, t, rhs, grad, cuts=None, cut_type=None, j=None
    ):
        """Store cut information (rhs and grad) to cuts for the j th step, for cut
        type cut_type and for stage t."""
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            MSP.models[t-1]._add_cut(rhs[0], grad[0])
            if cuts is not None:
                cuts[t-1][cut_type][j][:] = numpy.append(rhs[0], grad[0])
        else:
            for k in range(MSP.n_Markov_states[t-1]):
                rhs_ = numpy.dot(MSP.transition_matrix[t][k], rhs)
                grad_ = numpy.dot(MSP.transition_matrix[t][k], grad)
                MSP.models[t-1][k]._add_cut(rhs_, grad_)
                if cuts is not None:
                    cuts[t-1][cut_type][j][k][:] = numpy.append(rhs_, grad_)

    def _backward(self, forward_solution, j=None, lock=None, cuts=None):
        """Single backward step of SDDP serially or in parallel.

        Parameters
        ----------
        forward_solution:
            feasible solutions obtained from forward step

        j: int
            index of forward sampling

        lock: multiprocessing.Lock

        cuts: dict
            A dictionary stores cuts coefficients and rhs.
            Key of the dictionary is the cut type. Value of the dictionary is
            the cut coefficients and rhs.
        """
        MSP = self.MSP
        for t in range(MSP.T-1, 0, -1):
            models = (
                [MSP.models[t]]
                if MSP.n_Markov_states == 1
                else MSP.models[t]
            )
            n_Markov_states = (
                1 if MSP.n_Markov_states == 1 else MSP.n_Markov_states[t]
            )
            objLP = [None] * n_Markov_states
            gradLP = [None] * n_Markov_states
            for k, m in enumerate(models):
                if MSP.n_Markov_states != 1:
                    m._update_link_constrs(forward_solution[t-1])
                objLPScen, gradLPScen = m._solveLP()
                objLP[k], gradLP[k] = (
                    m._average(objLPScen),
                    m._average(gradLPScen),
                )
                objLP[k] -= numpy.dot(gradLP[k], forward_solution[t-1])
            self._add_and_store_cuts(t, objLP, gradLP, cuts, "B", j)
            self.add_cuts_additional_procedure(t, objLP, gradLP, cuts, "B", j)

    def add_cuts_additional_procedure(self, t, rhs, grad, cuts=None,
            cut_type=None, j=None):
        pass

    def _SDDP_single(self):
        """A single serial SDDP step. Returns the policy value."""
        # random_state is constructed by number of iteration.
        random_state = numpy.random.RandomState(self.iteration)
        temp = self._forward(random_state)
        forward_solution = temp['forward_solution']
        pv = temp['pv']
        self._backward(forward_solution)
        return [pv]

    def _SDDP_single_process(self, pv, jobs, lock, cuts):
        """Multiple SDDP jobs by single process. pv will store the policy values.
        cuts will store the cut information. Have not use the lock parameter so
        far."""
        # random_state is constructed by the number of iteration and the index
        # of the first job that the current process does
        random_state = numpy.random.RandomState([self.iteration, jobs[0]])
        for j in jobs:
            temp = self._forward(random_state)
            forward_solution = temp['forward_solution']
            pv[j] = temp['pv']
            self._backward(forward_solution, j, lock, cuts)

    def _add_cut_from_multiprocessing_array(self, cuts):
        for t in range(self.MSP.T-1):
            for cut_type in self.cut_type_list[t]:
                for cut in cuts[t][cut_type]:
                    if self.MSP.n_Markov_states == 1:
                        self.MSP.models[t]._add_cut(rhs=cut[0], gradient=cut[1:])
                    else:
                        for k in range(self.MSP.n_Markov_states[t]):
                            self.MSP.models[t][k]._add_cut(
                                rhs=cut[k][0], gradient=cut[k][1:])
        self.add_cut_from_multiprocessing_array_additional_procedure(cuts)

    def add_cut_from_multiprocessing_array_additional_procedure(self, cuts):
        pass

    def _remove_redundant_cut(self, clean_stages):
        for t in clean_stages:
            M = (
                [self.MSP.models[t]]
                if self.MSP.n_Markov_states == 1
                else self.MSP.models[t]
            )
            for m in M:
                m.update()
                constr = m.cuts
                for idx, cut in enumerate(constr):
                    if cut.sense == '>': cut.sense = '<'
                    elif cut.sense == '<': cut.sense = '>'
                    flag = 1
                    for k in range(m.n_samples):
                        m._update_uncertainty(k)
                        m.optimize()
                        if m.status == 4:
                            m.Params.DualReductions = 0
                            m.optimize()
                        if m.status not in [3,11]:
                            flag = 0
                    if flag == 1:
                        m._remove_cut(idx)
                    else:
                        if cut.sense == '>': cut.sense = '<'
                        elif cut.sense == '<': cut.sense = '>'
                m.update()

    def _compute_cut_type(self):
        pass

    def _SDDP_multiprocessesing(self):
        """Prepare a collection of multiprocessing arrays to store cuts.
        Cuts are stored in the form of:
         Independent case (index: t, cut_type, j):
            {t:{cut_type: [cut_coeffs_and_rhs]}
         Markovian case (index: t, cut_type, j, k):
            {t:{cut_type: [[cut_coeffs_and_rhs]]}
        """
        procs = [None] * self.n_processes
        if self.MSP.n_Markov_states == 1:
            cuts = {
                t:{
                    cut_type: [multiprocessing.RawArray("d",
                        [0] * (self.MSP.n_states[t]+1))
                        for _ in range(self.n_steps)]
                    for cut_type in self.cut_type}
            for t in range(self.MSP.T-1)}
        else:
            cuts = {
                t:{
                    cut_type: [
                        [multiprocessing.RawArray("d",
                            [0] * (self.MSP.n_states[t]+1))
                            for _ in range(self.MSP.n_Markov_states[t])]
                        for _ in range(self.n_steps)]
                    for cut_type in self.cut_type}
            for t in range(self.MSP.T-1)}

        pv = multiprocessing.Array("d", [0] * self.n_steps)
        lock = multiprocessing.Lock()

        for p in range(self.n_processes):
            procs[p] = multiprocessing.Process(
                target=self._SDDP_single_process,
                args=(pv, self.jobs[p], lock, cuts),
            )
            procs[p].start()
        for proc in procs:
            proc.join()

        self._add_cut_from_multiprocessing_array(cuts)

        return [item for item in pv]

    def solve(
            self,
            n_processes=1,
            n_steps=1,
            max_iterations=10000,
            max_stable_iterations=10000,
            max_time=1000000.0,
            tol=0.001,
            freq_evaluations=None,
            percentile=95,
            tol_diff=float("-inf"),
            random_state=None,
            evaluation_true=False,
            freq_comparisons=None,
            n_simulations=3000,
            n_simulations_true=3000,
            query=None,
            query_dual=None,
            query_stage_cost=None,
            n_periodical_stages=None,
            freq_clean=None,
            logFile=1,
            logToConsole=1,
            directory=''):
        """Solve approximation model.

        Parameters
        ----------

        n_processes: int, optional (default=1)
            The number of processes to run in parallel. Run serial SDDP if 1.
            If n_steps is 1, n_processes is coerced to be 1.

        n_steps: int, optional (default=1)
            The number of forward/backward steps to run in each cut iteration.
            It is coerced to be 1 if n_processes is 1.

        max_iterations: int, optional (default=10000)
            The maximum number of iterations to run SDDP.

        max_stable_iterations: int, optional (default=10000)
            The maximum number of iterations to have same deterministic bound

        tol: float, optional (default=1e-3)
            tolerance for convergence of bounds

        freq_evaluations: int, optional (default=None)
            The frequency of evaluating gap on approximation model. It will be
            ignored if risk averse

        percentile: float, optional (default=95)
            The percentile used to compute confidence interval

        diff: float, optional (default=-inf)
            The stablization threshold

        freq_comparisons: int, optional (default=None)
            The frequency of comparisons of policies

        n_simulations: int, optional (default=10000)
            The number of simluations to run when evaluating a policy
            on approximation model

        freq_clean: int/list, optional (default=None)
            The frequency of removing redundant cuts.
            If int, perform cleaning at the same frequency for all stages.
            If list, perform cleaning at different frequency for each stage;
            must be of length T-1 (the last stage does not have any cuts).

        random_state: int, RandomState instance or None, optional (default=None)
            Used in evaluations and comparisons. (In the forward step, there is
            an internal random_state which is not supposed to be changed.)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used by numpy.random.

        logFile: binary, optional (default=1)
            Switch of logging to log file

        logToConsole: binary, optional (default=1)
            Switch of logging to console
        """
        MSP = self.MSP
        if freq_clean is not None:
            if isinstance(freq_clean, (numbers.Integral, numpy.integer)):
                freq_clean = [freq_clean] * (MSP.T-1)
            if isinstance(freq_clean, ((abc.Sequence, numpy.ndarray))):
                if len(freq_clean) != MSP.T-1:
                    raise ValueError("freq_clean list must be of length T-1!")
            else:
                raise TypeError("freq_clean must be int/list instead of {}!"
                .format(type(freq_clean)))
        if not MSP._flag_update:
            MSP._update()
        stable_iterations = 0
        total_time = 0
        a = time.time()
        gap = 1.0
        right_end_of_CI = float("inf")
        db_past = MSP.bound
        self.percentile = percentile
        # distinguish pv_sim from pv
        pv_sim_past = None

        if n_processes != 1:
            self.n_steps = n_steps
            self.n_processes = min(n_steps, n_processes)
            self.jobs = allocate_jobs(self.n_steps, self.n_processes)

        logger_sddp = LoggerSDDP(
            logFile=logFile,
            logToConsole=logToConsole,
            n_processes=self.n_processes,
            percentile=self.percentile,
            directory=directory,
        )
        logger_sddp.header()
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation = LoggerEvaluation(
                n_simulations=n_simulations,
                percentile=percentile,
                logFile=logFile,
                logToConsole=logToConsole,
                directory=directory,
            )
            logger_evaluation.header()
        if freq_comparisons is not None:
            logger_comparison = LoggerComparison(
                n_simulations=n_simulations,
                percentile=percentile,
                logFile=logFile,
                logToConsole=logToConsole,
                directory=directory,
            )
            logger_comparison.header()
        try:
            while (
                self.iteration < max_iterations
                and total_time < max_time
                and stable_iterations < max_stable_iterations
                and tol < gap
                and tol_diff < right_end_of_CI
            ):
                start = time.time()

                self._compute_cut_type()

                if self.n_processes == 1:
                    pv = self._SDDP_single()
                else:
                    pv = self._SDDP_multiprocessesing()

                m = (
                    MSP.models[0]
                    if MSP.n_Markov_states == 1
                    else MSP.models[0][0]
                )
                m.optimize()
                if m.status not in [2,11]:
                    m.write_infeasible_model(
                        "backward_" + str(m._model.modelName) + ".lp"
                    )
                db = m.objBound
                self.db.append(db)
                MSP.db = db
                if self.n_processes != 1:
                    CI = compute_CI(pv,percentile)
                self.pv.append(pv)

                if self.iteration >= 1:
                    if db_past == db:
                        stable_iterations += 1
                    else:
                        stable_iterations = 0
                self.iteration += 1
                db_past = db

                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time

                if self.n_processes == 1:
                    logger_sddp.text(
                        iteration=self.iteration,
                        db=db,
                        pv=pv[0],
                        time=elapsed_time,
                    )
                else:
                    logger_sddp.text(
                        iteration=self.iteration,
                        db=db,
                        CI=CI,
                        time=elapsed_time,
                    )
                if (
                    freq_evaluations is not None
                    and self.iteration%freq_evaluations == 0
                    or freq_comparisons is not None
                    and self.iteration%freq_comparisons == 0
                ):
                    directory = '' if directory is None else directory
                    start = time.time()
                    evaluation = Evaluation(MSP)
                    evaluation.run(
                        n_simulations=n_simulations,
                        query=query,
                        query_dual=query_dual,
                        query_stage_cost=query_stage_cost,
                        percentile=percentile,
                        n_processes=n_processes,
                        n_periodical_stages=n_periodical_stages
                    )
                    pandas.DataFrame(evaluation.pv).to_csv(directory+
                        "iter_{}_pv.csv".format(self.iteration))
                    if query is not None:
                        for item in query:
                            evaluation.solution[item].to_csv(directory+
                                "iter_{}_{}.csv".format(self.iteration, item))
                    if query_dual is not None:
                        for item in query_dual:
                            evaluation.solution_dual[item].to_csv(directory+
                                "iter_{}_{}.csv".format(self.iteration, item))
                    if query_stage_cost:
                        evaluation.stage_cost.to_csv(directory+
                            "iter_{}_stage_cost.csv".format(self.iteration))
                    if evaluation_true:
                        evaluationTrue = EvaluationTrue(MSP)
                        evaluationTrue.run(
                            n_simulations=n_simulations,
                            query=query,
                            query_dual=query_dual,
                            query_stage_cost=query_stage_cost,
                            percentile=percentile,
                            n_processes=n_processes,
                            n_periodical_stages=n_periodical_stages
                        )
                    pandas.DataFrame(evaluationTrue.pv).to_csv(directory+
                        "iter_{}_pv.csv".format(self.iteration))
                    if query is not None:
                        for item in query:
                            evaluationTrue.solution[item].to_csv(directory+
                                "iter_{}_{}_true.csv".format(self.iteration, item))
                    if query_dual is not None:
                        for item in query_dual:
                            evaluationTrue.solution_dual[item].to_csv(directory+
                                "iter_{}_{}_true.csv".format(self.iteration, item))
                    if query_stage_cost:
                        evaluationTrue.stage_cost.to_csv(directory+
                            "iter_{}_stage_cost_true.csv".format(self.iteration))
                    elapsed_time = time.time() - start
                    gap = evaluation.gap
                    if n_simulations == -1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            pv=evaluation.epv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    elif n_simulations == 1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            pv=evaluation.pv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    else:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            CI=evaluation.CI,
                            gap=gap,
                            time=elapsed_time,
                        )
                if (
                    freq_comparisons is not None
                    and self.iteration%freq_comparisons == 0
                ):
                    start = time.time()
                    pv_sim = evaluation.pv
                    if self.iteration / freq_comparisons >= 2:
                        diff = MSP.sense*(numpy.array(pv_sim_past)-numpy.array(pv_sim))
                        if n_simulations == -1:
                            diff_mean = numpy.mean(diff)
                            right_end_of_CI = diff_mean
                        else:
                            diff_CI = compute_CI(diff, self.percentile)
                            right_end_of_CI = diff_CI[1]
                        elapsed_time = time.time() - start
                        if n_simulations == -1:
                            logger_comparison.text(
                                iteration=self.iteration,
                                ref_iteration=self.iteration-freq_comparisons,
                                diff=diff_mean,
                                time=elapsed_time,
                            )
                        else:
                            logger_comparison.text(
                                iteration=self.iteration,
                                ref_iteration=self.iteration-freq_comparisons,
                                diff_CI=diff_CI,
                                time=elapsed_time,
                            )
                    pv_sim_past = pv_sim
                if freq_clean is not None:
                    clean_stages = [
                        t
                        for t in range(1,MSP.T-1)
                        if self.iteration%freq_clean[t] == 0
                    ]
                    if len(clean_stages) != 0:
                        self._remove_redundant_cut(clean_stages)
                # self._clean()
        except KeyboardInterrupt:
            stop_reason = "interruption by the user"
        # SDDP iteration stops
        MSP.db = self.db[-1]
        if self.iteration >= max_iterations:
            stop_reason = "iteration:{} has reached".format(max_iterations)
        if total_time >= max_time:
            stop_reason = "time:{} has reached".format(max_time)
        if stable_iterations >= max_stable_iterations:
            stop_reason = "stable iteration:{} has reached".format(max_stable_iterations)
        if gap <= tol:
            stop_reason = "convergence tolerance:{} has reached".format(tol)
        if right_end_of_CI <= tol_diff:
            stop_reason = "stablization threshold:{} has reached".format(tol_diff)

        b = time.time()
        logger_sddp.footer(reason=stop_reason)
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation.footer()
        if freq_comparisons is not None:
            logger_comparison.footer()
        self.total_time = total_time
    @property
    def first_stage_solution(self):
        return (
            {var.varName: var.X for var in self.MSP.models[0].states}
            if self.MSP.n_Markov_states == 1
            else {var.varName: var.X for var in self.MSP.models[0][0].states}
        )

    def plot_bounds(self, start=0, window=1, smooth=0, ax=None):
        """plot the evolution of bounds

        Parameters
        ----------
        ax: Matplotlib AxesSubplot instance, optional
            The specified subplot is used to plot; otherwise a new figure is created.

        window: int, optional (default=1)
            The length of the moving windows to aggregate the policy values. If
            length is bigger than 1, approximate confidence interval of the
            policy values and statistical bounds will be plotted.

        smooth: bool, optional (default=0)
            If 1, fit a smooth line to the policy values to better visualize
            the trend of statistical values/bounds.

        start: int, optional (default=0)
            The start iteration to plot the bounds. Set start to other values
            can zoom in the evolution of bounds in most recent iterations.

        Returns
        -------
        matplotlib.pyplot.figure instance
        """
        from msppy.utils.plot import plot_bounds
        return plot_bounds(self.db, self.pv, self.MSP.sense, self.percentile,
        start=start, window=window, smooth=smooth, ax=ax)
    @property
    def bounds(self):
        df = pandas.DataFrame.from_records(self.pv)
        df['db'] = self.db
        return df

class SDDiP(SDDP):
    __doc__ = SDDP.__doc__

    def solve(
            self,
            cuts,
            pattern=None,
            relax_stage=None,
            level_step_size=0.2929,
            level_max_stable_iterations=1000,
            level_max_iterations=1000,
            level_max_time=1000,
            level_mip_gap=1e-4,
            level_tol=1e-3,
            *args,
            **kwargs):
        """Call SDDiP solver to solve approximation model.

        Parameters
        ----------
        n_processes: int, optional (default=1)
            The number of processes to run in parallel. Run serial SDDP if 1.
            If n_steps is 1, n_processes is coerced to be 1.

        n_steps: int, optional (default=1)
            The number of forward/backward steps to run in each cut iteration.
            It is coerced to be 1 if n_processes is 1.

        max_iterations: int, optional (default=10000)
            The maximum number of iterations to run SDDP.

        max_stable_iterations: int, optional (default=10000)
            The maximum number of iterations to have the same deterministic bound

        tol: float, optional (default=1e-3)
            tolerance for convergence of bounds

        freq_evaluations: int, optional (default=None)
            The frequency of evaluating gap on approximation model. It will be
            ignored if risk averse

        percentile: float, optional (default=95)
            The percentile used to compute confidence interval

        diff: float, optional (default=-inf)
            The stablization threshold

        freq_comparisons: int, optional (default=None)
            The frequency of comparisons of policies

        n_simulations: int, optional (default=10000)
            The number of simluations to run when evaluating a policy
            on approximation model

        freq_clean: int/list, optional (default=None)
            The frequency of removing redundant cuts.
            If int, perform cleaning at the same frequency for all stages.
            If list, perform cleaning at different frequency for each stage;
            must be of length T-1 (the last stage does not have any cuts).

        random_state: int, RandomState instance or None, optional (default=None)
            Used in evaluations and comparisons. (In the forward step, there is
            an internal random_state which is not supposed to be changed.)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used by numpy.random.

        logFile: binary, optional (default=1)
            Switch of logging to log file

        logToConsole: binary, optional (default=1)
            Switch of logging to console

        cuts: list
            Entries of the list could be 'B','SB','LG'

        pattern: dict, optional (default=None)
            The pattern of adding cuts can be cyclical or barrier-in.
            See the example below.

        relax_stage: int, optional (default=None)
            All stage models after relax_stage (exclusive) will be relaxed.

        level_step_size: float, optional (default=0.2929)
            Step size for level method.

        level_max_stable_iterations: int, optional (default=1000)
            The maximum number of iterations to have the same deterministic g_*
            for the level method.

        level_mip_gap: float, optional (default=1e-4)
            The MIPGap used when solving the inner problem for the level method.

        level_max_iterations: int, optional (default=1000)
            The maximum number of iterations to run for the level method.

        level_max_time: int, optional (default=1000)
            The maximum number of time to run for the level method.

        level_tol: float, optional (default=1e-3)
            Tolerance for convergence of bounds for the level method.

        Examples
        --------
        Assume the cuts given is ["B","SB","LG"], the pattern can be:
            {"cycle": (4, 1, 1)}, meaning for every six iterations adding:
                Benders' cuts for the first four,
                strengthened Benders' cuts for the fifth,
                Lagrangian cuts for the last.
            {"in": (0, 4, 5)}, meaning adding:
                Benders' cuts from the beginning,
                Strengthened Benders' cuts from the fourth iteration,
                Lagragian cuts from the fifth iteration.
        """
        if pattern != None:
            if not all(
                len(item) == len(cuts)
                for item in pattern.values()
            ):
                raise Exception("pattern is not compatible with cuts!")
        self.relax_stage = relax_stage if relax_stage != None else self.MSP.T - 1
        self.cut_type = cuts
        self.cut_pattern = pattern
        self.level_step_size = level_step_size
        self.level_max_stable_iterations = level_max_stable_iterations
        self.level_max_iterations = level_max_iterations
        self.level_max_time = level_max_time
        self.level_mip_gap = level_mip_gap
        self.level_tol = level_tol
        super().solve(*args, **kwargs)

    def _backward(self, forward_solution, j=None, lock=None, cuts=None):
        MSP = self.MSP
        for t in range(MSP.T-1, 0, -1):
            models = (
                [MSP.models[t]]
                if MSP.n_Markov_states == 1
                else MSP.models[t]
            )
            n_Markov_states = (
                1 if MSP.n_Markov_states == 1
                else MSP.n_Markov_states[t]
            )
            objLP = [None] * n_Markov_states
            gradLP = [None] * n_Markov_states
            objSB = [None] * n_Markov_states
            objLG = [None] * n_Markov_states
            gradLG = [None] * n_Markov_states
            for k, model in enumerate(models):
                if MSP.n_Markov_states != 1:
                    model._update_link_constrs(forward_solution[t-1])
                model.update()
                m = model.relax() if model.isMIP else model
                objLPScen, gradLPScen = m._solveLP()
                # B and SB share the gradient
                if (
                    "B" in self.cut_type_list[t-1]
                    or "SB" in self.cut_type_list[t-1]
                ):
                    gradLP[k] = m._average(gradLPScen)
                # SB and LG share the same model
                if (
                    "SB" in self.cut_type_list[t-1]
                    or "LG" in self.cut_type_list[t-1]
                ):
                    m = model.copy()
                    # SB and LG relax the link constraints
                    m._delete_link_constrs()
                # compute objective in all types of cuts
                if "B" in self.cut_type_list[t-1]:
                    objLP[k] = m._average(objLPScen)
                    objLP[k] -= numpy.dot(gradLP[k], forward_solution[t-1])
                if "SB" in self.cut_type_list[t-1]:
                    objSB[k] = m._solveSB(gradLPScen)
                if "LG" in self.cut_type_list[t-1]:
                    objVal_primal = model._solvePrimal()
                    flag_bin = (
                        True if hasattr(self, "n_binaries")
                        else False
                    )
                    objLG[k], gradLG[k] = m._solveLG(
                        gradLPScen=gradLPScen,
                        given_bound=MSP.bound,
                        objVal_primal=objVal_primal,
                        flag_tight = flag_bin,
                        forward_solution=forward_solution[t-1],
                        step_size=self.level_step_size,
                        max_stable_iterations=self.level_max_stable_iterations,
                        max_iterations=self.level_max_iterations,
                        max_time=self.level_max_time,
                        MIPGap=self.level_mip_gap,
                        tol=self.level_tol,
                    )
            #! Markov states iteration ends
            if "B" in self.cut_type_list[t-1]:
                self._add_and_store_cuts(t, objLP, gradLP, cuts, "B", j)
            if "SB" in self.cut_type_list[t-1]:
                self._add_and_store_cuts(t, objSB, gradLP, cuts, "SB", j)
            if "LG" in self.cut_type_list[t-1]:
                self._add_and_store_cuts(t, objLG, gradLG, cuts, "LG", j)
        #! Time iteration ends

    def _compute_cut_type_by_iteration(self):
        if self.cut_pattern == None:
            return list(self.cut_type)
        else:
            if "cycle" in self.cut_pattern.keys():
                cycle = self.cut_pattern["cycle"]
                ## decide pos belongs to which interval ##
                interval = numpy.cumsum(cycle) - 1
                pos = self.iteration % sum(cycle)
                for i in range(len(interval)):
                    if pos <= interval[i]:
                        return [self.cut_type[i]]
            if "in" in self.cut_pattern.keys():
                barrier_in = self.cut_pattern["in"]
                cut = []
                for i in range(len(barrier_in)):
                    if self.iteration >= barrier_in[i]:
                        cut.append(self.cut_type[i])
                if "B" in cut and "SB" in cut:
                    cut.remove("B")
                return cut

    def _compute_cut_type_by_stage(self, t, cut_type):
        if t > self.relax_stage or self.MSP.isMIP[t] != 1:
            cut_type = ["B"]
        return cut_type

    def _compute_cut_type(self):
        cut_type_list = [None] * (self.MSP.T-1)
        cut_type_by_iteration = self._compute_cut_type_by_iteration()
        for t in range(1, self.MSP.T):
            cut_type_list[t-1] = self._compute_cut_type_by_stage(
                t, cut_type_by_iteration)
        self.cut_type_list = cut_type_list

class SDDP_infinity(SDDP):


    def add_cuts_additional_procedure(
        self, t, rhs, grad, cuts=None, cut_type=None, j=None
    ):
        """Store cut information (rhs and grad) to cuts for the j th step, for cut
        type cut_type and for stage t."""
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            if t == 1:
                MSP.models[-1]._add_cut(
                    rhs[0],
                    grad[0]
                )
        else:
            raise NotImplementedError

    def add_cut_from_multiprocessing_array_additional_procedure(self, cuts):
        for cut_type in self.cut_type_list[0]:
            for cut in cuts[0][cut_type]:
                if self.MSP.n_Markov_states == 1:
                    self.MSP.models[-1]._add_cut(
                        rhs=cut[0],
                        gradient=cut[1:]
                    )
                else:
                    raise NotImplementedError

    def _forward(
            self,
            random_state=None,
            sample_path_idx=None,
            markovian_idx=None,
            markovian_samples=None,
            solve_true=False,
            query=None,
            query_dual=None,
            query_stage_cost=None):
        """Single forward step. """
        MSP = self.MSP
        T = MSP.n_periodical_stages
        forward_solution = [None for _ in range(T)]
        pv = 0
        query = [] if query is None else list(query)
        query_dual = [] if query_dual is None else list(query_dual)
        solution = {item: numpy.full(T,numpy.nan) for item in query}
        solution_dual = {item: numpy.full(T,numpy.nan) for item in query_dual}
        stage_cost = numpy.full(T,numpy.nan)
        # time loop
        for t in range(T):
            idx = t%(MSP.T-1) if (t%(MSP.T-1) != 0 or t == 0) else -1
            if MSP._type == "stage-wise independent":
                m = MSP.models[idx]
            else:
                raise NotImplementedError
            if t > 0:
                m._update_link_constrs(forward_solution[t-1])
                # exhaustive evaluation when the sample paths are given
                if sample_path_idx is not None:
                    if MSP._type == "stage-wise independent":
                        scen = sample_path_idx[t]
                    else:
                        scen = sample_path_idx[0][t]
                    m._update_uncertainty(scen)
                # true stagewise independent randomness is infinite and solve
                # for true
                elif m._type == 'continuous' and solve_true:
                    m._sample_uncertainty(random_state)
                # true stagewise independent randomness is large and solve
                # for true
                elif m._type == 'discrete' and m._flag_discrete == 1 and solve_true:
                    scen = rand_int(
                        k=m.n_samples_discrete,
                        probability=m.probability,
                        random_state=random_state,
                    )
                    m._update_uncertainty(scen)
                # other cases include
                # 1: true stagewise independent randomness is infinite and solve
                # for approximation problem
                # 2: true stagewise independent randomness is large and solve
                # for approximation problem
                # 3: true stagewise independent randomness is small. In this
                # case, true problem and approximation problem are the same.
                else:
                    scen = rand_int(
                        k=m.n_samples,
                        probability=m.probability,
                        random_state=random_state,
                    )
                    m._update_uncertainty(scen)
            m.optimize()
            if m.status not in [2,11]:
                m.write_infeasible_model("forward_" + str(m.modelName))
            forward_solution[t] = MSP._get_forward_solution(m, t)
            for var in m.getVars():
                if var.varName in query:
                    solution[var.varName][t] = var.X
            for constr in m.getConstrs():
                if constr.constrName in query_dual:
                    solution_dual[constr.constrName][t] = constr.PI
            if query_stage_cost:
                stage_cost[t] = MSP._get_stage_cost(m, t)/pow(MSP.discount, t)
            pv += MSP._get_stage_cost(m, t)
            if markovian_idx is not None:
                raise NotImplementedError
        #! time loop
        idx = numpy.arange(MSP.T-1)

        # method one
        # for t in range(1,MSP.T):
        #     indices = numpy.arange(t-1,T,MSP.T-1)
        #     n_indices = len(indices)
        #     idx[t-1] = indices[int(rand_int(
        #         k=n_indices,
        #         random_state=random_state,
        #     ))]
        #     MSP.models[t]._update_link_constrs(forward_solution[idx[t-1]])
        # forward_solution_shuffled = [forward_solution[item] for item in idx]
        # if query == [] and query_dual == [] and query_stage_cost is None:
        #     return forward_solution_shuffled, pv

        # method two
        pick_from_two = rand_int(2,random_state)
        if pick_from_two == 1:
            idx[0] = MSP.T - 1
        for t in range(1,MSP.T):
            MSP.models[t]._update_link_constrs(forward_solution[idx[t-1]])
        forward_solution_shuffled = [forward_solution[item] for item in idx]
        if query == [] and query_dual == [] and query_stage_cost is None:
            return {
                'forward_solution':forward_solution_shuffled,
                'pv':pv
            }
        else:
            return {
                'solution':solution,
                'soultion_dual':solution_dual,
                'stage_cost':stage_cost,
                'forward_solution':forward_solution_shuffled,
                'pv':pv
            }

        # method four
        # for t in range(1,MSP.T):
        #     MSP.models[t]._update_link_constrs(forward_solution[t-1])
        # if query == [] and query_dual == [] and query_stage_cost is None:
        #     return forward_solution[:MSP.T-1], pv
        # else:
        #     return {
        #         'solution':solution,
        #         'soultion_dual':solution_dual,
        #         'stage_cost':stage_cost,
        #         'forward_solution':forward_solution,
        #         'pv':pv
        #     }


class Extensive(object):
    """Extensive solver class.

    Attributes
    ----------
    MSP:
        The multistage stochastic program

    extensive_model:
        The constructed extensive model

    solving_time:
        The time cost in solving extensive model

    construction_time:
        The time cost in constructing extensive model

    Methods
    -------
    solve:
        Solve the extensive model
    """

    def __init__(self, MSP, reset=True):
        if reset:
            MSP._reset()
        self.MSP = MSP
        self.solving_time = None
        self.construction_time = None
        self.total_time = None

    def __getattr__(self, name):
        try:
            return getattr(self.extensive_model, name)
        except AttributeError:
            raise AttributeError("no attribute named {}".format(name))

    def solve(self, **kwargs):
        """Call extensive solver to solve approximation model. It will first
        construct the extensive model and then call Gurobi solver to solve it.

        Parameters
        ----------
        **kwargs: optional
            Gurobipy attributes to specify on extensive model.
        """
        # extensive solver is able to solve MSLP with CTG or without CTG
        self.MSP._check_individual_stage_models()
        self.MSP._check_multistage_model()

        construction_start_time = time.time()

        self.extensive_model = gurobipy.Model()
        self.extensive_model.modelsense = self.MSP.sense

        for k, v in kwargs.items():
            setattr(self.extensive_model.Params, k, v)

        self._construct_extensive()

        construction_end_time = time.time()
        solving_start_time = time.time()
        self.extensive_model.optimize()
        solving_end_time = time.time()

        self.construction_time = construction_end_time - construction_start_time
        self.solving_time = solving_end_time - solving_start_time
        self.total_time = self.construction_time + self.solving_time

        return self.extensive_model.objVal

    def _construct_extensive(self):
        """Construct extensive model."""
        MSP = self.MSP
        T = MSP.T
        n_Markov_states = MSP.n_Markov_states
        n_samples = (
            [MSP.models[t].n_samples for t in range(T)]
            if n_Markov_states == 1
            else [MSP.models[t][0].n_samples for t in range(T)]
        )
        n_states = MSP.n_states
        # check if CTG variable is added or not
        initial_model = (
            MSP.models[0] if n_Markov_states == 1 else MSP.models[0][0]
        )
        flag_CTG = 1 if initial_model.alpha is not None else -1
        # |       stage 0       |        stage 1       | ... |       stage T-1      |
        # |local_copies, states | local_copies, states | ... | local_copies, states |
        # |local_copies,        | local_copies,        | ... | local_copies, states |
        # extensive formulation only includes necessary variables
        states = None
        sample_paths = None
        if flag_CTG == 1:
            stage_cost = None
        for t in reversed(range(T)):
            M = [MSP.models[t]] if n_Markov_states == 1 else MSP.models[t]
            # stage T-1 needs to add the states. sample path corresponds to
            # current node.
            if t == T-1:
                n_sample_paths, sample_paths = MSP._enumerate_sample_paths(t)
                states = [
                    self.extensive_model.addVars(sample_paths)
                    for _ in range(n_states[t])
                ]
            # new_states is the local_copies. new_sample_paths corresponds to
            # previous node
            if t != 0:
                n_new_sample_paths, new_sample_paths = MSP._enumerate_sample_paths(
                    t-1)
                new_states = [
                    self.extensive_model.addVars(new_sample_paths)
                    for _ in range(n_states[t-1])
                ]
                if flag_CTG == 1:
                    new_stage_cost = {
                        new_sample_path: 0
                        for new_sample_path in new_sample_paths
                    }
            else:
                new_states = [
                    self.extensive_model.addVars(sample_paths)
                    for _ in range(n_states[t])
                ]

            for j in range(n_samples[t]):
                for k, m in enumerate(M):
                    # copy information from model in scenario j and markov state
                    # k.
                    m._update_uncertainty(j)
                    m.update()
                    # compute sample paths that go through the current node
                    current_sample_paths = (
                        [
                            item
                            for item in sample_paths
                            if item[0][t] == j and item[1][t] == k
                        ]
                        if n_Markov_states != 1
                        else [item for item in sample_paths if item[t] == j]
                    )
                    controls_ = m.controls
                    states_ = m.states
                    local_copies_ = m.local_copies
                    controls_dict = {v: i for i, v in enumerate(controls_)}
                    states_dict = {v: i for i, v in enumerate(states_)}
                    local_copies_dict = {
                        v: i for i, v in enumerate(local_copies_)
                    }

                    for current_sample_path in current_sample_paths:
                        if t != 0:
                            # compute sample paths that go through the
                            # ancester node
                            past_sample_path = (
                                current_sample_path[:-1]
                                if n_Markov_states == 1
                                else (
                                    current_sample_path[0][:-1],
                                    current_sample_path[1][:-1],
                                )
                            )
                        else:
                            past_sample_path = current_sample_path

                        if flag_CTG == -1 or t == 0:
                            weight = MSP.discount ** (
                                t
                            ) * MSP._compute_weight_sample_path(
                                current_sample_path
                            )
                        else:
                            currentWeight = MSP._compute_current_weight_sample_path(
                                current_sample_path)

                        for i in range(n_states[t]):
                            obj = (
                                states_[i].obj * numpy.array(weight)
                                if flag_CTG == -1 or t == 0
                                else 0
                            )
                            states[i][current_sample_path].lb = states_[i].lb
                            states[i][current_sample_path].ub = states_[i].ub
                            states[i][current_sample_path].obj = obj
                            states[i][current_sample_path].vtype = states_[
                                i
                            ].vtype
                            states[i][current_sample_path].varName = states_[
                                i
                            ].varName + str(current_sample_path).replace(
                                " ", ""
                            )
                            # cost-to-go update
                            if t != 0 and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                    states[i][current_sample_path]
                                    * states_[i].obj
                                    * currentWeight
                                )

                        if t == 0:
                            for i in range(n_states[t]):
                                new_states[i][current_sample_path].lb = local_copies_[i].lb
                                new_states[i][current_sample_path].ub = local_copies_[i].ub
                                new_states[i][current_sample_path].obj = local_copies_[i].obj
                                new_states[i][current_sample_path].vtype = local_copies_[i].vtype
                                new_states[i][current_sample_path].varName = local_copies_[i].varname + str(current_sample_path).replace(" ", "")
                        # copy local variables
                        controls = [None for _ in range(len(controls_))]
                        for i, var in enumerate(controls_):
                            obj = (
                                var.obj * weight
                                if flag_CTG == -1 or t == 0
                                else 0
                            )
                            controls[i] = self.extensive_model.addVar(
                                lb=var.lb,
                                ub=var.ub,
                                obj=obj,
                                vtype=var.vtype,
                                name=var.varname
                                + str(current_sample_path).replace(" ", ""),
                            )
                            # cost-to-go update
                            if t != 0 and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                    controls[i] * var.obj * currentWeight
                                )
                        # self.extensive_model.update()
                        # add constraints
                        if t != T - 1 and flag_CTG == 1:
                            self.extensive_model.addConstr(
                                MSP.sense
                                * (
                                    controls[
                                        controls_dict[m.getVarByName("alpha")]
                                    ]
                                    - stage_cost[current_sample_path]
                                )
                                >= 0
                            )
                        for constr_ in m.getConstrs():
                            rhs_ = constr_.rhs
                            expr_ = m.getRow(constr_)
                            lhs = gurobipy.LinExpr()
                            for i in range(expr_.size()):

                                if expr_.getVar(i) in controls_dict.keys():
                                    pos = controls_dict[expr_.getVar(i)]
                                    lhs += expr_.getCoeff(i) * controls[pos]
                                elif expr_.getVar(i) in states_dict.keys():
                                    pos = states_dict[expr_.getVar(i)]
                                    lhs += (
                                        expr_.getCoeff(i)
                                        * states[pos][current_sample_path]
                                    )
                                elif (
                                    expr_.getVar(i) in local_copies_dict.keys()
                                ):
                                    pos = local_copies_dict[expr_.getVar(i)]
                                    if t != 0:
                                        lhs += (
                                            expr_.getCoeff(i)
                                            * new_states[pos][past_sample_path]
                                        )
                                    else:
                                        lhs += (
                                            expr_.getCoeff(i)
                                            * new_states[pos][current_sample_path]
                                        )
                            #! end expression loop
                            self.extensive_model.addConstr(
                                lhs=lhs, sense=constr_.sense, rhs=rhs_
                            )
                        #! end copying the constraints
                    #! end MC loop
                #! end scenarios loop
            #! end scenario loop
            states = new_states
            if flag_CTG == 1:
                stage_cost = new_stage_cost
            sample_paths = new_sample_paths
        # !end time loop
