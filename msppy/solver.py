from msppy.utils.logger import LoggerSDDP,LoggerEvaluation,LoggerComparison,Logger
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
    """
    SDDP solver base class.

    Parameters
    ----------
    MSP: list
        A multi-stage stochastic program object.
    """

    def __init__(self, MSP):
        self.db = []
        self.pv = []
        self.MSP = MSP
        self.forward_T = MSP.T
        self.cut_T = MSP.T - 1
        self.cut_type = ["B"]
        self.cut_type_list = [["B"] for t in range(self.cut_T)]
        self.iteration = 0
        self.n_processes = 1
        self.n_steps = 1
        self.percentile = 95

    def __repr__(self):
        return (
            "<{} solver instance, {} processes, {} steps>"
            .format(self.__class__, self.n_processes, self.n_steps)
        )

    def _compute_idx(self, t):
        return (t,t)

    def _select_trial_solution(self, random_state, forward_solution):
        return forward_solution[:-1]

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
        forward_solution = [None for _ in range(self.forward_T)]
        pv = 0
        query = [] if query is None else list(query)
        query_dual = [] if query_dual is None else list(query_dual)
        solution = {item: numpy.full(self.forward_T,numpy.nan) for item in query}
        solution_dual = {item: numpy.full(self.forward_T,numpy.nan) for item in query_dual}
        stage_cost = numpy.full(self.forward_T,numpy.nan)
        # time loop
        for t in range(self.forward_T):
            idx, tm_idx = self._compute_idx(t)
            if MSP._type == "stage-wise independent":
                m = MSP.models[idx]
            else:
                if t == 0:
                    m = MSP.models[idx][0]
                    state = 0
                else:
                    if sample_path_idx is not None:
                        state = sample_path_idx[1][t]
                    elif markovian_idx is not None:
                        state = markovian_idx[t]
                    else:
                        state = random_state.choice(
                            range(MSP.n_Markov_states[idx]),
                            p=MSP.transition_matrix[tm_idx][state]
                        )
                    m = MSP.models[idx][state]
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
            if self.iteration != 0 and self.rgl_a != 0:
                m.regularize(self.rgl_center[t], self.rgl_norm, self.rgl_a,
                self.rgl_b, self.iteration)
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
                m._update_uncertainty_dependent(MSP.Markov_states[idx][markovian_idx[t]])
            if self.iteration != 0 and self.rgl_a != 0:
                m._deregularize()
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
            MSP.models[t-1]._add_cut(rhs, grad)
            if cuts is not None:
                cuts[t-1][cut_type][j][:] = numpy.append(rhs, grad)
        else:
            for k in range(MSP.n_Markov_states[t-1]):
                MSP.models[t-1][k]._add_cut(rhs[k], grad[k])
                if cuts is not None:
                    cuts[t-1][cut_type][j][k][:] = numpy.append(rhs[k], grad[k])

    def _compute_cuts(self, t, m, objLPScen, gradLPScen):
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            return m._average(objLPScen[0], gradLPScen[0])
        objLPScen = objLPScen.reshape(
            MSP.n_Markov_states[t]*MSP.n_samples[t])
        gradLPScen = gradLPScen.reshape(
            MSP.n_Markov_states[t]*MSP.n_samples[t],MSP.n_states[t])
        probability_ind = (
            m.probability if m.probability
            else numpy.ones(m.n_samples)/m.n_samples
        )
        probability = numpy.einsum('ij,k->ijk',MSP.transition_matrix[t],
            probability_ind)
        probability = probability.reshape(MSP.n_Markov_states[t-1],
            MSP.n_Markov_states[t]*MSP.n_samples[t])
        objLP = numpy.empty(MSP.n_Markov_states[t-1])
        gradLP = numpy.empty((MSP.n_Markov_states[t-1],MSP.n_states[t]))
        for k in range(MSP.n_Markov_states[t-1]):
            objLP[k], gradLP[k] = m._average(objLPScen, gradLPScen,
                probability[k])
        return objLP, gradLP

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
            if MSP.n_Markov_states == 1:
                M, n_Markov_states = [MSP.models[t]], 1
            else:
                M, n_Markov_states = MSP.models[t], MSP.n_Markov_states[t]
            objLPScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            gradLPScen = numpy.empty((n_Markov_states, MSP.n_samples[t],
                MSP.n_states[t]))
            for k,m in enumerate(M):
                if MSP.n_Markov_states != 1:
                    m._update_link_constrs(forward_solution[t-1])
                objLPScen[k], gradLPScen[k] = m._solveLP()
            objLP, gradLP = self._compute_cuts(t, m, objLPScen, gradLPScen)
            objLP -= numpy.matmul(gradLP, forward_solution[t-1])
            self._add_and_store_cuts(t, objLP, gradLP, cuts, "B", j)
            self._add_cuts_additional_procedure(t, objLP, gradLP, objLPScen,
            gradLPScen, forward_solution[t-1], cuts, "B", j)

    def _add_cuts_additional_procedure(*args, **kwargs):
        pass

    def _SDDP_single(self):
        """A single serial SDDP step. Returns the policy value."""
        # random_state is constructed by number of iteration.
        random_state = numpy.random.RandomState(self.iteration)
        temp = self._forward(random_state)
        solution = temp['forward_solution']
        pv = temp['pv']
        self.rgl_center = solution
        solution = self._select_trial_solution(random_state, solution)
        self._backward(solution)
        return [pv]

    def _SDDP_single_process(self, pv, jobs, lock, cuts, forward_solution=None):
        """Multiple SDDP jobs by single process. pv will store the policy values.
        cuts will store the cut information. Have not use the lock parameter so
        far."""
        # random_state is constructed by the number of iteration and the index
        # of the first job that the current process does
        random_state = numpy.random.RandomState([self.iteration, jobs[0]])
        for j in jobs:
            temp = self._forward(random_state)
            solution = temp['forward_solution']
            pv[j] = temp['pv']
            # regularization needs to store last forward_solution
            if j == jobs[-1] and self.rgl_a != 0:
                for t in range(self.forward_T):
                    idx,_ = self._compute_idx(t)
                    for i in range(self.MSP.n_states[idx]):
                        forward_solution[t][i] = solution[t][i]
            solution = self._select_trial_solution(random_state, solution)
            self._backward(solution, j, lock, cuts)

    def _add_cut_from_multiprocessing_array(self, cuts):
        for t in range(self.cut_T):
            for cut_type in self.cut_type_list[t]:
                for cut in cuts[t][cut_type]:
                    if self.MSP.n_Markov_states == 1:
                        self.MSP.models[t]._add_cut(rhs=cut[0], gradient=cut[1:])
                    else:
                        for k in range(self.MSP.n_Markov_states[t]):
                            self.MSP.models[t][k]._add_cut(
                                rhs=cut[k][0], gradient=cut[k][1:])

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
                    for cut_type in self.cut_type_list[t]}
            for t in range(self.cut_T)}
        else:
            cuts = {
                t:{
                    cut_type: [
                        [multiprocessing.RawArray("d",
                            [0] * (self.MSP.n_states[t]+1))
                            for _ in range(self.MSP.n_Markov_states[t])]
                        for _ in range(self.n_steps)]
                    for cut_type in self.cut_type_list[t]}
            for t in range(self.cut_T)}

        pv = multiprocessing.Array("d", [0] * self.n_steps)
        lock = multiprocessing.Lock()
        forward_solution = None
        # regularization needs to store last forward_solution
        if self.rgl_a != 0:
            forward_solution = [multiprocessing.Array(
                "d",[0] * self.MSP.n_states[self._compute_idx(t)[0]])
                for t in range(self.forward_T)
            ]

        for p in range(self.n_processes):
            procs[p] = multiprocessing.Process(
                target=self._SDDP_single_process,
                args=(pv, self.jobs[p], lock, cuts, forward_solution),
            )
            procs[p].start()
        for proc in procs:
            proc.join()

        self._add_cut_from_multiprocessing_array(cuts)
        # regularization needs to store last forward_solution
        if self.rgl_a != 0:
            self.rgl_center = [list(item) for item in forward_solution]

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
            query_T=None,
            query_dual=None,
            query_stage_cost=False,
            query_policy_value=False,
            freq_clean=None,
            logFile=1,
            logToConsole=1,
            directory='',
            rgl_norm='L2',
            rgl_a=0,
            rgl_b=0.95):
        """Solve the discretized problem.

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
            The frequency of evaluating gap on the discretized problem. It will
            be ignored if risk averse

        percentile: float, optional (default=95)
            The percentile used to compute confidence interval

        diff: float, optional (default=-inf)
            The stabilization threshold

        freq_comparisons: int, optional (default=None)
            The frequency of comparisons of policies

        n_simulations: int, optional (default=10000)
            The number of simluations to run when evaluating a policy
            on the discretized problem

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

        Examples
        --------

        >>> SDDP().solve(max_iterations=10, max_time=10,
            max_stable_iterations=10)
        Optimality gap based stopping criteria: evaluate the obtained policy
        every freq_evaluations iterations by running n_simulations Monte Carlo
        simulations. If the gap becomes not larger than tol, the algorithm
        will be stopped.
        >>> SDDP().solve(freq_evaluations=10, n_simulations=1000, tol=1e-2)
        Simulation can be turned off; the solver will evaluate the exact expected
        policy value.
        >>> SDDP().solve(freq_evaluation=10, n_simulations=-1, tol=1e-2)
        Stabilization based stopping criteria: compare the policy every
        freq_comparisons iterations by computing the CI of difference of the
        expected policy values. If the upper end of CI becomes not larger
        than tol diff, the algorithm will be stopped.
        >>> SDDP().solve(freq_comparisons=10, n_simulations=1000, tol=1e-2)
        Turn off simulation and

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
        self.rgl_norm = rgl_norm
        self.rgl_a = rgl_a
        self.rgl_b = rgl_b

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
                and (tol_diff < right_end_of_CI or right_end_of_CI < 0)
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
                        query_T=query_T,
                        query_dual=query_dual,
                        query_stage_cost=query_stage_cost,
                        percentile=percentile,
                        n_processes=n_processes,
                    )
                    if query_policy_value:
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
                            query_T=query_T,
                            query_dual=query_dual,
                            query_stage_cost=query_stage_cost,
                            percentile=percentile,
                            n_processes=n_processes,
                        )
                        if query_policy_value:
                            pandas.DataFrame(evaluationTrue.pv).to_csv(directory+
                                "iter_{}_pv_true.csv".format(self.iteration))
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
            stop_reason = "stabilization threshold:{} has reached".format(tol_diff)

        b = time.time()
        logger_sddp.footer(reason=stop_reason)
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation.footer()
        if freq_comparisons is not None:
            logger_comparison.footer()
        self.total_time = total_time

    @property
    def first_stage_solution(self):
        """the obtained solution in the first stage"""
        return (
            {var.varName: var.X for var in self.MSP.models[0].getVars()}
            if self.MSP.n_Markov_states == 1
            else {var.varName: var.X for var in self.MSP.models[0][0].getVars()}
        )

    def plot_bounds(self, start=0, window=1, smooth=0, ax=None):
        """
        plot the evolution of bounds

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
        """dataframe of the obtained bound"""
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
        """Call SDDiP solver to solve the discretized problem.

        Parameters
        ----------
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
        >>> SDDiP().solve(max_iterations=10, cut=['SB'])

        The following cyclical add difference cuts. Specifically, for every six
        iterations add Benders' cuts for the first four,
        strengthened Benders' cuts for the fifth,
        and Lagrangian cuts for the last.

        >>> SDDiP().solve(max_iterations=10, cut=['B','SB','LG'],
        ...     pattern={"cycle": (4, 1, 1)})

        The following add difference cuts from certain iterations. Specifically,
        add Benders' cuts from the beginning,
        Strengthened Benders' cuts from the fourth iteration,
        and Lagragian cuts from the fifth iteration.

        >>> SDDiP().solve(max_iterations=10, cut=['B','SB','LG'],
        ...     pattern={'in': (0, 4, 5)})
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
            if MSP.n_Markov_states == 1:
                M, n_Markov_states = [MSP.models[t]], 1
            else:
                M, n_Markov_states = MSP.models[t], MSP.n_Markov_states[t]
            objLPScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            gradLPScen = numpy.empty((n_Markov_states, MSP.n_samples[t],
                MSP.n_states[t]))
            objSBScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            objLGScen = numpy.empty((n_Markov_states, MSP.n_samples[t]))
            gradLGScen = numpy.empty((n_Markov_states, MSP.n_samples[t],
                MSP.n_states[t]))
            for k, model in enumerate(M):
                if MSP.n_Markov_states != 1:
                    model._update_link_constrs(forward_solution[t-1])
                model.update()
                m = model.relax() if model.isMIP else model
                objLPScen[k], gradLPScen[k] = m._solveLP()
                # SB and LG share the same model
                if (
                    "SB" in self.cut_type_list[t-1]
                    or "LG" in self.cut_type_list[t-1]
                ):
                    m = model.copy()
                    m._delete_link_constrs()
                if "SB" in self.cut_type_list[t-1]:
                    objSBScen[k] = m._solveSB(gradLPScen[k])
                if "LG" in self.cut_type_list[t-1]:
                    objVal_primal = model._solvePrimal()
                    flag_bin = (
                        True if hasattr(self, "n_binaries")
                        else False
                    )
                    objLGScen[k], gradLGScen[k] = m._solveLG(
                        gradLPScen=gradLPScen[k],
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
                objLP, gradLP = self._compute_cuts(t, m, objLPScen, gradLPScen)
                objLP -= numpy.matmul(gradLP, forward_solution[t-1])
                self._add_and_store_cuts(t, objLP, gradLP, cuts, "B", j)
                self._add_cuts_additional_procedure(t, objLP, gradLP, objLPScen,
                    gradLPScen, forward_solution[t-1], cuts, "B", j)
            if "SB" in self.cut_type_list[t-1]:
                objSB, gradLP = self._compute_cuts(t, m, objSBScen, gradLPScen)
                self._add_and_store_cuts(t, objSB, gradLP, cuts, "SB", j)
                self._add_cuts_additional_procedure(t, objSB, gradLP, objSBScen,
                    gradLPScen, forward_solution[t-1], cuts, "SB", j)
            if "LG" in self.cut_type_list[t-1]:
                objLG, gradLG = self._compute_cuts(t, m, objLGScen, gradLGScen)
                self._add_and_store_cuts(t, objLG, gradLG, cuts, "LG", j)
                self._add_cuts_additional_procedure(t, objLG, gradLG, objLGScen,
                    gradLGScen, forward_solution[t-1], cuts, "LG", j)
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
        cut_type_list = [None] * self.cut_T
        cut_type_by_iteration = self._compute_cut_type_by_iteration()
        for t in range(1, self.cut_T+1):
            cut_type_list[t-1] = self._compute_cut_type_by_stage(
                t, cut_type_by_iteration)
        self.cut_type_list = cut_type_list

class PSDDP(SDDP):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_T = self.MSP.T
        self.cut_T = self.MSP.T
        self.period = self.MSP.T-1
        self.cut_type_list = [["B"] for t in range(self.cut_T)]

    def solve(self, forward_T=None, *args, **kwargs):
        """Solve the discretized problem.

        Parameters
        ----------
        forward_T: int, optional
            The number of stages to consider in forward passes and trial point
            selection
        """
        if forward_T: self.forward_T = forward_T
        self.MSP._set_up_CTG_for_t(-1)
        self.MSP._flag_infinity = 1
        super().solve(*args, **kwargs)


    def _add_cuts_additional_procedure(
        self, t, rhs, grad, objScen, gradScen, fwdSoln, cuts=None, cut_type=None,
        j=None
    ):
        if t != 1: return
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            MSP.models[-1]._add_cut(rhs, grad)
            if cuts is not None:
                cuts[MSP.T-1][cut_type][j][:] = numpy.append(rhs, grad)
        else:
            objScen = objScen.reshape(
                MSP.n_Markov_states[1]*MSP.n_samples[1])
            gradScen = gradScen.reshape(
                MSP.n_Markov_states[1]*MSP.n_samples[1],MSP.n_states[1])
            probability_ind = numpy.array([
                m.probability if m.probability
                else numpy.ones(m.n_samples)/m.n_samples
                for m in MSP.models[1]
            ])
            probability = numpy.einsum('ij,jk->ijk',MSP.transition_matrix[-1],
                probability_ind)
            probability = probability.reshape(MSP.n_Markov_states[-1],
                MSP.n_Markov_states[1]*MSP.n_samples[1])
            for k,m in enumerate(MSP.models[-1]):
                rhs_, grad_ = m._average(objScen, gradScen, probability[k])
                if cut_type == 'B':
                    rhs_ -= numpy.matmul(grad_, fwdSoln)
                m._add_cut(rhs_, grad_)
                if cuts is not None:
                    cuts[MSP.T-1][cut_type][j][k][:] = numpy.append(rhs_, grad_)

    def _add_cut_from_multiprocessing_array_additional_procedure(self, cuts):
        for cut_type in self.cut_type_list[0]:
            for cut in cuts[0][cut_type]:
                if self.MSP.n_Markov_states == 1:
                    self.MSP.models[-1]._add_cut(
                        rhs=cut[0],
                        gradient=cut[1:]
                    )
                else:
                    for k,cut_k in enumerate(cut):
                        self.MSP.models[-1][k]._add_cut(
                            rhs=cut_k[0],
                            gradient=cut_k[1:]
                        )

    def _compute_idx(self, t):
        idx = t%self.period if (t%self.period != 0 or t == 0) else self.period
        # transition matrix at stage period+1, 2*period+1, ... should be
        # additionaly specified.
        tm_idx = idx if (t%self.period != 1 or t == 1) else self.period+1
        return (idx,tm_idx)

    def _select_trial_solution(self, random_state, forward_solution):
        # if solving more than one single period, only part of obtained solutions
        # would be selected
        if self.forward_T > self.period + 1:
            indices = numpy.arange(0, self.forward_T, self.period)
            idx = indices[rand_int(k=len(indices), random_state=random_state)]
            if idx + self.period > self.forward_T:
                idx = idx - self.period
            for t in range(1, self.period+1):
                self.MSP.models[t]._update_link_constrs(forward_solution[idx+t-1])
            return forward_solution[idx:idx+self.period]
        return forward_solution


class PSDDiP(PSDDP, SDDiP):
    pass


class Extensive(object):
    """Extensive solver class. Can solve
    1. small-scale stgage-wise independent finite discrete risk netural problem 2.
    small-scale Markov chain risk neutral problem.

    Parameters
    ----------
    MSP: list
        A multi-stage stochastic program object.

    Attributes
    ----------
    extensive_model:
        The constructed extensive model

    solving_time:
        The time cost in solving extensive model

    construction_time:
        The time cost in constructing extensive model
    """

    def __init__(self, MSP):
        self.MSP = MSP
        self.solving_time = None
        self.construction_time = None
        self.total_time = None
        self._type = MSP._type

    def __getattr__(self, name):
        try:
            return getattr(self.extensive_model, name)
        except AttributeError:
            raise AttributeError("no attribute named {}".format(name))

    def solve(self, start=0, flag_rolling=0, **kwargs):
        """Call extensive solver to solve the discretized problem. It will first
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

        self.extensive_model = gurobipy.Model(env=self.MSP.env)
        self.extensive_model.modelsense = self.MSP.sense
        self.start = start

        for k, v in kwargs.items():
            setattr(self.extensive_model.Params, k, v)
        self._construct_extensive(flag_rolling)
        construction_end_time = time.time()
        self.construction_time = construction_end_time - construction_start_time
        solving_start_time = time.time()
        self.extensive_model.optimize()
        solving_end_time = time.time()
        self.solving_time = solving_end_time - solving_start_time
        self.total_time = self.construction_time + self.solving_time
        return self.extensive_model.objVal

    def _get_varname(self):
        if type(self.MSP.models[self.start]) != list:
            names = [var.varname for var in self.MSP.models[self.start].getVars()]
        else:
            names = [var.varname for var in self.MSP.models[self.start][0].getVars()]
        return names

    def _get_first_stage_vars(self):
        names = self._get_varname()
        if self._type not in ['Markovian', 'Markov chain']:
            vars = {name:self.extensive_model.getVarByName(name+'(0,)')
                for name in names}
        else:
            vars = {name:self.extensive_model.getVarByName(name+'((0,),(0,))')
                for name in names}
        return vars

    def _get_first_stage_states(self):
        names = self._get_varname()
        if self._type not in ['Markovian', 'Markov chain']:
            states = {name:self.extensive_model.getVarByName(name+'(0,)')
                for name in names}
        else:
            states = {name:self.extensive_model.getVarByName(name+'((0,),(0,))')
                for name in names}
        return states

    @property
    def first_stage_solution(self):
        """the obtained solution in the first stage"""
        states = self._get_first_stage_states()
        return {k:v.X for k,v in states.items()}

    @property
    def first_stage_all_solution(self):
        vars = self._get_first_stage_vars()
        return {k:v.X for k,v in vars.items()}

    @property
    def first_stage_cost(self):
        vars = self._get_first_stage_vars()
        return sum(v.obj*v.X for k,v in vars.items())

    def _construct_extensive(self, flag_rolling):
        ## Construct extensive model
        MSP = self.MSP
        T = MSP.T
        start = self.start
        n_Markov_states = MSP.n_Markov_states
        n_samples = (
            [MSP.models[t].n_samples for t in range(T)]
            if n_Markov_states == 1
            else [MSP.models[t][0].n_samples for t in range(T)]
        )
        n_states = MSP.n_states
        # check if CTG variable is added or not
        initial_model = (
            MSP.models[start] if n_Markov_states == 1 else MSP.models[start][0]
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
        for t in reversed(range(start,T)):
            M = [MSP.models[t]] if n_Markov_states == 1 else MSP.models[t]
            # stage T-1 needs to add the states. sample path corresponds to
            # current node.
            if t == T-1:
                _, sample_paths = MSP._enumerate_sample_paths(t,start,flag_rolling)
                states = [
                    self.extensive_model.addVars(sample_paths)
                    for _ in range(n_states[t])
                ]
            # new_states is the local_copies. new_sample_paths corresponds to
            # previous node
            if t != start:
                temp, new_sample_paths = MSP._enumerate_sample_paths(t-1,start,flag_rolling)
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
                            if item[0][t-start] == j and item[1][t-start] == k
                        ]
                        if n_Markov_states != 1
                        else [item for item in sample_paths if item[t-start] == j]
                    )
                    # when the sample path is too long, change the name of variables

                    controls_ = m.controls
                    states_ = m.states
                    local_copies_ = m.local_copies
                    controls_dict = {v: i for i, v in enumerate(controls_)}
                    states_dict = {v: i for i, v in enumerate(states_)}
                    local_copies_dict = {
                        v: i for i, v in enumerate(local_copies_)
                    }

                    for current_sample_path in current_sample_paths:
                        flag_reduced_name = 0
                        if len(str(current_sample_path)) > 100:
                            flag_reduced_name = 1
                        if t != start:
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

                        if flag_CTG == -1 or t == start:
                            weight = MSP.discount ** (
                                (t - start)
                            ) * MSP._compute_weight_sample_path(
                                current_sample_path, start
                            )
                        else:
                            currentWeight = MSP._compute_current_weight_sample_path(
                                current_sample_path)

                        for i in range(n_states[t]):
                            obj = (
                                states_[i].obj * numpy.array(weight)
                                if flag_CTG == -1 or t == start
                                else 0
                            )
                            states[i][current_sample_path].lb = states_[i].lb
                            states[i][current_sample_path].ub = states_[i].ub
                            states[i][current_sample_path].obj = obj
                            states[i][current_sample_path].vtype = states_[
                                i
                            ].vtype
                            if flag_reduced_name == 0:
                                states[i][current_sample_path].varName = states_[
                                    i
                                ].varName + str(current_sample_path).replace(
                                    " ", ""
                                )
                            # cost-to-go update
                            if t != start and flag_CTG == 1:
                                new_stage_cost[past_sample_path] += (
                                    states[i][current_sample_path]
                                    * states_[i].obj
                                    * currentWeight
                                )

                        if t == start:
                            for i in range(n_states[t]):
                                new_states[i][current_sample_path].lb = local_copies_[i].lb
                                new_states[i][current_sample_path].ub = local_copies_[i].ub
                                new_states[i][current_sample_path].obj = local_copies_[i].obj
                                new_states[i][current_sample_path].vtype = local_copies_[i].vtype
                                if flag_reduced_name == 0:
                                    new_states[i][current_sample_path].varName = local_copies_[i].varname + str(current_sample_path).replace(" ", "")
                        # copy local variables
                        controls = [None for _ in range(len(controls_))]
                        for i, var in enumerate(controls_):
                            obj = (
                                var.obj * weight
                                if flag_CTG == -1 or t == start
                                else 0
                            )
                            controls[i] = self.extensive_model.addVar(
                                lb=var.lb,
                                ub=var.ub,
                                obj=obj,
                                vtype=var.vtype,
                                name=(
                                    var.varname
                                    + str(current_sample_path).replace(" ", "")
                                    if flag_reduced_name == 0
                                    else ""
                                ),
                            )
                            # cost-to-go update
                            if t != start and flag_CTG == 1:
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
                                    if t != start:
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

class Rolling(object):

    def __init__(self, MSP):
        self.MSP = MSP

    def solve_single_process(self, a, jobs, query, query_stage_cost, solution,
            stage_cost, seed):
        MSP = self.MSP
        # random_state for simulations
        another_random_state = numpy.random.RandomState([2**32-1, jobs[0]])
        Markov_states = transition_matrix = n_samples = None
        if MSP._type == 'Markovian':
            markovian_samples = MSP.Markovian_uncertainty(another_random_state,
                len(jobs))
            Markov_states = [None for t in range(MSP.T)]
            transition_matrix = [None for t in range(MSP.T)]
        for i,j in enumerate(jobs):
            # random_state for discretization
            random_state = check_random_state(seed)
            for cur in range(MSP.T-1):
                if MSP._type == "Markovian":
                    Markov_states[cur] = markovian_samples[i][cur].reshape(1,-1)
                    Markov_states[cur+1] = numpy.array([
                        self.conditional_dist(
                            random_state=random_state,
                            prev=markovian_samples[i][cur],
                            t=cur+1,
                        )
                        for _ in range(self.n_branches)
                    ])
                    for t in range(cur+2,MSP.T):
                        Markov_states[t] = numpy.array([
                            self.conditional_dist(
                                random_state=random_state,
                                prev=Markov_states[t-1][k],
                                t=t,
                            )
                            for k in range(self.n_branches)
                        ])
                    transition_matrix[cur] = numpy.array([[1]])
                    transition_matrix[cur+1] = numpy.ones(
                        (1,self.n_branches))/self.n_branches
                    for t in range(cur+2,MSP.T):
                        transition_matrix[t] = numpy.eye(self.n_branches)
                if MSP[cur]._type == 'continuous':
                    MSP[cur]._sample_uncertainty(another_random_state)
                    MSP[cur]._flag_discrete = 1
                if MSP[cur+1]._type == 'continuous':
                    n_samples = [1] * MSP.T
                    n_samples[cur+1] = self.n_branches

                MSP.discretize(
                    n_samples=n_samples,
                    random_state=random_state,
                    replace=True,
                    method='input',
                    Markov_states=Markov_states,
                    transition_matrix=transition_matrix,
                    int_flag=0)
                ext = Extensive(MSP)
                ext.solve(outputFlag=0, start=cur, flag_rolling=1)
                if query is not None:
                    sol = ext.first_stage_all_solution
                    for k,v in sol.items():
                        if k in query:
                            solution[k][j][cur] = v
                if query_stage_cost:
                    stage_cost[j][cur] = ext.first_stage_cost
                a[j] += MSP.discount ** cur * ext.first_stage_cost
                MSP._reverse_discretize()
                MSP[cur]._delete_link_constrs()
                MSP[cur+1]._set_up_link_constrs()
                MSP[cur+1]._update_link_constrs(
                    [ext.first_stage_solution[v.varName] for v in MSP[cur+1].states])
            if MSP[-1]._type == 'continuous':
                MSP[-1]._sample_uncertainty(another_random_state)
            if MSP._type == 'Markovian':
                MSP[-1]._update_uncertainty_dependent(markovian_samples[i][-1])
            MSP[-1].optimize()
            MSP[-1]._delete_link_constrs()
            a[j] += MSP.discount ** (MSP.T-1) * MSP[-1].objVal
            for var in MSP[-1].getVars():
                if var.varname in query:
                    solution[var.varname][j][MSP.T-1] = var.X
            if query_stage_cost:
                stage_cost[j][MSP.T-1] = MSP[-1].objVal


    def solve(
            self,
            n_simulations,
            n_branches,
            conditional_dist=None,
            n_processes=1,
            percentile=95,
            query=None,
            query_stage_cost=False,
            random_state=None,):
        """Call rolling solver to solve the true problem. It will dynamically
        construct extensive models and then call Gurobi solver to solve it.

        Parameters
        ----------
        n_simulations: int
            The number of simulations;

        n_branches: int
            The number of branches in the next stage

        conditional_dist: callable (default=None)
            The conditional distribution. If the true problem is Markovian, it
            must be specified. It must take random_state, prev, t as arguments
            and returns an array-like with the same length of prev.

        n_processes: int, optional (default=1)
            The number of processes to run the algorithm.

        percentile: float, optional (default=95)
            The percentile used to compute the confidence interval.

        query: list, optional (default=None)
            The names of variables that are intended to query.

        query_stage_cost: bool, optional (default=False)
            Whether to query values of individual stage costs.

        random_state: None | int | instance of RandomState, optional, default=None
            If int, random_state is the seed used by the
            random number generator;
            If RandomState instance, random_state is the
            random number generator;
            If None, the random number generator is the
            RandomState instance used by numpy.random.
        """
        self.n_simulations = n_simulations
        self.n_branches = n_branches
        if self.MSP._type == 'Markovian':
            if conditional_dist is None:
                raise Exception("Conditional distribution must be given for "
                    +"Markovian problem!")
        self.conditional_dist = conditional_dist
        a = multiprocessing.Array("d", [0] * n_simulations)
        procs = [None] * n_processes
        jobs = allocate_jobs(n_simulations, n_processes)
        query = query if query is not None else []
        solution = stage_cost = None
        if query is not None:
            solution = {
                item: [
                    multiprocessing.RawArray("d",[0] * (self.MSP.T))
                    for _ in range(n_simulations)
                ]
                for item in query
            }
        if query_stage_cost:
            stage_cost = [
                multiprocessing.RawArray("d",[0] * (self.MSP.T))
                for _ in range(n_simulations)
            ]
        for p in range(n_processes):
            procs[p] = multiprocessing.Process(
                target=self.solve_single_process,
                args=(a,jobs[p],query,query_stage_cost,solution,stage_cost,
                    random_state)
            )
            procs[p].start()
        for proc in procs:
            proc.join()
        if query is not None:
            self.solution = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution.items()
            }
        if query_stage_cost:
            self.stage_cost = pandas.DataFrame(numpy.array(stage_cost))
        self.pv = [item for item in a]
        if self.n_simulations != 1:
            self.CI = compute_CI(self.pv, percentile)
