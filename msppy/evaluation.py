from msppy.utils.statistics import (rand_int,check_random_state,compute_CI,
allocate_jobs)
import pandas
import time
import numpy
import multiprocessing


class _Evaluation(object):
    """Evaluation base class.

    Parameters
    ----------
    MSP: list
        A multi-stage stochastic program object.

    Attributes
    ----------
    db: list
        The deterministic bounds.

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

    n_sample_paths: int
        The number of sample paths to evaluate policy.

    sample_paths_idx: list
        The index list of exhaustive sample paths if simulation is turned off.

    markovian_samples:
        The simulated Markovian type samples.

    markovian_idx: list
        The Markov state that is the closest to the markovian_samples.
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
        self.n_sample_paths = None
        self.sample_path_idx = None
        self.markovian_idx = None
        self.markovian_samples = None
        self.solve_true = False

    def _compute_gap(self):
        if self.MSP.measure != 'risk neutral':
            self.gap = -1
            return
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
            self.gap = -1

    def _compute_sample_path_idx_and_markovian_path(self):
        pass

    def run(
            self,
            n_simulations,
            percentile=95,
            query=None,
            query_dual=None,
            query_stage_cost=False,
            T=None,
            n_processes=1,):
        """Run a Monte Carlo simulation to evaluate the policy.

        Parameters
        ----------
        n_simulations: int/-1
            If int: the number of simulations;
            If -1: exhuastive evaluation.

        percentile: float, optional (default=95)
            The percentile used to compute the confidence interval.

        query: list, optional (default=None)
            The names of variables that are intended to query.

        query_dual: list, optional (default=None)
            The names of constraints whose dual variables are intended to query.

        query_stage_cost: bool, optional (default=False)
            Whether to query values of individual stage costs.

        n_processes: int, optional (default=1)
            The number of processes to run the simulation.

        T: int, optional (default=None)
            For infinite horizon problem, how many stages to evaluate the policy.
        """

        from msppy.solver import SDDP, SDDP_infinity
        MSP = self.MSP
        # overwrite original specified number of stages
        if MSP.infinity and T:
            T_original = MSP.T
            MSP.T = T
            modified_horizon = True
        else:
            T = MSP.T
            modified_horizon = False

        if not MSP.infinity:
            self.solver = SDDP(MSP)
        else:
            self.solver = SDDP_infinity(MSP)
        self.n_simulations = n_simulations
        query_stage_cost = query_stage_cost
        self._compute_sample_path_idx_and_markovian_path()
        self.pv = numpy.zeros(self.n_sample_paths)
        stage_cost = solution = solution_dual = None
        if query_stage_cost:
            stage_cost = [
                multiprocessing.RawArray("d",[0] * (T))
                for _ in range(self.n_sample_paths)
            ]
        if query is not None:
            solution = {
                item: [
                    multiprocessing.RawArray("d",[0] * (T))
                    for _ in range(self.n_sample_paths)
                ]
                for item in query
            }
        if query_dual is not None:
            solution_dual = {
                item: [
                    multiprocessing.RawArray("d",[0] * (T))
                    for _ in range(self.n_sample_paths)
                ]
                for item in query_dual
            }
        jobs = allocate_jobs(self.n_sample_paths, n_processes)
        pv = multiprocessing.Array("d", [0] * self.n_sample_paths)
        procs = [None] * n_processes
        for p in range(n_processes):
            procs[p] = multiprocessing.Process(
                target=self.run_single,
                args=(pv,jobs[p],query,query_dual,query_stage_cost,stage_cost,
                    solution,solution_dual)
            )
            procs[p].start()
        for proc in procs:
            proc.join()
        self.pv = [item for item in pv]
        if self.n_simulations == -1:
            self.epv = numpy.dot(
                pv,
                [
                    MSP._compute_weight_sample_path(self.sample_path_idx[j])
                    for j in range(self.n_sample_paths)
                ],
            )
        if self.n_simulations not in [-1,1]:
            self.CI = compute_CI(self.pv, percentile)
        self._compute_gap()
        if query is not None:
            self.solution = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution.items()
            }
        if query_dual is not None:
            self.solution_dual = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution_dual.items()
            }
        if query_stage_cost:
            self.stage_cost = pandas.DataFrame(numpy.array(stage_cost))
        # recover original specified number of stages
        if MSP.infinity and modified_horizon:
            MSP.T = T_original

    def run_single(self, pv, jobs, query=None, query_dual=None,
            query_stage_cost=False, stage_cost=None,
            solution=None, solution_dual=None):
        random_state = numpy.random.RandomState([2**32-1, jobs[0]])
        for j in jobs:
            sample_path_idx = (self.sample_path_idx[j]
                if self.sample_path_idx is not None else None)
            markovian_idx = (self.markovian_idx[j]
                if self.markovian_idx is not None else None)
            markovian_samples = (self.markovian_samples[j]
                if self.markovian_samples is not None else None)
            result = self.solver._forward(
                random_state=random_state,
                sample_path_idx=sample_path_idx,
                markovian_idx=markovian_idx,
                markovian_samples=markovian_samples,
                solve_true=self.solve_true,
                query=query,
                query_dual=query_dual,
                query_stage_cost=query_stage_cost
            )
            if query is not None:
                for item in query:
                    for i in range(len(solution[item][0])):
                        solution[item][j][i] = result['solution'][item][i]
            if query_dual is not None:
                for item in solution_dual:
                    for i in range(len(solution_dual[item][0])):
                        solution_dual[item][j][i] = result['solution_dual'][item][i]
            if query_stage_cost:
                for i in range(len(stage_cost[0])):
                    stage_cost[j][i] = result['stage_cost'][i]
            pv[j] = result['pv']

class Evaluation(_Evaluation):

    __doc__ = _Evaluation.__doc__
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)

    def _compute_sample_path_idx_and_markovian_path(self):
        if self.n_simulations == -1:
            self.n_sample_paths,self.sample_path_idx = self.MSP._enumerate_sample_paths(self.MSP.T-1)
        else:
            self.n_sample_paths = self.n_simulations


class EvaluationTrue(Evaluation):

    __doc__ = Evaluation.__doc__
    def run(self, *args, **kwargs):
        MSP = self.MSP
        if MSP.__class__.__name__ == 'MSIP':
            MSP._back_binarize()
        # discrete finite model should call evaluate instead
        if (
            MSP._type in ["stage-wise independent", "Markov chain"]
            and MSP._individual_type == "original"
            and not hasattr(MSP,"bin_stage")
        ):
            return super().run(*args, **kwargs)
        return _Evaluation.run(self, *args, **kwargs)

    def _compute_sample_path_idx_and_markovian_path(self):
        MSP = self.MSP
        if (
            MSP._type in ["stage-wise independent", "Markov chain"]
            and MSP._individual_type == "original"
            and not hasattr(MSP,"bin_stage")
        ):
            return super()._compute_sample_path_idx_and_markovian_path()
        self.n_sample_paths = self.n_simulations
        self.solve_true = True
        if MSP._type == "Markovian":
            self.markovian_samples = MSP.Markovian_uncertainty(
                numpy.random.RandomState(2**32-1),self.n_simulations)
            self.markovian_idx = numpy.zeros([self.n_simulations,MSP.T],dtype=int)
            for t in range(1,MSP.T):
                dist = numpy.empty([self.n_simulations,MSP.n_Markov_states[t]])
                for idx, markov_state in enumerate(MSP.Markov_states[t]):
                    temp = self.markovian_samples[:,t,:] - markov_state
                    dist[:,idx] = numpy.sum(temp**2, axis=1)
                self.markovian_idx[:,t] = numpy.argmin(dist,axis=1)
