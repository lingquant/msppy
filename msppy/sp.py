import gurobipy
import numpy
from msppy.utils.statistics import rand_int,check_random_state
from msppy.utils.exception import SampleSizeError,DistributionError
from msppy.utils.measure import Expectation
import msppy.utils.copy as deepcopy
from collections import abc
from numbers import Number
import time
import math


class StochasticModel(object):
    """The StochasticModel class"""
    def __init__(self, name="", env=None):
        genv = env if env is not None else gurobipy.Env()
        self.env = genv
        self._model = gurobipy.Model(env=genv, name=name)
        # each and every instance must have state variables, local copy variables
        self.states = []
        self.local_copies = []
        # (discretized) uncertainties
        # stage-wise independent discrete uncertainties
        self.uncertainty_rhs = {}
        self.uncertainty_coef = {}
        self.uncertainty_obj = {}
        # indices of stage-dependent uncertainties
        self.uncertainty_rhs_dependent = {}
        self.uncertainty_coef_dependent = {}
        self.uncertainty_obj_dependent = {}
        # true uncertainties
        # stage-wise independent true continuous uncertainties
        self.uncertainty_rhs_continuous = {}
        self.uncertainty_coef_continuous = {}
        self.uncertainty_obj_continuous = {}
        self.uncertainty_mix_continuous = {}
        # stage-wise independent true discrete uncertainties
        self.uncertainty_rhs_discrete = {}
        self.uncertainty_coef_discrete = {}
        self.uncertainty_obj_discrete = {}
        # cutting planes approximation of recourse variable alpha
        self.alpha = None
        self.cuts = []
        # linking constraints
        self.link_constrs = []
        # number of discrete uncertainties
        self.n_samples = 1
        # number of state varibles
        self.n_states = 0
        # probability measure for discrete uncertainties
        self.probability = None
        # type of true problem: continuous/discrete
        self._type = None
        # flag to indicate discretization of true problem
        self._flag_discrete = 0
        # collection of all specified dim indices of Markovian uncertainties
        self.Markovian_dim_index = []
        # risk measure
        self.measure = Expectation

    def __getattr__(self, name):
        try:
            return getattr(self._model, name)
        except AttributeError:
            raise AttributeError("no attribute named {}".format(name))

    # representation of true problem
    def __repr__(self):
        uncertainty = ""
        if self._type == "discrete":
            uncertainty_rhs = (
                ""
                if self.uncertainty_rhs == {}
                else "discrete uncertainties on the RHS of constraints, "
            )
            uncertainty_coef = (
                ""
                if self.uncertainty_coef == {}
                else "discrete uncertainties on the coefficients of constraints, "
            )
            uncertainty_obj = (
                ""
                if self.uncertainty_obj == {}
                else "discrete uncertainties in the objective, "
            )
            uncertainty = uncertainty_rhs + uncertainty_coef + uncertainty_obj
        elif self._type == "continuous":
            uncertainty_rhs_continuous = (
                ""
                if self.uncertainty_rhs_continuous == {}
                else "continuous uncertainties on the RHS of constraints, "
            )
            uncertainty_coef_continuous = (
                ""
                if self.uncertainty_coef_continuous == {}
                else "continuous uncertainties on the coefficients of constraints, "
            )
            uncertainty_obj_continuous = (
                ""
                if self.uncertainty_obj_continuous == {}
                else "continuous uncertainties in the objective, "
            )
            uncertainty_mix_continuous = (
                ""
                if self.uncertainty_mix_continuous == {}
                else "continuous uncertainties in multiple locations, "
            )
            uncertainty = (
                uncertainty_rhs_continuous
                + uncertainty_coef_continuous
                + uncertainty_obj_continuous
                + uncertainty_mix_continuous
            )
        comma = "" if uncertainty == "" else ", "
        return (
            "<Stochastic "
            + repr(self._model)[1:-1]
            + ", {} state variables, {} samples".format(
                self.n_states, self.n_samples
            )
            + comma
            + uncertainty
            + ">"
        )

    def _copy(self, model):
        cls = self.__class__
        result = cls.__new__(cls)
        # copy the internal Gurobi model
        result._model = model.copy()
        for attribute, value in self.__dict__.items():
            if attribute == "_model":
                pass
            else:
                # copy all attributes that have not been assigned a value
                setattr(result, attribute, None)
                dict = {'value': value, 'target': result, 'attribute': attribute}
                # copy all uncertainties
                if attribute.startswith("uncertainty"):
                    setattr(result, attribute, {})
                    if attribute.startswith("uncertainty_rhs"):
                        deepcopy._copy_uncertainty_rhs(**dict)
                    elif attribute.startswith("uncertainty_coef"):
                        deepcopy._copy_uncertainty_coef(**dict)
                    elif attribute.startswith("uncertainty_obj"):
                        deepcopy._copy_uncertainty_obj(**dict)
                    elif attribute.startswith("uncertainty_mix"):
                        deepcopy._copy_uncertainty_mix(**dict)
                    else:
                        raise Exception("alien uncertainties added!")
                # copy all variables
                elif attribute in ["states", "local_copies", "alpha"]:
                    deepcopy._copy_vars(**dict)
                # copy all constraints
                elif attribute in ["cuts", "link_constrs"]:
                    deepcopy._copy_constrs(**dict)
                # copy probability measure
                elif attribute == "probability":
                    result.probability = None if value is None else list(value)
                # copy other numerical stuff
                else:
                    setattr(result, attribute, value)
        return result

    def _check_uncertainty(self, uncertainty, flag_dict, list_dim):
        """Make sure the input uncertainty is in the correct form. Return a
        copied uncertainty to avoid making changes to mutable object given by
        the users.

        Check data structure
        --------------------
        In discrete case:

        Uncertainty added by addConstr must be a dictionary. Value of the
        dictionary must be unidimensional list (flag_dict=1, list_dim=1).

        Uncertainty added by addVar must be a array-like (flag_dict=0, list_dim=1).

        Uncertainty added by addConstrs and addVars must be a multidimensional
        array-like (flag_dict=0, list_dim>1)
        The multidimensional array-like has the shape (a,b), where a should be
        the dimension of the object added indicated by list_dim (>1) and b
        should be the number of samples.

        In continuous case:

        Uncertainty added by addConstr must be a dictionary. Value of the
        dictionary must be a callable that generates a single number
        (flag_dict=1, list_dim=1).

        Uncertainty added by addVar must be a callable that generates a single
        number (flag_dict=0, list_dim=1).

        Uncertainty added by addConstrs and addVars must be a callable that
        generates an array-like (flag_dict=0, list_dim>1)
        The generated array-like has the shape (a,b), where a should the
        dimension of the object added indicated by list_dim (>1) and b should be
        the number of samples.

        All callable should take numpy RandomState as its only argument.

        Check type of uncertainty
        -------------------------
        The true problem must be either continuous or discrete. Hence, once a
        continuous uncertainty has been added, discrete uncertainty is no longer
        accepted, vice versa.
        """
        if isinstance(uncertainty, abc.Mapping):
            uncertainty = dict(uncertainty)
            for key, item in uncertainty.items():
                if callable(item):
                    if not self._type:
                        # add uncertainty for the first time
                        self._type = "continuous"
                    else:
                        # already added uncertainty
                        if self._type != "continuous":
                            raise SampleSizeError(
                                self._model.modelName,
                                self.n_samples,
                                uncertainty,
                                "infinite"
                            )
                    try:
                        item(numpy.random)
                    except TypeError:
                        raise DistributionError(arg=False)
                    try:
                        float(item(numpy.random))
                    except (ValueError,TypeError):
                        raise DistributionError(ret=False)
                else:
                    try:
                        item = numpy.array(item, dtype='float64')
                    except ValueError:
                        raise ValueError("Scenarios must only contains numbers!")
                    if item.ndim != 1:
                        raise ValueError(
                            "dimension of the distribution is {} while \
                            dimension of the added object is {}!"
                            .format(item.ndim, 1)
                        )
                    uncertainty[key] = list(item)

                    if not self._type:
                        # add uncertainty for the first time
                        self._type = "discrete"
                        self.n_samples = len(item)
                    else:
                        # already added uncertainty
                        if self._type != "discrete":
                            raise SampleSizeError(
                                self._model.modelName,
                                "infinite",
                                {key:item},
                                len(item)
                            )
                        if self.n_samples != len(item):
                            raise SampleSizeError(
                                self._model.modelName,
                                self.n_samples,
                                {key:item},
                                len(item)
                            )
            if flag_dict == 0:
                raise TypeError("wrong uncertainty format!")
        elif isinstance(uncertainty, abc.Callable):
            try:
                sample = uncertainty(numpy.random)
            except TypeError:
                raise DistributionError(arg=False)
            if list_dim == 1:
                try:
                    float(sample)
                except (ValueError,TypeError):
                    raise DistributionError(ret=False)
            else:
                try:
                    sample = [float(item) for item in sample]
                except (ValueError,TypeError):
                    raise DistributionError(ret=False)
                if list_dim != len(uncertainty(numpy.random)):
                    raise ValueError(
                        "dimension of the distribution is {} while \
                        dimension of the added object is {}!"
                        .format(len(uncertainty(numpy.random)), list_dim)
                    )
            if not self._type:
                # add uncertainty for the first time
                self._type = "continuous"
            else:
                # already added uncertainty
                if self._type != "continuous":
                    raise SampleSizeError(
                        self._model.modelName,
                        self.n_samples,
                        uncertainty,
                        "infinite"
                    )
        elif isinstance(uncertainty, (abc.Sequence, numpy.ndarray)):
            uncertainty = numpy.array(uncertainty)
            if list_dim == 1:
                if uncertainty.ndim != 1:
                    raise ValueError("dimension of the scenarios is {} while \
                                     dimension of the added object is 1!"
                        .format(uncertainty.ndim)
                    )
                try:
                    uncertainty = [float(item) for item in uncertainty]
                except ValueError:
                    raise ValueError("Scenarios must only contains numbers!")
            else:
                # list to list
                if uncertainty.ndim != 2 or uncertainty.shape[1] != list_dim:
                    dim = None if uncertainty.ndim == 1 else uncertainty.shape[1]
                    raise ValueError("dimension of the scenarios is {} while \
                                     dimension of the added object is 1!"
                        .format(dim, uncertainty.ndim)
                    )
                try:
                    uncertainty = numpy.array(uncertainty, dtype='float64')
                except ValueError:
                    raise ValueError("Scenarios must only contains numbers!")
                uncertainty = [list(item) for item in uncertainty]
            if not self._type:
                self._type = "discrete"
                self.n_samples = len(uncertainty)
            else:
                if self._type != "discrete":
                    raise SampleSizeError(
                        self._model.modelName,
                        "infinite",
                        uncertainty,
                        len(uncertainty)
                    )
                if self.n_samples != len(uncertainty):
                    raise SampleSizeError(
                        self._model.modelName,
                        self.n_samples,
                        uncertainty,
                        len(uncertainty)
                    )
        else:
            raise TypeError("wrong uncertainty format!")

        return uncertainty

    def _check_uncertainty_dependent(
        self, uncertainty_dependent, flag_dict, list_dim
    ):
        """Make sure the input uncertainty location index is in the correct form.
        Return a copied uncertainty to avoid making changes to mutable object
        given by the users.

        Check data structure
        --------------------

        Uncertainty added by addConstr must be a dictionary. Value of the
        dictionary must be an int (flag_dict=1, list_dim=1).

        Uncertainty added by addVar must be an int (flag_dict=0, list_dim=1).

        Uncertainty added by addConstrs and addVars must be a array-like of int
        array-like (flag_dict=0, list_dim>1). The length of the array-like
        should equal list_dim.
        """
        if isinstance(uncertainty_dependent, abc.Mapping):
            if flag_dict == 0:
                raise TypeError("wrong uncertainty_dependent format!")
            for key, item in uncertainty_dependent.items():
                try:
                    item = int(item)
                    uncertainty_dependent[key] = item
                except (TypeError,ValueError):
                    raise ValueError("location index of individual component \
                                     of uncertainty_dependent must be integer!")
                self.Markovian_dim_index.append(item)

        elif isinstance(uncertainty_dependent, (abc.Sequence, numpy.ndarray)):
            uncertainty_dependent = list(uncertainty_dependent)
            if len(uncertainty_dependent) != list_dim:
                raise ValueError(
                    "dimension of the scenario is {} while \
                    dimension of added object is {}!"
                    .format(len(uncertainty_dependent), list_dim)
                )
            self.Markovian_dim_index += uncertainty_dependent

        elif isinstance(uncertainty_dependent, Number):
            uncertainty_dependent = int(uncertainty_dependent)
            if list_dim != 1:
                raise ValueError(
                    "dimension of the scenario is 1 while \
                    dimension of added object is {}!"
                    .format(list_dim)
                )
            self.Markovian_dim_index.append(uncertainty_dependent)
        else:
            raise TypeError("wrong uncertainty_dependent format")
        return uncertainty_dependent

    def copy(self):
        """
        Create a deepcopy of a stochastic model.

        Returns
        -------
        The copied StochasticModel object.
        """
        return self._copy(self._model)

    def relax(self):
        """
        Return the relaxed version of the stochastic MIP model.

        Returns
        -------
        a new relaxed StochasticModel object

        Notes
        -----
        If the model is already continuous, then this method produces the
        same result as cloning the model.
        """
        return self._copy(self._model.relax())

    def addStateVars(
            self,
            *indices,
            lb=0.0,
            ub=1e+100,
            obj=0.0,
            vtype='C',
            name="",
            uncertainty=None,
            uncertainty_dependent=None
    ):
        """
        Add state variables in bulk. Generalize gurobipy.addVars() to
        incorporate uncertainty in the objective function. Variables are added
        as state variables and the corresponding local copy variables will be
        added behind the scene

        Parameters
        ----------

        uncertainty: array-like, optional, default=None,
            The scenarios of stage-wise independent uncertain objective
            coefficients.

        uncertainty: callable, optional, default=None
            The multivariate random variable generator of stage-wise
            independent uncertain objective coefficients. It must take
            numpy RandomState as its only argument.

        uncertainty_dependent: array-like, optional, default=None
            The location index in the stochastic process generator of
            stage-wise dependent uncertain objective coefficients.

        Returns
        -------
        (the created state variables, the corresponding local_copy variables): tuple

        Examples
        --------
        >>> now,past = model.addStateVars(
        ...     2,
        ...     ub=2.0,
        ...     uncertainty={[[2,4],[3,5]]}
        ... )
        >>> now,past = model.addStateVars(
        ...     [(1,2),(2,1)],
        ...     ub=2.0,
        ...     uncertainty={[[2,4],[3,5]]}
        ... )

        stage-wise independent continuous uncertain objective coefficients

        >>> def f(random_state):
        ...     return random_state.multivariate_normal(
        ...         mean = [0,0],
        ...         cov = [[1,0],[0,1]]
        ...     )
        >>> now,past = model.addStateVars(2, ub=2.0, uncertainty=f)

        Markovian objective coefficients

        >>> now,past = model.addStateVars(2, ub=2.0, uncertainty_dependent=[1,2])
        """
        state = self._model.addVars(
            *indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name
        )
        local_copy = self._model.addVars(
            *indices, lb=lb, ub=ub, name=name + "_local_copy"
        )
        self._model.update()
        self.states += state.values()
        self.local_copies += local_copy.values()
        self.n_states += len(state)

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, 0, len(state))
            if callable(uncertainty):
                self.uncertainty_obj_continuous[
                    tuple(state.values())
                ] = uncertainty
            else:
                self.uncertainty_obj[tuple(state.values())] = uncertainty


        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, 0, len(state)
            )
            self.uncertainty_obj_dependent[
                tuple(state.values())
            ] = uncertainty_dependent

        return state, local_copy

    def addStateVar(
            self,
            lb=0.0,
            ub=1e+100,
            obj=0.0,
            vtype='C',
            name="",
            column=None,
            uncertainty=None,
            uncertainty_dependent=None,
    ):
        """
        Add a state variable to the model. Generalize gurobipy.addVar() to
        incorporate uncertainty in the objective function. The variable is added
        as a state variable and the corresponding local copy variable will be
        added behind the scene

        Parameters
        ----------
        uncertainty: array-like, optional, default=None
            The scenarios of the stage-wise independent uncertain
            objective coefficient

        uncertainty: callable, optional, default=None
            The univariate random variable generator of stage-wise
            independent uncertain objective coefficient. The callable must
            take numpy.random.

        uncertainty: int, optional, default=None
            The location index in the stochastic process generator of the
            stage-wise dependent uncertain objective coefficient

        Returns
        -------
        (the created state variable, the corresponding local copy variable): tuple

        Examples
        --------
        >>> now,past = model.addStateVar(ub=2.0, uncertainty=[1,2,3])

        stage-wise independent continuous uncertain objective coefficient

        >>> def f(random_state):
        ...     return random_state.normal(0, 1)
        >>> now,past = model.addStateVar(ub=2.0, uncertainty=f)

        Markovian objective coefficient

        >>> now,past = model.addStateVar(ub=2.0, uncertainty_dependent=[1,2])
        """
        state = self._model.addVar(
            lb=lb, ub=ub, obj=obj, vtype=vtype, name=name, column=column,
        )
        local_copy = self._model.addVar(
            name=name+"_local_copy", lb=lb, ub=ub,
        )
        self._model.update()
        self.states += [state]
        self.local_copies += [local_copy]
        self.n_states += 1

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, 0, 1)
            if callable(uncertainty):
                self.uncertainty_obj_continuous[state] = uncertainty
            else:
                self.uncertainty_obj[state] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, 0, 1
            )
            self.uncertainty_obj_dependent[state] = uncertainty_dependent

        return state, local_copy

    def addVars(
            self,
            *indices,
            lb=0.0,
            ub=1e+100,
            obj=0.0,
            vtype='C',
            name="",
            uncertainty=None,
            uncertainty_dependent=None
    ):
        """
        Add variables in bulk. Generalize gurobipy.addVars() to
        incorporate uncertainty in the objective function

        Parameters
        ----------
        uncertainty: array-like, optional, default=None
            The scenarios of the stage-wise independent uncertain
            objective coefficients

        uncertainty: callable, optional, default=None
            The multivariate random variable generator of stage-wise
            independent uncertain objective coefficients. The callable must
            take numpy RandomState as its only argument

        uncertainty_dependent: array-like, optional, default=None
            The locations index in the stochastic process generator of the
            stage-wise dependent objective coefficients

        Returns
        -------
        The created variables: list of gurobipy.Var

        Examples
        --------
        >>> newVars = model.addVars(
        ...     3,
        ...     ub=2.0,
        ...     uncertainty={[[2,4,6],[3,5,7]]}
        ... )
        >>> newVars = model.addVars(
        ...     [(1,2),(2,1)],
        ...     ub=2.0,
        ...     uncertainty={[[2,4],[3,5],[4,6]]}
        ... )

        stage-wise independent continuous uncertain objective coefficients

        >>> def f(random_state):
        ...     return random_state.multivariate_normal(
        ...         mean = [0,0],
        ...         cov = [[1,0],[0,100]]
        ...     )
        >>> newVars = model.addVars(
        ...     2,
        ...     ub=2.0,
        ...     uncertainty=f
        ... )

        Markovian objective coefficients

        >>> newVars = model.addVars(
        ...     2,
        ...     ub=2.0,
        ...     uncertainty_dependent=[1,2]
        ... )
        """
        var = self._model.addVars(
            *indices, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name
        )
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, 0, len(var))
            if callable(uncertainty):
                self.uncertainty_obj_continuous[
                    tuple(var.values())
                ] = uncertainty
            else:
                self.uncertainty_obj[tuple(var.values())] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, 0, len(var)
            )
            self.uncertainty_obj_dependent[
                tuple(var.values())
            ] = uncertainty_dependent

        return var

    def addVar(
            self,
            lb=0.0,
            ub=1e+100,
            obj=0.0,
            vtype='C',
            name="",
            column=None,
            uncertainty=None,
            uncertainty_dependent=None,
    ):
        """
        Add a variable to the model. Generalize gurobipy.addVar() to
        incorporate uncertainty in the objective function

        Parameters
        ----------
        uncertainty: array-like, optional, default=None
            The scenarios of the stage-wise independent uncertain
            objective coefficient

        uncertainty: callable, optional, default=None
            The univariate random variable generator of stage-wise independent
            uncertain objective coefficient. The callable must take numpy
            RandomState as its only argument.

        uncertainty: int, optional, default=None
            The location index in the sample path generator of the stage-wise
            dependent uncertain objective coefficient

        Returns
        -------
        The created variable: gurobipy.Var

        Examples
        --------
        >>> newVar = model.addVar(ub=2.0, uncertainty=[1,2,3])

        stage-wise independent continuous uncertain objective coefficient

        >>> def f(random_state):
        ...     return random_state.normal(0, 1)
        ... newVar = model.addVar(ub=2.0, uncertainty=f)

        Markovian objective coefficient

        >>> newVar = model.addVar(ub=2.0, uncertainty_dependent=[1])
        """
        var = self._model.addVar(
            lb=lb, ub=ub, obj=obj, vtype=vtype, name=name, column=column
        )
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, 0, 1)
            if callable(uncertainty):
                self.uncertainty_obj_continuous[var] = uncertainty
            else:
                self.uncertainty_obj[var] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, 0, 1
            )
            self.uncertainty_obj_dependent[var] = uncertainty_dependent

        return var

    def addConstr(
            self,
            lhs,
            sense=None,
            rhs=None,
            name="",
            uncertainty=None,
            uncertainty_dependent=None,
    ):
        """Add a constraint to the model. Generalize gurobipy.addConstr()
        to incorporate uncertainty in a constraint

        Parameters
        ----------
        uncertainty: dict, optional, default=None
            The scenarios/univariate random variable generator of the
            stage-wise independent uncertain constraint coefficient and RHS

        uncertainty_dependent: dict, optional, default=None
            The location index in the sample path genator of the stage-wise
            dependent uncertain constraint coefficient and RHS

        Returns
        -------
        The created constraint: gurobipy.Constr

        Examples
        --------
        >>> new, past = model.addStateVar(ub=2.0)

        stage-wise independent finite discrete uncertain rhs/constraint coefficient

        >>> newConstr = model.addConstr(
        ...     new + past == 3.0,
        ...     uncertainty={'rhs': [1,2,3], new: [3,4,5]}
        ... )

        The above example dictates scenarios of RHS to be [1,2,3] and
        coefficient of new to be [3,4,5]

        stage-wise independent continuous uncertain rhs/constraint coefficient

        >>> def f(random_state):
        ...     return random_state.normal(0, 1)
        >>> newConstr = model.addConstr(
        ...     ub=2.0,
        ...     uncertainty={new: f},
        ...     uncertainty_dependent = {'rhs': [1]}
        ... )

        The above constraint contains a stage-wise independent uncertain
        constraint coefficient and a Markovian RHS
        """
        constr = self._model.addConstr(lhs, sense=sense, rhs=rhs, name=name)
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, 1, 1)
            for key, value in uncertainty.items():
                # key can be a gurobipy.Var or "rhs"
                # Append constr to the key
                if type(key) == gurobipy.Var:
                    if callable(value):
                        self.uncertainty_coef_continuous[(constr, key)] = value
                    else:
                        self.uncertainty_coef[(constr, key)] = value
                elif type(key) == str and key.lower() == "rhs":
                    if callable(value):
                        self.uncertainty_rhs_continuous[constr] = value
                    else:
                        self.uncertainty_rhs[constr] = value
                else:
                    raise ValueError("wrong uncertainty key!")

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, 1, 1
            )
            for key, value in uncertainty_dependent.items():
                # key can be a gurobipy.Var or "rhs"
                # Append constr to the key
                if type(key) == gurobipy.Var:
                    if not any(key is item for item in self._model.getVars()):
                        raise ValueError("wrong uncertainty key!")
                    self.uncertainty_coef_dependent[(constr, key)] = value
                elif type(key) == str and key.lower() == "rhs":
                    self.uncertainty_rhs_dependent[constr] = value
                else:
                    raise ValueError("wrong uncertainty key!")

        return constr

    def addConstrs(
        self, generator, name="", uncertainty=None, uncertainty_dependent=None
    ):
        """Add constraints in bulk to the model. Generalize gurobipy.addConstrs()
        to incorporate uncertainty on the RHS of the constraints.
        If you want to add constraints with uncertainties on coefficients,
        use addConstr() instead and add those constraints one by one

        Parameters
        ----------
        uncertainty: array-like/callable, optional, default=None
            The scenarios/multivariate random variable generator of the
            stage-wise independent uncertain RHS. A generator must take numpy
            RandomState as its only argument

        uncertainty_dependent: array-like, optional, default=None
            The locations in the sample path generator of the stage-wise
            independent uncertain RHS. A generator must take numpy
            RandomState as its only argument

        Returns
        -------
        The created constraints: list of gurobipy.Constr

        Examples
        --------
        >>> new, past = model.addStateVar(ub=2.0)

        stage-wise independent discrete uncertain RHSs

        >>> newConstrs = model.addConstrs(
        ...     new[i] + past[i] == 0 for i in range(2),
        ...     uncertainty=[[1,2],[2,3]]
        ... )

        The above example dictates scenarios of RHSs to be [1,2] and [2,3]

        stage-wise independent continuous uncertain RHSs

        >>> def f(random_state):
        ...     return random_state.multivariate_normal(
        ...         mean = [0,0],
        ...         cov = [[1,0],[0,100]]
        ...     )
        >>> newConstrs = model.addConstrs(
        ...        (new[i] + past[i] == 0 for i in range(2)),
        ...        uncertainty=f
        ... )

        Markovian uncertain RHSs

        >>> newConstrs = model.addConstrs(
        ...     (new[i] + past[i] == 0 for i in range(2)),
        ...     uncertainty_dependent = [0,1],
        ... )
        """
        constr = self._model.addConstrs(generator, name=name)
        self._model.update()

        if uncertainty is not None:
            uncertainty = self._check_uncertainty(uncertainty, 0, len(constr))
            if callable(uncertainty):
                self.uncertainty_rhs_continuous[
                    tuple(constr.values())
                ] = uncertainty
            else:
                self.uncertainty_rhs[tuple(constr.values())] = uncertainty

        if uncertainty_dependent is not None:
            uncertainty_dependent = self._check_uncertainty_dependent(
                uncertainty_dependent, 0, len(constr)
            )
            self.uncertainty_rhs_dependent[
                tuple(constr.values())
            ] = uncertainty_dependent

        return constr

    def _discretize(self, n_samples, random_state, replace=True):
        """Discretize stage-wise independent continuous uncertainties.

        Parameters
        ----------
        n_samples: int
            number of samples to generate uniformly from the distribution

        random_state: None | int | instance of RandomState, optional
            If int, random_state is the seed used by the
            random number generator;
            If RandomState instance, random_state is the
            random number generator;
            If None, the random number generator is the
            RandomState instance used by numpy.random.
            Default is None.

        replace: boolean, optional, default is True
            Whether the sample is with or without replacement.
        """
        if hasattr(self,'_flag_discrete') and self._flag_discrete == 1: return
        # Discretize continuous true problem
        if self._type == "continuous":
            self.n_samples = n_samples
            # Order of discretization matters
            for key, dist in sorted(
                self.uncertainty_rhs_continuous.items(),
                key=lambda t: repr(t[0]),
            ):
                self.uncertainty_rhs[key] = [
                    dist(random_state) for _ in range(self.n_samples)
                ]
            for key, dist in sorted(
                self.uncertainty_obj_continuous.items(),
                key=lambda t: repr(t[0]),
            ):
                self.uncertainty_obj[key] = [
                    dist(random_state) for _ in range(self.n_samples)
                ]
            for key, dist in sorted(
                self.uncertainty_coef_continuous.items(),
                key=lambda t: repr(t[0]),
            ):
                self.uncertainty_coef[key] = [
                    dist(random_state) for _ in range(self.n_samples)
                ]
            for keys, dist in sorted(
                self.uncertainty_mix_continuous.items(),
                key=lambda t: repr(t[0]),
            ):
                for i in range(self.n_samples):
                    sample = dist(random_state)
                    for index, key in enumerate(keys):
                        if type(key) == gurobipy.Var:
                            if key not in self.uncertainty_obj.keys():
                                self.uncertainty_obj[key] = [sample[index]]
                            else:
                                self.uncertainty_obj[key].append(sample[index])
                        elif type(key) == gurobipy.Constr:
                            if key not in self.uncertainty_rhs.keys():
                                self.uncertainty_rhs[key] = [sample[index]]
                            else:
                                self.uncertainty_rhs[key].append(sample[index])
                        else:
                            if key not in self.uncertainty_coef.keys():
                                self.uncertainty_coef[key] = [sample[index]]
                            else:
                                self.uncertainty_coef[key].append(
                                    sample[index]
                                )
        # Discretize discrete true problem
        else:
            if n_samples > self.n_samples:
                raise Exception(
                    "n_samples should be smaller than the total number of samples!"
                )
            for key, samples in sorted(
                self.uncertainty_rhs.items(), key=lambda t: repr(t[0])
            ):
                self.uncertainty_rhs_discrete[key] = samples
                # numpy.random.choice does not work on multi-dimensional arrays
                drawed_indices = rand_int(
                    self.n_samples,
                    random_state,
                    size=n_samples,
                    probability=self.probability,
                    replace=replace,
                )
                self.uncertainty_rhs[key] = [
                    samples[index]
                    for index in drawed_indices
                ]
            for key, samples in sorted(
                self.uncertainty_obj.items(), key=lambda t: repr(t[0])
            ):
                self.uncertainty_obj_discrete[key] = samples
                drawed_indices = rand_int(
                    self.n_samples,
                    random_state,
                    size=n_samples,
                    probability=self.probability,
                    replace=replace,
                )
                self.uncertainty_obj[key] = [
                    samples[index]
                    for index in drawed_indices
                ]
            for key, samples in sorted(
                self.uncertainty_coef.items(), key=lambda t: repr(t[0])
            ):
                self.uncertainty_coef_discrete[key] = samples
                drawed_indices = rand_int(
                    self.n_samples,
                    random_state,
                    size=n_samples,
                    probability=self.probability,
                    replace=replace,
                )
                self.uncertainty_coef[key] = [
                    samples[index]
                    for index in drawed_indices
                ]
            self.n_samples_discrete = self.n_samples
            self.n_samples = n_samples
        self._flag_discrete = 1

    def _update_uncertainty(self, k):
        # Update model with the k^th stage-wise independent discrete uncertainty
        if self.uncertainty_coef is not None:
            for (constr, var), value in self.uncertainty_coef.items():
                self._model.chgCoeff(constr, var, value[k])
        if self.uncertainty_rhs is not None:
            for constr_tuple, value in self.uncertainty_rhs.items():
                if type(constr_tuple) == tuple:
                    self._model.setAttr("RHS", list(constr_tuple), value[k])
                else:
                    constr_tuple.setAttr("RHS", value[k])
        if self.uncertainty_obj is not None:
            for var_tuple, value in self.uncertainty_obj.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr("Obj", list(var_tuple), value[k])
                else:
                    var_tuple.setAttr("Obj", value[k])

    def _update_uncertainty_discrete(self, k):
        # update model with the k^th stage-wise independent true discrete uncertainty
        if self.uncertainty_coef_discrete is not None:
            for (constr, var), value in self.uncertainty_coef_discrete.items():
                self._model.chgCoeff(constr, var, value[k])
        if self.uncertainty_rhs_discrete is not None:
            for constr_tuple, value in self.uncertainty_rhs_discrete.items():
                if type(constr_tuple) == tuple:
                    self._model.setAttr("RHS", list(constr_tuple), value[k])
                else:
                    constr_tuple.setAttr("RHS", value[k])
        if self.uncertainty_obj_discrete is not None:
            for var_tuple, value in self.uncertainty_obj_discrete.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr("Obj", list(var_tuple), value[k])
                else:
                    var_tuple.setAttr("Obj", value[k])

    def _sample_uncertainty(self, random_state=None):
        # Sample stage-wise independent true continuous uncertainty
        random_state = check_random_state(random_state)
        if self.uncertainty_coef_continuous is not None:
            for (
                (constr, var),
                dist,
            ) in self.uncertainty_coef_continuous.items():
                self._model.chgCoeff(constr, var, dist(random_state))
        if self.uncertainty_rhs_continuous is not None:
            for constr_tuple, dist in self.uncertainty_rhs_continuous.items():
                if type(constr_tuple) == tuple:
                    self._model.setAttr("RHS", list(constr_tuple), dist(random_state))
                else:
                    constr_tuple.setAttr("RHS", dist(random_state))
        if self.uncertainty_obj_continuous is not None:
            for var_tuple, dist in self.uncertainty_obj_continuous.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr("Obj", list(var_tuple), dist(random_state))
                else:
                    var_tuple.setAttr("Obj", dist(random_state))
        if self.uncertainty_mix_continuous is not None:
            for keys, dist in self.uncertainty_mix_continuous.items():
                sample = dist(random_state)
                for index, key in enumerate(keys):
                    if type(key) == gurobipy.Var:
                        key.setAttr("Obj", sample[index])
                    elif type(key) == gurobipy.Constr:
                        key.setAttr("RHS", sample[index])
                    else:
                        self._model.chgCoeff(key[0], key[1], sample[index])

    def _update_uncertainty_dependent(self, Markov_state):
        # Update model with a Markov state
        if self.uncertainty_coef_dependent is not None:
            for (constr,var), value in self.uncertainty_coef_dependent.items():
                self._model.chgCoeff(constr, var, Markov_state[value])
        if self.uncertainty_rhs_dependent is not None:
            for constr_tuple, value in self.uncertainty_rhs_dependent.items():
                if type(constr_tuple) == tuple:
                    self._model.setAttr(
                        "RHS",
                        list(constr_tuple),
                        [Markov_state[i] for i in value],
                    )
                else:
                    constr_tuple.setAttr("RHS", Markov_state[value])
        if self.uncertainty_obj_dependent is not None:
            for var_tuple, value in self.uncertainty_obj_dependent.items():
                if type(var_tuple) == tuple:
                    self._model.setAttr(
                        "Obj",
                        list(var_tuple),
                        [Markov_state[i] for i in value],
                    )
                else:
                    var_tuple.setAttr("Obj", Markov_state[value])

    def _set_up_link_constrs(self):
        if self.link_constrs == []:
            self.link_constrs = list(
                self._model.addConstrs(
                    (var == var.lb for var in self.local_copies),
                    name="link_constrs",
                ).values()
            )

    def _delete_link_constrs(self):
        if self.link_constrs != []:
            for constr in self.link_constrs:
                self._model.remove(constr)
            self.link_constrs = []

    def _set_up_CTG(self, discount, bound):
        # if it's a minimization problem, we need a lower bound for alpha
        if self.modelsense == 1:
            if self.alpha is None:
                self.alpha = self._model.addVar(
                    lb=bound,
                    ub=gurobipy.GRB.INFINITY,
                    obj=discount,
                    name="alpha",
                )
        # if it's a maximation problem, we need an upper bound for alpha
        else:
            if self.alpha is None:
                self.alpha = self._model.addVar(
                    ub=bound,
                    lb=-gurobipy.GRB.INFINITY,
                    obj=discount,
                    name="alpha",
                )
    def _delete_CTG(self):
        if self.alpha is not None:
            self._model.remove(self.alpha)
            self.alpha = None

    def _update_link_constrs(self, fwdSoln):
        self._model.setAttr("RHS", self.link_constrs, fwdSoln)

    def _add_cut(self, rhs, gradient):
        temp = gurobipy.LinExpr(gradient, self.states)
        self.cuts.append(
            self._model.addConstr(
                self.modelSense * (self.alpha - temp - rhs) >= 0
            )
        )
        self._model.update()

    def _remove_cut(self, i):
        self._model.remove(self.cuts[i])
        del self.cuts[i]
        self._model.update()

    def _reset(self):
        for cut in self.cuts:
            self._model.remove(cut)
        self.cuts = []
        self._delete_link_constrs()
        self._delete_CTG()
        self._model.update()
        self._model.reset()

    def _solveLP(self):
        objLPScen = numpy.empty(self.n_samples)
        gradLPScen = numpy.empty((self.n_samples, self.n_states))
        for k in range(self.n_samples):
            self._update_uncertainty(k)
            self.optimize()
            if self._model.status not in [2,11]:
                self.write_infeasible_model("backward_" + str(self._model.modelName))
            objLPScen[k] = self.objVal
            gradLPScen[k] = self.getAttr("Pi", self.link_constrs)
        return objLPScen, gradLPScen

    def _average(self, objLPScen, gradLPScen, probability=None):
        p = self.probability if probability is None else probability
        return self.measure(
            obj=objLPScen,
            grad=gradLPScen,
            p=p,
            sense=self._model.modelSense)

    def set_probability(self, probability):
        """
        Set probability measure of discrete scenarios.

        Parameters
        ----------
        probability: array-like
            Probability of scenarios. Default is uniform measure
            [1/n_samples for _ in range(n_samples)].
            Length of the list must equal length of uncertainty.
            The order of the list must match with the order of
            uncertainty list.

        Examples
        --------
        >>> newVar = model.addVar(ub=2.0, uncertainty=[1,2,3])
        >>> model.setProbability([0.2,0.3,0.4])
        """
        self.probability = list(probability)
        if len(probability) != self.n_samples:
            raise ValueError(
                "probability tree != compatible with scenario tree"
            )

    def add_continuous_uncertainty(self, uncertainty, locations):
        """Add continuous stage-wise independent uncertainties.

        Parameters
        ----------
        uncertainty: callable
            A random variable generator that takes numpy RandomState as its only
            argument.

        location: list
            Entries of the list can be gurobipy.Var, gurobipy.Constr,
            or (gurobipy.Constr, gurobipy.var).

        The dimension of random variable generator should equal
        the length of locations.

        Examples
        --------
        >>> now, past = model.addStateVar()
        >>> TS = model.addConstr(now - past == 0)
        >>> model.add_continuous_uncertainty(f, [(TS, past), TS])
        """
        for item in locations:
            if type(item) not in [gurobipy.Var, gurobipy.Constr]:
                if type(item) != tuple:
                    raise TypeError("wrong locations format!")
                else:
                    if (
                        type(item[0]) != gurobipy.Constr
                        or type(item[1]) != gurobipy.Var
                    ):
                        raise TypeError("wrong locations format!")
        self._check_uncertainty(
            uncertainty, flag_dict=1, list_dim=len(locations)
        )
        self.uncertainty_mix_continuous = {tuple(locations): uncertainty}
        self._type = "continuous"

    def _record_discrete_uncertainty_to_cache(self):
        cache = {}
        cache['uncertainty_coef'] = self.uncertainty_coef
        cache['uncertainty_rhs'] = self.uncertainty_rhs
        cache['uncertainty_obj'] = self.uncertainty_obj
        cache['probability'] = self.probability
        cache['n_samples'] = self.n_samples
        return cache

    def _remove_discrete_uncertainty(self):
        self.uncertainty_coef = {}
        self.uncertainty_rhs = {}
        self.uncertainty_obj = {}
        self.probability = None
        self.n_samples = 1
        self._flag_discrete = 0

    def _recover_discrete_uncertainty_from_cache(self, cache):
        for k,v in cache.items():
            setattr(self, k, v)

    @property
    def controls(self):
        """Get control variables"""
        vars = self._model.getVars()
        states_name = [state.varName for state in self.states]
        local_copies_name = [
            local_copy.varName for local_copy in self.local_copies
        ]
        return [
            var
            for var in vars
            if var.varName not in states_name + local_copies_name
        ]

    @property
    def states_and_controls(self):
        """Get state and control variables"""
        return self.states + self.controls

    def get_cut_coeffs_and_rhs(self):
        """Get coefficients and rhs of cuts.
        If minimization, cuts take the form of alpha + ax + by >= c,
        If maximization, cuts take the form of alpha + ax + by <= c,
        Returns a dictionary:
            {x.varName: [a1,a2], y.varName: [b1,b2], rhs: [c1,c2]}
        """
        result = {state.varName: [] for state in self.states}
        result["rhs"] = []
        for cut in self.cuts:
            flag = 1 if self.getCoeff(cut, self.alpha) == 1 else -1
            for state in self.states:
                result[state.varName].append(flag * self.getCoeff(cut, state))
            result["rhs"].append(flag * cut.rhs)
        return result

    def optimize(self):
        # Just for time statistics
        self._model.optimize()

    def write_infeasible_model(self, text):
        self._model.write('./' + text + ".lp")
        self._model.computeIIS()
        self._model.write('./' + text + ".ilp")
        raise Exception(
            "infeasibility caught; check complete recourse condition!"
        )

    def regularize(self, center, norm, a, b, i):
        """Regularize a stochastic model.

        Parameters
        ----------

        center: array-like
            The regularization center with length n_states.

        norm: 'L1'/'L2'
            The norm to use for regularization.

        a,b,i: float,float,integer (a>0, 0<b<1, i>0)
            The coefficient of the regularization term is a*b^{i}, where i is
            the index of iteration.
        """
        self.rgl = self._model.addVar(
            lb=0,
            obj=self.modelsense*b**i,
            name='rgl'
        )
        if norm == 'L1':
            self.rgl_constr = self._model.addConstrs(
                (self.rgl >= a*(self.states[i] - center[i])
                for i in range(self.n_states)),
                name = 'rgl'
            ).values()
        elif norm == 'L2':
            self.rgl_constr = [self._model.addQConstr(
                self.rgl -
                a*gurobipy.QuadExpr(
                    gurobipy.quicksum([
                        self.states[i] * self.states[i]
                        - self.states[i] * 2 * center[i]
                        + center[i] * center[i]
                        for i in range(self.n_states)
                    ])
                )
                >=0,
                name = 'rgl'
            )]
        self._model.update()

    def _deregularize(self):
        self._model.remove(self.rgl)
        for constr in self.rgl_constr:
            self._model.remove(constr)
        self._model.update()


class StochasticModelLG(StochasticModel):
    def _solveSB(self, gradLPScen):
        objSBScen = numpy.empty(self.n_samples)
        for i in range(self.n_samples):
            self.setAttr("obj", self.local_copies, [-x for x in gradLPScen[i]])
            self._update_uncertainty(i)
            self.optimize()
            objSBScen[i] = self.objBound
        return objSBScen

    def _solvePrimal(self):
        objVal_primal = [None for _ in range(self.n_samples)]
        for i in range(self.n_samples):
            self._update_uncertainty(i)
            self.optimize()
            objVal_primal[i] = self.objBound
        return objVal_primal

    def _solveLG(
            self,
            gradLPScen,
            given_bound,
            objVal_primal,
            flag_tight,
            forward_solution,
            step_size,
            max_stable_iterations,
            max_iterations,
            max_time,
            MIPGap,
            tol):
        n_local_copies = len(self.local_copies)
        objLGScen = numpy.empty(self.n_samples)
        gradLGScen = numpy.empty((self.n_samples, self.n_states))
        for k in range(self.n_samples):
            # Benchmark is objVal of primal problem if LG is tight, otherwise
            # it is updated later as the objVal of cut problem
            benchmark = objVal_primal[k]
            self._update_uncertainty(k)
            # Initialize the objVal_best_so_far objVal as the known bound and
            # related objVal_best_so_far gradient as the solution of duals
            objVal_best_so_far = given_bound
            grad_best_so_far = gradLPScen[k]
            # Initialize the current gradient as the solution of dual variables
            grad_current = gradLPScen[k]
            # Set up projection model
            model_proj = gurobipy.Model(env=self.env)
            model_proj.Params.outputFlag = 0
            pi_proj = model_proj.addVars(
                n_local_copies,
                lb=-gurobipy.GRB.INFINITY,
            ).values()
            model_proj.update()
            # the objective of projection model is \pi^2 - 2\pi\grad_best_so_far
            model_proj.setObjective(gurobipy.quicksum(x * x for x in pi_proj))
            # Set up cut model
            if not flag_tight:
                model_cut = gurobipy.Model(env=self.env)
                model_cut.Params.outputFlag = 0
                model_cut.modelsense = -self.modelsense
                theta = model_cut.addVar(lb=-gurobipy.GRB.INFINITY, obj=1)
                if self.modelSense == 1:
                    theta.ub = objVal_primal[k]
                else:
                    theta.lb = objVal_primal[k]
                pi_cut = model_cut.addVars(
                    n_local_copies,
                    lb=-gurobipy.GRB.INFINITY,
                ).values()
                model_cut.update()

            stable_iterations = 0
            iterations = 0
            total_time = 0
            while (
                stable_iterations < max_stable_iterations
                and total_time < max_time
                and iterations < max_iterations
            ):
                start = time.time()
                # Solve the inner problem
                self.setAttr(
                    "obj", self.local_copies, [-x for x in grad_current]
                )
                self.Params.MIPGap = MIPGap
                self.optimize()
                if self.status not in [2,11]:
                    break
                # get the current objVal and gradient for the outer problem
                grad_outer = [
                    forward_solution[i] - self.local_copies[i].X
                    for i in range(n_local_copies)
                ]
                objVal_current = self.objBound
                objVal_current += sum(
                    x * y for x, y in zip(grad_current, forward_solution)
                )
                # Update cut model
                if not flag_tight:
                    cut_const = objVal_current - sum(
                        x * y for x, y in zip(grad_current, grad_outer)
                    )
                    cut_expr = gurobipy.LinExpr(grad_outer, pi_cut)
                    model_cut.addConstr(
                        self.modelSense * (theta-cut_expr-cut_const) <= 0
                    )
                    model_cut.optimize()
                    if model_cut.status not in [2,11]:
                        break
                    benchmark = model_cut.objVal
                # update outer problem best so far solution and optimal value
                if self.modelsense * (objVal_current - objVal_best_so_far) > 0:
                    objVal_best_so_far = objVal_current
                    grad_best_so_far = grad_current
                    stable_iterations = 0
                else:
                    stable_iterations += 1
                # Update projection model
                model_proj.setAttr(
                    "obj", pi_proj, [-2 * x for x in grad_best_so_far]
                )
                # determine the level
                delta = benchmark - objVal_best_so_far
                if abs(delta) <= tol * abs(benchmark):
                    break
                level = step_size * objVal_best_so_far + (1-step_size) * benchmark
                # current + gradient * (pi_proj - grad_current) >=(<=) level
                temp1 = sum(x * y for x, y in zip(grad_outer, grad_current))
                temp2 = gurobipy.LinExpr(grad_outer, pi_proj)
                new_cut = model_proj.addConstr(
                    self.modelsense * (objVal_current + temp2 - temp1 - level)
                    >= 0
                )
                model_proj.optimize()
                # Numerical issue may occur if closed to optimality
                if model_proj.status not in [2,11]:
                    break
                # Update gradient
                grad_current = model_proj.getAttr("X", pi_proj)
                iterations += 1
                end = time.time()
                total_time += end - start
            #! level iterations end
            # objLGScen[k] = objVal_best_so_far
            gradLGScen[k] = grad_best_so_far
            self.setAttr(
                "obj", self.local_copies, [-x for x in grad_best_so_far]
            )
            self.optimize()
            objLGScen[k] = self.objBound
        #! scenario iterations end
        return objLGScen, gradLGScen

    def _binarize(self, precision, n_binaries, transition=0):
        # Binarize StochasticModel. StochasticModel at transition stage keeps
        # states in original space while binarzing local_copies
        self.n_states_original_space = self.n_states
        self.local_copies_original_space = self.local_copies
        self.states_original_space = self.states
        if transition == 0:
            self.states = []
            self.n_states = 0
        self.local_copies = []
        for i, (x, y) in enumerate(
            zip(self.states_original_space, self.local_copies_original_space)
        ):
            if transition == 0:
                states = self.addVars(
                    n_binaries[i], vtype=gurobipy.GRB.BINARY, name=x.varName
                ).values()
            local_copies = self.addVars(
                n_binaries[i], vtype=gurobipy.GRB.BINARY, name=y.varName
            ).values()
            self.update()
            if transition == 0:
                temp1 = gurobipy.quicksum(
                    pow(2, i) * states[i] for i in range(n_binaries[i])
                )
            temp2 = gurobipy.quicksum(
                pow(2, i) * local_copies[i] for i in range(n_binaries[i])
            )
            # Assume bounds are the same over time!
            if x.vtype not in ["I","B"]:
                if transition == 0:
                    self.addConstr(
                        temp1 == precision * (x - x.lb),
                        name="binarize_states_{}".format(i),
                    )
                self.addConstr(
                    temp2 == precision * (y - y.lb),
                    name="binarize_local_copies_{}".format(i),
                )
            else:
                x.lb = math.ceil(x.lb)
                y.lb = math.ceil(y.lb)
                if transition == 0:
                    self.addConstr(
                        temp1 == x - x.lb,
                        name="binarize_states_{}".format(i),
                    )
                self.addConstr(
                    temp2 == y - y.lb,
                    name="binarize_local_copies_{}".format(i),
                )
            if transition == 0:
                self.states += states
                self.n_states += n_binaries[i]
            self.local_copies += local_copies

    def _back_binarize(self, precision, n_binaries, transition=0):
        if not hasattr(self, "states_original_space"):
            return
        for i, (x, y) in enumerate(
            zip(self.states_original_space, self.local_copies_original_space)
        ):
            # Binarized states don't exist at transition stage
            if x.vtype not in ["B","I"]:
                if transition == 0:
                    temp = self.getConstrByName("binarize_states_{}".format(i))
                    expr = self.getRow(temp)
                    rhs = temp.rhs
                    self.remove(temp)
                    self.addConstr(
                        expr >= rhs,
                        name="back_binarize_states_lower"
                    )
                    self.addConstr(
                        expr <= rhs+0.99,
                        name="back_binarize_states_upper"
                    )
                temp = self.getConstrByName("binarize_local_copies_{}".format(i))
                expr = self.getRow(temp)
                rhs = temp.rhs
                self.remove(temp)
                self.addConstr(expr >= rhs)
                self.addConstr(expr <= rhs+0.99)

        # Re-set-up states and local copies
        self.states = self.states_original_space
        self.local_copies = self.local_copies_original_space
        self.n_states = len(self.states)
        # Re-set-up linking constraints
        for constr in self.link_constrs:
            self.remove(constr)
        self.link_constrs = []
        self._model.update()

    def _copy(self, model):
        result = super()._copy(model)
        if hasattr(self, "n_states_original_space"):
            result.n_states_original_space = self.n_states_original_space
        if hasattr(self, "states_original_space"):
            result.states_original_space = [
                result._model.getVarByName(x.varName)
                for x in self.states_original_space
            ]
        if hasattr(self, "local_copies_original_space"):
            result.local_copies_original_space = [
                result._model.getVarByName(x.varName)
                for x in self.local_copies_original_space
            ]
        return result

    # def removeCut(self, fwdSoln, objLP, gradLP):
    #
    #     # initialize cut information #
    #     if not hasattr(self, "_fwdSolnTotal"):
    #         self._fwdSolnTotal = []
    #     if not hasattr(self, "_maxCut"):
    #         self._maxCut = []
    #     if not hasattr(self, "_maxValue"):
    #         self._maxValue = []
    #
    #     numIter = len(self._fwdSolnTotal)
    #
    #     for i in range(numIter):
    #         newValue = objLP - numpy.dot(gradLP, self._fwdSolnTotal[i])
    #         if newValue > self._maxValue[i]:
    #             self._maxCut[i] = self.cuts[-1]
    #             self._maxValue[i] = newValue
    #
    #     maxValue = objLP - numpy.dot(gradLP, fwdSoln)
    #     maxCut = self.cuts[-1]
    #
    #     for cut in self.cuts[:-1]:
    #         flag = 1 if self.getCoeff(cut, self.alpha) == 1 else -1
    #         value = flag * cut.rhs - numpy.dot([flag * self.getCoeff(cut, state) for state in uncertainty_obj_dependent], fwdSoln)
    #         if value > maxValue:
    #             maxValue = value
    #             maxCut = cut
    #
    #     self._maxValue.append(maxValue)
    #     self._maxCut.append(maxCut)
    #
    #     retainedCuts = set(self._maxCut[i] for i in range(numIter+1))
    #     for cut in self.cuts:
    #         if not cut in retainedCuts:
    #             self.remove(cut)
    #     self.cuts = list(retainedCuts)
    #     self._fwdSolnTotal.append(fwdSoln)
