#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import logging


class Logger(object):
    """Log base class.
    Parameters
    ----------
        logFile: bool
            The switch of logging to files

        logToConsole: bool
            The switch of logging to console

    Attributes
    ----------
        logger:
            The logger

        time:
            The time spent on the logged jobs

        n_slots:
            The number of horizontal slots the logger needs
    """
    def __init__(self, logFile, logToConsole, directory):
        name = self.__repr__()
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        if logFile != 0:
            handler = logging.FileHandler(directory + name + ".log", mode="a")
            logger.addHandler(handler)
        if logToConsole != 0:
            streamHandler = logging.StreamHandler()
            logger.addHandler(streamHandler)
        self.logger = logger
        self.time = 0

    def __repr__(self):
        return ""

    def header(self):
        pass

    def text(self):
        pass

    def footer(self):
        self.logger.info("-" * self.n_slots)
        self.logger.info("Time: {} seconds".format(self.time))

class LoggerSDDP(Logger):
    def __init__(self, percentile, n_processes, **kwargs):
        self.percentile = percentile
        self.n_processes = n_processes
        super().__init__(**kwargs)
        self.n_slots = 84 if self.n_processes > 1 else 64

    def __repr__(self):
        return "SDDP"

    def header(self):
        self.logger.info("-" * self.n_slots)
        temp = "{:^}"
        self.logger.info(
            "{:^{width}s}".format("SDDP Solver, Lingquan Ding", width=self.n_slots)
        )
        self.logger.info("-" * self.n_slots)
        if self.n_processes > 1:
            self.logger.info(
                "{:>12s}{:>20s}{:^40s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value {}% CI ({})".format(self.percentile,self.n_processes),
                    "Time"
                )
            )
        else:
            self.logger.info(
                "{:>12s}{:>20s}{:>20s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value",
                    "Time"
                )
            )
        self.logger.info("-" * self.n_slots)

    def text(self, iteration, db, time, pv=None, CI=None):
        if self.n_processes > 1:
            self.logger.info(
                "{:>12d}{:>20f}{:>19f}, {:<19f}{:>12f}".format(
                    iteration, db, CI[0], CI[1], time
                )
            )
        else:
            self.logger.info(
                "{:>12d}{:>20f}{:>20f}{:>12f}".format(
                    iteration, db, pv, time
                )
            )
        self.time += time

    def footer(self, reason):
        super().footer()
        self.logger.info("Algorithm stops since " + reason)


class LoggerEvaluation(Logger):
    def __init__(self, percentile, n_simulations, **kwargs):
        self.percentile = percentile
        self.n_simulations = n_simulations
        self.n_slots = 76 if self.n_simulations in [-1,1] else 96
        super().__init__(**kwargs)

    def __repr__(self):
        return "Evaluation"

    def header(self):
        self.logger.info("-" * self.n_slots)
        self.logger.info(
            "{:^{width}s}".format(
                "Evaluation for approximation model, Lingquan Ding",
                width=self.n_slots
            )
        )
        self.logger.info("-" * self.n_slots)
        if self.n_simulations not in [-1,1]:
            self.logger.info(
                "{:>12s}{:>20s}{:^40s}{:>12s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value {}% CI ({})".format(self.percentile,self.n_simulations),
                    "Time",
                    "Gap",
                )
            )
        else:
            self.logger.info(
                "{:>12s}{:>20s}{:>20s}{:>12s}"
                .format(
                    "Iteration",
                    "Bound",
                    "Value",
                    "Time",
                )
            )
        self.logger.info("-" * self.n_slots)

    def text(self, iteration, db, time, pv=None, CI=None, gap=None):
        if self.n_simulations > 1:
            self.logger.info(
                "{:>12d}{:>20f}{:>19f}, {:<19f}{:>12f}{:>12}".format(
                    iteration, db, CI[0], CI[1], time, gap
                )
            )
        else:
            self.logger.info(
                "{:>12d}{:>20f}{:>20f}{:>12f}{:>12}".format(
                    iteration, db, pv, time, gap
                )
            )
        self.time += time

class LoggerComparison(Logger):
    def __init__(self, percentile, n_simulations, **kwargs):
        self.percentile = percentile
        self.n_simulations = n_simulations
        self.n_slots = 64 if self.n_simulations in [-1,1] else 84
        super().__init__(**kwargs)

    def __repr__(self):
        return "Comparison"

    def header(self):
        assert self.n_simulations != 1
        self.logger.info("-" * self.n_slots)
        self.logger.info(
            "{:^{width}s}".format(
                "Comparison for approximation model, Lingquan Ding",
                width=self.n_slots
            )
        )
        self.logger.info("-" * self.n_slots)
        if self.n_simulations != -1:
            self.logger.info(
                "{:>12s}{:>20s}{:^40s}{:>12s}"
                .format(
                    "Iteration",
                    "Referece iter.",
                    "Difference {}% CI ({})".format(self.percentile,self.n_simulations),
                    "Time",
                )
            )
        else:
            self.logger.info(
                "{:>12s}{:>20s}{:>20s}{:>12s}"
                .format(
                    "Iteration",
                    "Referece iter.",
                    "Difference",
                    "Time",
                )
            )
        self.logger.info("-" * self.n_slots)

    def text(self, iteration, ref_iteration, time, diff_CI=None, diff=None):
        assert self.n_simulations != 1
        if self.n_simulations != -1:
            self.logger.info(
                "{:>12d}{:>20d}{:>19f}, {:<19f}{:>12f}".format(
                    iteration, ref_iteration, diff_CI[0], diff_CI[1], time
                )
            )
        else:
            self.logger.info(
                "{:>12d}{:>20d}{:>20f}{:>12f}".format(
                    iteration, ref_iteration, diff, time
                )
            )
        self.time += time
