#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""
import numpy
from numbers import Number
import pandas
import multiprocessing


class Markovian(object):
    def __init__(self, f, n_Markov_states, n_sample_paths, int_flag=0):
        """
        Parameters
        ----------
        f: callable
            The sample path generator. It must take random_state and n_samples as
            arguments and returns a three dimensional numpy array
            (n_samples * t * n_states)

        n_Markov_states: array-like
            The intended number of Markov states for each stages

        n_sample_paths: int
            The intended number of sample paths to train the Markov chain

        int_flag: bool
            Whether to round the Markov states to integer
        """
        self.samples = f(numpy.random.RandomState(0),size=n_sample_paths)
        shape = self.samples.shape
        self.T, self.dim_Markov_states = shape[1:]
        self.n_Markov_states = n_Markov_states
        self.f = f
        self.int_flag = int_flag
        self.n_samples = n_sample_paths
        self.Markov_states = [None for t in range(self.T)]
        self.Markov_states[0] = self.samples[0,0,:].reshape(1,-1)

    def _initialize(self):
        """initialize Markov states."""
        for t in range(1,self.T):
            self.Markov_states[t] = self.samples[:self.n_Markov_states[t],t,:]

    def _initialize_matrix(self):
        """initialize transition matrix."""
        self.transition_matrix = [numpy.array([[1]])]
        self.transition_matrix += (
            [numpy.zeros([self.n_Markov_states[t-1],self.n_Markov_states[t]])
            for t in range(1,self.T)]
        )

    def round_to_int(self):
        """round Markov states to integer."""
        for t in range(1,self.T):
            self.Markov_states[t] = numpy.rint(self.Markov_states[t])

    def SA(self):
        """Use stochastic approximation to compute the partition."""
        self._initialize()
        for idx, sample in enumerate(self.samples):
            step_size = 1.0/(idx+1)
            for t in range(1,self.T):
                temp = self.Markov_states[t] - sample[t]
                idx = numpy.argmin(numpy.sum(temp**2, axis=1))
                self.Markov_states[t][idx] += (
                    (sample[t]-self.Markov_states[t][idx]) * step_size
                )
        self.train_transition_matrix()
        return (self.Markov_states,self.transition_matrix)

    def RSA(self):
        """Use robust stochastic approximation to compute the partition."""
        self._initialize()
        self.iterate = [
            self.Markov_states[t].copy()
            for t in range(self.T)
        ]
        step_size = 1.0/numpy.sqrt(self.n_samples)
        for idx, sample in enumerate(self.samples):
            for t in range(1,self.T):
                temp = self.iterate[t] - sample[t]
                idx = numpy.argmin(numpy.sum(temp**2, axis=1))
                self.iterate[t][idx] += (
                    (sample[t]-self.iterate[t][idx]) * step_size
                )
            for t in range(1,self.T):
                self.Markov_states[t] += self.iterate[t]
        for t in range(1,self.T):
            self.Markov_states[t] = self.Markov_states[t]/self.n_samples
        self.train_transition_matrix()
        return (self.Markov_states,self.transition_matrix)

    def SAA(self):
        """Use K-means method to discretize the Markovian process."""
        from sklearn.cluster import KMeans
        if self.int_flag == 0:
            labels = numpy.zeros(self.n_samples,dtype=int)
        self._initialize_matrix()
        for t in range(1,self.T):
            kmeans = KMeans(
                n_clusters=self.n_Markov_states[t],
                random_state=0,
            ).fit(self.samples[:,t,:])
            self.Markov_states[t] = kmeans.cluster_centers_
            if self.int_flag == 0:
                labels_new = kmeans.labels_
                counts = numpy.zeros([self.n_Markov_states[t-1],1])
                for i in range(self.n_samples):
                    counts[labels[i]] += 1
                    self.transition_matrix[t][labels[i]][labels_new[i]] += 1
                self.transition_matrix[t] /= counts
                labels = labels_new
        if self.int_flag == 1:
            self.train_transition_matrix()

        return (self.Markov_states,self.transition_matrix)

    def train_transition_matrix(self):
        """Use the generated sample to train the transition matrix by frequency
        counts."""
        if self.int_flag == 1:
            self.round_to_int()
        labels = numpy.zeros([self.n_samples,self.T],dtype=int)
        for t in range(1,self.T):
            self.Markov_states[t] = numpy.unique(self.Markov_states[t],axis=0)
            self.n_Markov_states[t] = len(self.Markov_states[t])
        for t in range(1,self.T):
            dist = numpy.empty([self.n_samples,self.n_Markov_states[t]])
            for idx, markov_state in enumerate(self.Markov_states[t]):
                temp = self.samples[:,t,:] - markov_state
                dist[:,idx] = numpy.sum(temp**2, axis=1)
            labels[:,t] = numpy.argmin(dist,axis=1)
        self._initialize_matrix()
        for k in range(self.n_samples):
            for t in range(1,self.T):
                self.transition_matrix[t][labels[k,t-1]][labels[k,t]] += 1
        for t in range(1,self.T):
            counts = numpy.sum(self.transition_matrix[t], axis=1)
            idx = numpy.where(counts==0)[0]
            if len(idx) > 0:
                self.Markov_states[t-1] = numpy.delete(
                    self.Markov_states[t-1], obj=idx, axis=0
                    )
                self.n_Markov_states[t-1] -= len(idx)
                self.transition_matrix[t-1] = numpy.delete(
                    self.transition_matrix[t-1], obj=idx, axis=1
                    )
                self.transition_matrix[t] = numpy.delete(
                    self.transition_matrix[t], obj=idx, axis=0
                    )
                counts = numpy.delete(counts, obj=idx)
            self.transition_matrix[t] /= counts.reshape(-1,1)

    def write(self, path):
        """Write Markov states and transition matrix to disk."""
        for t in range(self.T):
            pandas.DataFrame(self.Markov_states[t]).to_csv(
            path + "Markov_states_{}.csv".format(t))
            pandas.DataFrame(self.transition_matrix[t]).to_csv(
            path + "transition_matrix_{}.csv".format(t))

    def simulate(self, n_samples):
        """A utility function. Generate a three dimensional array
        (n_samples * T * n_states) representing n_samples number of sample paths.
        Can be used to generate fan plot to compare with the historical data."""
        sim = numpy.empty([n_samples,self.T,self.dim_Markov_states])
        for i in range(n_samples):
            state = 0
            random_state = numpy.random.RandomState(i)
            for t in range(self.T):
                state = random_state.choice(
                    range(self.n_Markov_states[t]),
                    p=self.transition_matrix[t][state],
                )
                sim[i][t]=self.Markov_states[t][state]
        return sim
