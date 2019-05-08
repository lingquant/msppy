#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:05:15 2019

@author: lingquan
"""

import pandas
import numpy
import itertools
import scipy.stats as stats

items = ['SA50','SA100','RSA50','RSA100','SAA50','TS50','TS100']
results = {item: None for item in items}
for item in items:
    results[item] = pandas.read_csv("./result/{}.csv".format(item),index_col=0).values

comparison = pandas.DataFrame(columns=items,index=items)
for one,two in itertools.product(items,items):
    if one != two:
        result_one = results[one]
        result_two = results[two]
        if numpy.mean(result_one) < numpy.mean(result_two):
            comparison[two][one] = stats.ttest_rel(result_one, result_two).pvalue[0]
        else:
            comparison[one][two] = stats.ttest_rel(result_one, result_two).pvalue[0]
# convert to one-sided p-value
comparison = comparison/2
comparison.style.format("{:.2e}")
comparison.to_csv("./result/comparison.csv")
