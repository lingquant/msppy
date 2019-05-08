#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:05:15 2019

@author: lingquan
"""

import pandas
import matplotlib.pyplot as plt
import numpy
result_1 = pandas.read_csv("/Volumes/My Passport/apaper/hydro_thermal/1/result/MC_result_true.csv",index_col=0)
result_2 = pandas.read_csv("/Volumes/My Passport/apaper/hydro_thermal/2/result/MC_result_true.csv",index_col=0)
TS_100_889 = pandas.read_csv("/Users/lingquan/Documents/TS_100_889/TS_result_true.csv",index_col=0)
TS_100_890 = pandas.read_csv("/Users/lingquan/Documents/TS_100_890/TS_result_true.csv",index_col=0)
TS_10_888 = pandas.read_csv("/Users/lingquan/Documents/TS_10_888/TS_result_true.csv",index_col=0)


#plt.figure()
#plt.hist(list(SA['pv']),bins=100)
#plt.figure()
#plt.hist(list(SA_50['pv']),bins=100)
#plt.figure()
#plt.hist(list(RSA['pv']),bins=100)
#plt.figure()
#plt.hist(list(TS_100_889['pv']),bins=100)
#plt.figure()
#plt.hist(list(TS_10_888['pv']),bins=100)
print([numpy.quantile(result_1,0.1*t) for t in range(1,10)])
print([numpy.quantile(result_2,0.1*t) for t in range(1,10)])
#print([numpy.quantile(RSA,0.1*t) for t in range(1,10)])
print([numpy.quantile(TS_100_889,0.1*t) for t in range(1,10)])
print([numpy.quantile(TS_10_888,0.1*t) for t in range(1,10)])
from MSP.utils.statistics import compute_CI
a,b = compute_CI((result_1-TS_100_889).values,95)
(a+b)/(b-a)