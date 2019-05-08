#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""

from statsmodels.tsa.arima_model import ARMA
import pandas
import numpy
import statsmodels.api as sm

prices = pandas.read_csv("prices.csv",parse_dates=['Date'],index_col=0)
tickers = prices.columns[:-2]
prices = prices.resample('W').agg(lambda x:x[-1])
prices.dropna(axis=0, how='any', inplace=True)
rf = prices['^TNX'].values[:-1]
rf /= (52*100)
returns =  prices.iloc[:,:-1].pct_change()[1:]
rm = returns['^GSPC'].values
ri = returns.iloc[:,:-1].values
Ri = ri-rf[:,numpy.newaxis]
Rm = rm-rf
model = sm.OLS(Ri, sm.add_constant(Rm))
results = model.fit()
alpha,beta = results.params
epsilon = numpy.sqrt(Ri.var(axis=0) - beta**2*Rm.var(axis=0))
output = pandas.DataFrame(
    columns=['alpha','beta','epsilon'],
    index = tickers,
    data=numpy.array([alpha,beta,epsilon]).T
)
output.to_csv("coefficients.csv")
from arch.univariate import ARX, GARCH
arx = ARX(rm, lags=1)
arx.volatility = GARCH()
res = arx.fit(disp='off')
pandas.DataFrame(res.params).to_csv("parameters.csv")
