#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lingquan
"""

import pandas_datareader.data as web
import pandas
import numpy
start = '2016-01-01'
end = '2019-01-01'
get_price = lambda x: web.DataReader(x, 'yahoo', start=start, end=end)['Adj Close']
SP_500 = pandas.read_html('https://en.wikipedia.org/wiki/S%26P_100')
tickers = SP_500[2][1:].iloc[:,0].values
tickers = numpy.append(
    numpy.delete(tickers, numpy.where(numpy.isin(tickers,['DOW','BRK.B']))),
    ['^GSPC','^TNX'],
)
price = get_price(tickers)
price.to_csv("prices.csv")

