# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:12:10 2020

@author: zeroa
"""
import matplotlib
import pandas as pd
import alpaca_trade_api as atapi

rest= atapi.REST()
barsdf = rest.get_barset(['SPY'], 'day',10).df
barsdf = barsdf.pct_change().iloc[1:]
strtime = barsdf.index[0].strftime("%Y-%m-%d")
histdf = rest.get_portfolio_history(date_start = strtime, timeframe ='1D').df
histdf.index = histdf.index.normalize()



barsv = barsdf.pop(('SPY','volume'))
difdf=pd.concat([barsdf,histdf.loc[:,'equity'].pct_change().iloc[1:]],axis = 1)
difdf.plot.bar()


