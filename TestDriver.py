# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:03:33 2020

@author: zeroa
"""
import AutoTrader
import pandas as pd

class AutoTradeDriver:
    def __init__(self):
        self.at=AutoTrader.AutoTrader()
        self.at.run()
        self.symbols = self.at.read_universe()
        
    def test(self):
        if self.at is not None:
            if isinstance(self.at, AutoTrader):
symbols = at.read_universe()
X_val = pd.DataFrame()
X_val = at.get_bar_frame(X_val, symbols, window_size=11)
X_val = X_val.iloc[-11:]
X_val = at.preprocess(X_val,False)
X_val = at.n_to_1_mapping(10, X_val)
pred = at.neural_network.predict(X_val)
                return X_val, pred
        
    