# -*- coding: utf-8 -*-
import alpaca_trade_api as tradeapi
import datetime
import time
import threading
import warnings
from pytz import timezone
from collections import deque
import random
from pickle import dump,load

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import preprocessing as spp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from os import path

import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.metrics import RootMeanSquaredError
from keras.metrics import MeanSquaredError

import logging

#TODO: Fill in missing days(weekends, holidays) with interpolated results
# should remove errors due to variable time differentials between observations.
#TODO: Reframe data as deltas from open. could be less consistent but wont hit consistent sells.
#TODO: Handle Connection loss by reconnection attempt.
#TODO: Handle termination and restarting by checking for existence of NNfiles
#TODO: Convert to Alpaca bars.df instead of bars_to_data_frame
class AutoTrader:
    def __init__(self):
       warnings.filterwarnings("ignore")
       gmt = time.gmtime(time.time())
       logging.basicConfig(filename=str.format('Info_{}_{}_{}.log',gmt.tm_mon,gmt.tm_mday,gmt.tm_year),level=logging.INFO)
       self.api = tradeapi.REST()
       # API datetimes will match this format.
       self.api_time_format = '%Y-%m-%dT%H:%M:%S.%f%z'
       self.window_size = 100
       self.scaler = {}
       
       # self.random_forest_model = {}
       # self.rf_score = {}
       # self.linear_model = {}
       # self.linear_score = {}
       self.n = 10
       
    
    def run(self):
        print('Performing First Time Training')
        logging.info('\n----------------------------------------\nPerforming First Time Training: t = ' + self.string_time())
        logging.info('Canceling Orders: t = ' + self.string_time())
        self.api.cancel_all_orders()
        
        if not path.exists('Network'):
            #if the universe file does not exist yet create a universe file
            #containing a list of symbols trading between 5 and 25 USD.
            #ISSUE:this is very slow and prediction results are poor.
            symbols = list()
            if not path.exists('universe'):
                logging.info('No Universe File Found, Creating Universe:\n')
                logging.info('Fetching Assets: t = '+ self.string_time())
                assets = self.api.list_assets(status = 'active')
                logging.info('Assets Fetched: t = ' + self.string_time())            
                for asset in assets:              
                    if asset.tradable == True:
                        symbols.append(asset.symbol)
                #check last price to filter to only tradeable assets that fall within our price range
                logging.info('Checking Asset Trade Range: t = ' + self.string_time())
                self.data_frame = self.get_bar_frame(symbols)
                logging.info('Data Fetched: t = ' + self.string_time())
                self.data_frame = self.data_frame.sort_index().iloc[-self.window_size:]
                #drop the incomplete bar containing today's data
                self.data_frame = self.data_frame.loc[pd.Timestamp.today(tz ='EST').floor('D')]
                logging.info('Data Sorted: t = ' + self.string_time())
                self.data_frame = self.data_frame.interpolate(method = 'time')
                self.data_frame = self.data_frame.bfill()
    
                pop_indx = []
                suffixes = {'_o','_h','_l','_c','_v'}
                # names = {'open','high','low','close','volume'}
                for symbol in symbols:
                    #check if symbol has any data
                    for suffix in suffixes:
                        if not pop_indx.__contains__(symbol):
                            if not symbol+suffix in self.data_frame.columns:
                                pop_indx.append(symbol)
                            elif not suffix.startswith('_v'):
                                closes = self.data_frame.loc[:,symbol + suffix].fillna(0)
                                if closes.isna().sum()>0:
                                    pop_indx.append(symbol)
                                elif closes.gt(25).sum()>0:
                                    pop_indx.append(symbol)
                                elif closes.lt(5).sum()>0:
                                    pop_indx.append(symbol)
                logging.info('Symbols outside range identified: Count = ' + str(len(pop_indx)))
                for symbol in pop_indx:
                    symbols.remove(symbol)
                    for suffix in suffixes:
                        if symbol+suffix in self.data_frame.columns:
                            self.data_frame.pop(symbol + suffix)
                logging.info('Symbols removed: t = ' + self.string_time())
    
                self.universe_file = open('universe','w')
                for symbol in symbols:
                    self.universe_file.write(symbol+ '\n')
                self.universe_file.close()
            else:
                self.symbols = self.read_universe()
                #convert bar entity data into raw numerical data
                logging.info('Fetching Data For Training: t = ' + self.string_time())
                self.data_frame = self.get_bar_frame(self.symbols)
    
                logging.info('Training Data Fetched: t = ' + self.string_time())
                self.data_frame = self.data_frame.sort_index() #.iloc[-self.window_size:]
                logging.info('Training Data Sorted: t =' + self.string_time())
                
            
            
            self.data_frame = self.preprocess(self.data_frame,True)
            
            dump(self.scaler,open('scaler.pkl','wb'))
            # Train linear and random forest models for each symbol.
            # for symbol in self.symbols:     
            #     self.train_model(symbol)
            Y = self.data_frame.loc[:,self.data_frame.iloc[0].index.map(lambda t: t.endswith('_h') or t.endswith('_l'))]    
            self.time_data_generator = TimeseriesGenerator(self.data_frame.to_numpy(), Y.to_numpy(),
                 	length=self.n, sampling_rate=1, batch_size=10)
            self.train_neural_network(self.time_data_generator)
        else:
            self.neural_network = keras.models.load_model('Network')
            self.symbols = self.read_universe()
            self.scaler = load(open('scaler.pkl','rb'))
            
            
        while True:
            tAMO = threading.Thread(target = self.await_market_open())
            tAMO.start()
            tAMO.join()
            time.sleep(60*60*1)
            # Check if account is restricted from trading.
            if self.api.get_account().trading_blocked:
               print('Account is currently restricted from trading!')
               #TODO:should break out here if account blocked no point in trying ot trade.
            
            today = pd.Timestamp.today(tz = 'EST').floor('D')
            #the following should execute only once the market is open
            
            
            prices = self.get_bar_frame(symbols = self.symbols, window_size = 11).loc[:today]
            #process prediction data, dropping today's incomplete bar 
            pred = self.timeseries_prediction(prices)
            #get our current positions
            logging.info('Getting Positions: t = ' + self.string_time())
            positions = self.api.list_positions()
            orders = self.api.list_orders('closed',after = today.strftime(self.api_time_format)[:-2]+':00')
            #if we havent made any orders today. 
            #could cause issues with manual trading or other scripts/bots
            if len(orders)<=0:
                for position in positions:
                    symbol = position.symbol
                    qty = int(position.qty)
                    #multiply the last trading day's high&low by the fractional 
                    # predicted gains or losses to obtain expected high and low
                    high = prices.iloc[-1].loc[symbol+'_h']*(1+pred.loc[0, symbol + '_h'])
                    low = prices.iloc[-1].loc[symbol+'_l']*(1+pred.loc[0, symbol + '_l'])
                    
                    if high<=low:
                        print(str.format('Sym:{} high({})<=low({})', symbol,high,low))
                        self.api.submit_order(symbol,qty,'sell','limit','day',high)
                    else:
                        #3.	Place OCO orders 15 minutes* after market open on current positions based on estimated H/L.
                        self.api.submit_order(symbol = symbol, qty = qty, side = 'sell', type = 'limit', time_in_force ='day', order_class = 'oco', take_profit = {"limit_price":high},stop_loss = {"stop_price":low})

            #4. every minute while the market is open,from approximately midday until 15 minutes before 
            # market close, predict gains using today's data and create a queue
            # of symbols in order of predicted gains.
                # if we have more than 5% of our equity as available cash, make a
            # limit order for the next symbol in the queue for <%5 of our equity. 
            tAMD = threading.Thread(target = self.await_midday())
            tAMD.start()
            tAMD.join()
            clock = self.api.get_clock()
            next_close = clock.next_close
            while pd.Timestamp.now(tz='EST')<(next_close-pd.Timedelta(15,'min')).tz_convert('EST'):
                
                account = self.api.get_account()
                #set our maximum buy order value to 5% of our total equity
                self.MaxOrderCost = float(account.equity) * 0.05
                
                cash = float(account.cash) 
                orders = self.api.list_orders()
                for order in orders:
                    if order.side == 'buy':
                        cash = cash - (float(order.limit_price)*int(order.qty))
                if cash>=self.MaxOrderCost:
                    prices.append(self.get_bar_frame(self.symbols, window_size=1).loc[today])
                    pred = self.timeseries_prediction(prices)
                    queue = deque(pred.loc[:,pred.loc[0].index.map(lambda t: t.endswith('_h'))].sort_values(by=0,axis=1).columns.to_numpy(copy = True))
                    while cash>=self.MaxOrderCost:
                        symbol = queue.pop()[:-2]
                        price = prices.loc[today].loc[symbol+'_c']
                        qty = (self.MaxOrderCost//price)
                        self.api.submit_order(symbol=symbol,qty = qty,side = 'buy',type = 'limit',time_in_force = 'day',limit_price = price)
                        #adjust cash for new open order
                        cash  = cash-price*qty
                        #remove data
                    prices = prices.loc[:today]
                time.sleep(60)
            #5.	Cancel open orders 15 minutes* before market close.
            self.api.cancel_all_orders()
            

    def timeseries_prediction(self, data_frame):
        data = self.preprocess(data_frame)
        targets = data.loc[:,data.iloc[0].index.map(lambda t: t.endswith('_h') or t.endswith('_l'))]
        tdg = TimeseriesGenerator(data.to_numpy(),targets.to_numpy(),10,batch_size=10)
        pred = pd.DataFrame(self.neural_network.predict(tdg))
        pred.set_axis(targets.columns, axis = 'columns', inplace = True)
        self.inverse_scaling(pred,self.scaler)
        return pred
    
    def scale_data(self, data_frame, scaler, initial = False):
        logging.info('Scaling Data t = '+ self.string_time())
        for data in data_frame:
            if initial:
                if not data.startswith('t_'):
                    scaler[data] = spp.StandardScaler()
                    scaled = scaler[data].fit_transform(np.array(data_frame.loc[:,data]).reshape(-1,1))
                    index = 0
                    for date in data_frame.index:
                        data_frame.loc[date,data] = scaled[index][0]
                        index+=1
            else:
                if not data.startswith('t_'):
                    scaled = scaler[data].transform(np.array(data_frame.loc[:,data]).reshape(-1,1))
                    index = 0
                    for date in data_frame.index:
                        data_frame.loc[date,data] = scaled[index][0]
                        index+=1
        return data_frame
        logging.info('Data Normalized: t = '+self.string_time())
    
    #takes an integer indexed data frame and returns that data frame unscaled
    def inverse_scaling(self,data_frame,scaler):
        logging.info('Unscaling Data t = '+ self.string_time())
        for data in data_frame:
            if not data.startswith('t_'):
                scaled = scaler[data].inverse_transform(np.array(data_frame.loc[:,data]).reshape(-1,1))
                for i in range(len(scaled)):
                    data_frame.loc[i,data] = scaled[i][0]
                    
        

    def train_model(self, symbol,  linear = True, random_forest = True):
        logging.info(str.format('training models for symbol:{}',symbol))
        X = self.n_to_1_mapping(5, self.data_frame.loc[:,symbol+'_c':symbol+'_v'])
        self.train_model_X(X,symbol, linear, random_forest)
        
        
    def train_model_X(self, X, symbol, linear = True, random_forest = True):
        Y = self.data_frame.iloc[self.n:].loc[:,symbol + '_h':symbol + '_l']
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25)
        
        if linear:
            print('-------------------------------------')
            print('\nLinear Regression')
            linreg = LinearRegression()
            self.linear_model[symbol] = linreg.fit(X_train,Y_train)
            print(str.format("R^2 model:{}",self.linear_model[symbol].score(X_train,Y_train)))            
            self.linear_score[symbol] = self.linear_model[symbol].score(X_test,Y_test)
            print(str.format("R^2 test:{}",self.linear_score[symbol]))
            self.MSE(self.linear_model[symbol],X_test,Y_test)
        if random_forest:    
            print('-------------------------------------')
            print('\nRandom Forest Regression')
            rf = RandomForestRegressor()
            self.random_forest_model[symbol] = rf.fit(X_train,Y_train)
            print(str.format("R^2 model:{}",self.linear_model[symbol].score(X_train,Y_train)))
            self.rf_score[symbol] = self.random_forest_model[symbol].score(X_test,Y_test)
            print(str.format("R^2 test:{}",self.rf_score[symbol]))
            self.MSE(self.random_forest_model[symbol],X_test,Y_test)
        
        
    #obtain OHLCV bar data for securities returns a DataFrame for the past 
    #{window_size} {bar_length} indexed by symbol and day
    def get_bar_frame(self, symbols, algo_time = None, window_size = None, bar_length = 'day'):
        data_frame = pd.DataFrame()
        if window_size == None:
            window_size = self.window_size
        if not isinstance(bar_length,str):
            raise ValueError('bar_length must be a string.')
        index = 0
        batch_size = 200
        formatted_time = 0
        if algo_time is not None:
            # Convert the time to something compatable with the Alpaca API.
            formatted_time = algo_time.date().strftime(self.api_time_format[:-2]+':00')
        else:
            formatted_time = self.api.get_clock().timestamp.to_pydatetime().astimezone(timezone('EST'))     
        delta = pd.Timedelta(window_size,'D')
        logging.info('Getting Bars: t = ' + self.string_time())
        while index < len(symbols):
            symbol_batch = symbols[index:index+batch_size]
            logging.info(str.format('Getting Bars for indicies {}:{} t = {}',index,index+batch_size,self.string_time()))
            # Retrieve data for this batch of symbols
            bars = self.api.get_barset(
                symbols=symbol_batch,
                timeframe=bar_length,
                limit= window_size,
                end=formatted_time,
                start=(formatted_time - delta)
                )
            logging.info('Bars Recieved: t = ' + self.string_time())
            index+=batch_size
            #start threads here
            data_frame = data_frame.join(self.bars_to_data_frame(bars),how='outer')
            #join threads here
        return data_frame
    
    # Wait for market to open.
    # Checks the clock every minute while the market is not open.
    def await_market_open(self):
        clock = self.api.get_clock()
        openingTime = clock.next_open.astimezone('EST')
        while pd.Timestamp.now('EST')<=openingTime:
            currTime = pd.Timestamp.now('EST')
            timeToOpen = (openingTime - currTime) / 60
            print(str(timeToOpen.round('min') + " til market open."))
            time.sleep(60)
            
        
    def await_midday(self):
        today = pd.Timestamp.today()
        offset = pd.Timedelta(random.randint(-30,30),'m')
        while (pd.Timestamp.now('EST')<=(pd.Timestamp(today.year,today.month,today.day,12).tz_localize('EST')+offset)):
            time.sleep(60)
            
    #convert bars to a time indexed dataframe
    def bars_to_data_frame(self, bars):
        temp_data_frame = pd.DataFrame()
        for symbol in bars:
            for index in range(0,len(bars[symbol])):
                if not isinstance(bars[symbol][index],tradeapi.entity.Bar):
                    raise ValueError(str.format('Object:{} at [{}][{}] not an instance of {}',bars[symbol][index],symbol,index,tradeapi.entity.Bar))
                date = bars[symbol][index].t
                temp_data_frame.at[date,symbol + '_c'] = bars[symbol][index].c
                temp_data_frame.at[date,symbol + '_h'] = bars[symbol][index].h
                temp_data_frame.at[date,symbol + '_l'] = bars[symbol][index].l
                temp_data_frame.at[date,symbol + '_o'] = bars[symbol][index].o
                temp_data_frame.at[date,symbol + '_v'] = bars[symbol][index].v
        return temp_data_frame

    def string_time(self):
        gmt = time.gmtime(time.time())
        return str.format("{}:{}:{}",gmt.tm_hour,gmt.tm_min,gmt.tm_sec)            
            
    # takes a DataFrame 'data_frame' and calculates a DataFrame 'X' where
    # X[n] = (data_frame[n]-data_frame[n-1])/data_frame[n-1]
    def as_deltas(self, data_frame):
        dt_index = data_frame.iloc[:-1].index
        int_index = pd.RangeIndex(stop = len(data_frame)-1)
        #recalculate data as fractional gain/loss
        A = data_frame.iloc[1:].set_index(int_index)
        A_step_back = data_frame.iloc[:-1].set_index(int_index)
        return A.sub(A_step_back,axis = 1).div(A_step_back).set_index(dt_index)
    
    def MSE(self, model, X_test, Y_test):
        Y_pred = model.predict(X_test)
        Y_delta_square = np.square(np.subtract(Y_test, Y_pred))
        mean_square_error = np.sum(Y_delta_square)/len(Y_pred)
        print(str.format('MSE: {}', mean_square_error))
        return mean_square_error
  
    def train_neural_network(self, generator, epochs = 10):
        self.neural_network = Sequential()
        self.neural_network.add(LSTM(10,input_shape = (10,2530)))
        self.neural_network.add(Dense(10, activation = 'relu'))
        self.neural_network.add(Dense(1010, activation = 'relu'))
        self.neural_network.compile('adam',loss = 'mse', metrics = [MeanSquaredError(),RootMeanSquaredError()])
        self.neural_network.fit(generator, steps_per_epoch=len(generator), epochs = epochs, use_multiprocessing=True)
        self.neural_network.save('Network')
    
    def read_universe(self):
        symbols = list()
        self.universe_file = open('universe','r')
        for line in self.universe_file:
            symbols.append(line.strip())
        self.universe_file.close()
        return symbols


    def preprocess(self, data_frame, initial = False):
        data_frame = data_frame.interpolate(method = 'time')
        data_frame = data_frame.bfill()
        data_frame = self.as_deltas(data_frame)
        #Convert weekday into One-Hot categories
        oneHotEncoder = OneHotEncoder(categories= 'auto')
        time_data_frame =pd.DataFrame(
                oneHotEncoder.fit_transform(np.array(data_frame.index.weekday).reshape(-1,1)).toarray()).set_index(data_frame.index)
        time_data_frame = time_data_frame.add_prefix('t_')
        data_frame = time_data_frame.join(data_frame)
        #Scale data in columns
        data_frame = self.scale_data(data_frame,self.scaler,initial)
        return data_frame
    