# -*- coding: utf-8 -*-
import alpaca_trade_api as tradeapi
import time
import threading
import warnings
from collections import deque
import random
from pickle import dump,load

import pandas as pd
import numpy as np

from sklearn import preprocessing as spp
from sklearn.preprocessing import OneHotEncoder

from os import path

import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.metrics import RootMeanSquaredError
from keras.metrics import MeanSquaredError

import logging.config
import logging

import configparser
import os, errno

class AutoTrader:
    def __init__(self):
        warnings.filterwarnings("ignore")
        #setup logging configuration to log all messages.
        logging.config.dictConfig(
            {
                "version": 1,
                'formatters':
                    {
                        'default':
                            {
                                'format':"[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
                            },
                    },
                "handlers":
                    {
                         'f':
                             {
                                 "class" : "logging.handlers.TimedRotatingFileHandler",
                                 "formatter": 'default',
                                 "filename": "Log.log",
                                 "when": "D",
                                 "backupCount": 5
                                 }
                    },
                "root":
                    {
                        "level":logging.NOTSET,
                        "handlers":"f"
                    }
                })
        #parse the config file and get the users keys setting them as
        #environment variables for this process and all children.
        config = configparser.ConfigParser()
        config.read('config.txt')
        if config['APIKEYS']['API_Key_ID'] != '':
            os.environ['APCA_API_KEY_ID'] = config['APIKEYS']['API_Key_ID']
        if config['APIKEYS']['Secret_Key'] != '':
            os.environ['APCA_API_SECRET_KEY'] = config['APIKEYS']['Secret_Key']
        os.environ['APCA_API_BASE_URL'] = config['APIKEYS']['Endpoint']
        #connect to Alpaca and their API
        self.api = tradeapi.REST()
        self.window_size = 1000
        self.seq_length = 10
        self.scaler = None
        #check if the universe file exists
        self.get_universe()
 
    class AccountRestrictedError(Exception):
        # Exception Raised for accounts restricted from trading.
        def __init__(self, account, message):
            self.message = message
            self.account = account
            
    def run(self):
        while True:
            tAMO = threading.Thread(target = self.await_market_open())
            tAMO.start()
            tAMO.join()
            #the following should execute one hour after the market is open unless started during market hours
            
            # Check if account is restricted from trading.
            account = self.api.get_account()
            if account.trading_blocked:
                logging.error('account is currently restricted from trading.')
                raise self.AccountRestrictedError(
                    account,
                    'account is currently restricted from trading.')
            
            today = pd.Timestamp.today(tz = 'EST').floor('D')
            
            prices = self.get_bar_frame(self.symbols, window_size = self.seq_length+1).loc[:today]
            #process prediction data, dropping today's incomplete bar 
            pred = self.timeseries_prediction(prices,self.seq_length+1)
            #get our current positions
            logging.info('Getting Positions')
            positions = self.api.list_positions()
            orders = self.api.list_orders('closed',after = today.isoformat())
            #if we havent made any orders today. place bracket sell orders on our positions.
            #could cause issues with manual trading or other scripts/bots            
            if len(orders)<=0:
                for position in positions:
                    symbol = position.symbol
                    qty = int(position.qty)
                    #multiply the last trading day's high&low by the fractional 
                    # predicted gains or losses to obtain expected high and low
                    high = (prices.iloc[-1].loc[[(symbol,'high')]]*(1+pred.loc[today, [(symbol,'high')]]))[0]
                    low = (prices.iloc[-1].loc[[(symbol,'low')]]*(1+pred.loc[today, [(symbol,'low')]]))[0]
                    
                    if high<=low:
                        logging.error(str.format(
                            'Market Sell Sym:{} [high({})<=low({})]',
                            symbol,high,low,high))
                        self.api.submit_order(symbol,qty,'sell','market','day')
                    else:
                        #3.	Place OCO orders 15 minutes* after market open on current positions based on estimated H/L.
                        logging.info(str.format(
                            'OCO Limit sell, Stop Loss {} limit_price {} stop_price {}',
                            symbol,high,low))
                        self.api.submit_order(symbol, qty,'sell', 'limit', 'day',
                                              order_class = 'oco',
                                              take_profit = {"limit_price":high},
                                              stop_loss = {"stop_price":low}
                                              )
                        
            #4. every minute while the market is open,from midday until 15 minutes before 
            # market close, predict gains using today's data and create a queue
            # of symbols in order of predicted gains.
            # if we have more than 5% of our equity as available cash,
            # check the next symbol in the queue, if we don't have a postion 
            # or open order for that symbol make a limit order for that 
            # symbol for <%5 of our equity.
            tAMD = threading.Thread(target = self.await_midday())
            tAMD.start()
            tAMD.join()
            clock = self.api.get_clock()
            next_close = clock.next_close
            while pd.Timestamp.now(tz='EST')<(next_close-pd.Timedelta(15,'min')).tz_convert('EST'):
                account = self.api.get_account()
                MaxOrderCost = float(account.equity) * 0.05
                cash = self.get_available_cash()
                if cash>=MaxOrderCost:
                    prices.append(self.get_bar_frame(self.symbols, window_size=1).loc[today])
                    pred = self.timeseries_prediction(prices, self.seq_length+1)

                    order_symbols = [order.symbol for order in self.api.list_orders(status = 'all', after = today.isoformat()) if order.side == 'buy']
                    position_symbols = [position.symbol for position in self.api.list_positions()]
                    do_not_buy = list(set(order_symbols)|set(position_symbols))
                    columns = pred.columns.map(lambda t: t[1].startswith('high'))
                    queue = deque(pred.loc[:,columns].sort_values(by=today,axis=1).columns.to_numpy(copy = True))
                    
                    while cash>=MaxOrderCost:
                        symbol = (queue.pop()[0])
                        if not symbol in do_not_buy:
                            price = prices.loc[today,[(symbol,'close')]][0]
                            qty = (MaxOrderCost//price)
                            logging.info(str.format(
                                '\tLimit Buy {} shares of {} limit price = {}',
                                qty,symbol,
                                price
                                ))
                            try:
                                self.api.submit_order(symbol,qty,'buy','limit','day',price)
                                #adjust cash for new open order
                                cash  = cash-(price*qty)
                            except ValueError as ve:
                                logging.error(
                                            str.format("{} : {}, buying {}{}",
                                                       ve.__class__.__name__,
                                                       str(ve),
                                                       qty,symbol,
                                                       price,
                                                       ))
                        if symbol in position_symbols:
                            for order_id in [order.id for order in self.api.list_orders('all',after = today.isoformat()) if order.symbol == symbol]:
                                self.api.cancel_order(order_id)
                        if symbol in order_symbols:
                            for order in self.api.list_orders('open',after = today.isoformat()):
                                if order.symbol == symbol:
                                    order_id = order.id
                                    oprice = float(order.limit_price)
                                    oqty = int(order.qty)
                                    price = prices.loc[today,[(symbol,'close')]][0]
                                    qty = (MaxOrderCost//price)
                                    logging.info(str.format(
                                        '\tUPDATE Limit Buy {} shares {},${} from {} shares {},${}',
                                        qty,symbol,
                                        price,
                                        oqty,symbol,
                                        oprice
                                        ))
                                    try:
                                        self.api.replace_order(order_id,qty,price)
                                        cash = cash + (oprice*oqty) - (price*qty)
                                    except tradeapi.rest.APIError as ex:
                                        logging.error(
                                            str.format("{} : {}, replacing ORD#:{}",
                                                       ex.__class__.__name__,
                                                       str(ex),
                                                       order_id
                                                       ))
                    #remove data
                    prices = prices.loc[:today]
                time.sleep(60)
            #5.	Cancel open orders 15 minutes* before market close.
            logging.info('canceling all orders')
            self.api.cancel_all_orders()
            time.sleep(60*16)

    def get_universe(self):
        #if the universe file does not exist raise FileNotFoundError
        symbols = list()
        if path.exists('universe'):
            self.symbols = self.read_universe()
            self.get_network()
        else:
            logging.error('No Universe File Found!')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'universe')
            # yet create a universe file
            # containing a list of symbols trading between 5 and 25 USD over
            # the last 10 trading days.
            logging.info('Fetching Assets')
            assets = self.api.list_assets(status = 'active')
            logging.info('Assets Fetched')   
            for asset in assets:              
                if asset.tradable == True:
                    symbols.append(asset.symbol)
            #check last price to filter to only tradeable assets that fall
            # within our price range
            logging.info('Checking Asset Trade Range')
            data_frame = self.get_bar_frame(symbols,10)
            logging.info('Data Fetched')
            data_frame = data_frame.sort_index().iloc[-self.window_size:]
            #drop the incomplete bar containing today's data
            data_frame = data_frame.loc[:pd.Timestamp.today(tz ='EST').floor('D')]
            logging.info('Data Sorted')
            data_frame = data_frame.interpolate(method = 'time')
            data_frame = data_frame.bfill()
            
            pop_indx = []
            suffixes = {'open','high','low','close','volume'}
            for symbol in symbols:
                #check if symbol has any data
                for suffix in suffixes:
                    if not pop_indx.__contains__(symbol):
                        if not (symbol,suffix) in data_frame.columns:
                            pop_indx.append(symbol)
                        elif not suffix.startswith('v'):
                            prices = data_frame.loc[:,[(symbol,suffix)]].fillna(0)
                            if prices.isna().sum()>0:
                                pop_indx.append(symbol)
                            elif prices.gt(25).sum()>0:
                                pop_indx.append(symbol)
                            elif prices.lt(5).sum()>0:
                                pop_indx.append(symbol)
            logging.info(str.format(
                'Symbols outside range identified: Count = {}',
                str(len(pop_indx))))
            for symbol in pop_indx:
                symbols.remove(symbol)
                for suffix in suffixes:
                    if (symbol,suffix) in data_frame.columns:
                        data_frame.pop((symbol,suffix))
            logging.info('Symbols removed')

            with open('universe','w') as universe_file:
                for symbol in symbols:
                    universe_file.write(symbol+ '\n')
            self.symbols = symbols
            self.get_network(True)
            
    def get_network(self,retrain = False):
        #create and train or load ANN from file
        if path.exists('Network') and not retrain:
            self.get_scaler()
            self.neural_network = keras.models.load_model('Network')
        else:
            logging.info('Performing First Time Training')
            logging.info('Fetching Data For Training')
            data_frame = self.get_bar_frame(self.symbols)
            logging.info('Training Data Fetched')
            self.history = self.train_neural_network(data_frame)

    def get_scaler(self)
        #check to ensure scaler file exists
        if !path.exists('scaler.pkl')
            logging.error('scaler file not found!')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'scaler.pkl')
     
    def get_available_cash(self):
        #gets and returns the value of liquid cash less pending orders cost
        account = self.api.get_account()
        cash = float(account.cash) 
        orders = self.api.list_orders()
        for order in orders:
            if order.side == 'buy':
                cash = cash - (float(order.limit_price)*int(order.qty))
        return cash
        
    def timeseries_prediction(self, data_frame, window_size):
        #takes a data frame and expected length/window size and returns a 
        #data frame containing the predicted change in high and low prices
        data_frame = self.preprocess(data_frame, window_size, algo_time=data_frame.index[-1])
        tdg = self.create_generator(data_frame)
        pred = pd.DataFrame(self.neural_network.predict(tdg))
        pred.set_axis(data_frame.loc[:,data_frame.columns.map(lambda x: x[1].startswith('high')|x[1].startswith('low'))].columns, axis = 'columns', inplace = True)
        pred = self.inverse_scaling(pred)
        pred.set_axis(data_frame.iloc[self.seq_length:].index, axis = 'index', inplace = True)
        return pred
    
    def create_generator(self, data_frame):
        #create sequences of 10 days woth of data using keras Timeseries Generator
         targets = data_frame.loc[:,data_frame.columns.map(lambda t: t[1].startswith('h') or t[1].startswith('l'))]
         tdg = TimeseriesGenerator(
             data_frame.to_numpy(),
             targets.to_numpy(),
             self.seq_length,
             batch_size=10)
         return tdg
    
    def scale_data(self, data_frame, initial = False):
        #takes a data frame and scales each collumn independently
        #saves the initial scaler so future data can be scaled on the same scale
        logging.info('Scaling Data')
        for data in data_frame:
            if initial:
                self.scaler = {}
                if not data[0]=='day':
                    self.scaler[data] = spp.StandardScaler()
                    scaled = self.scaler[data].fit_transform(np.array(data_frame.loc[:,data]).reshape(-1,1))
                    index = 0
                    for date in data_frame.index:
                        data_frame.loc[date,data] = scaled[index][0]
                        index+=1
                dump(self.scaler,open('scaler.pkl','wb')) 
            else:
                if self.scaler == None:
                    self.scaler = load(open('scaler.pkl','rb'))
                if not data[0]=='day':
                    scaled = self.scaler[data].transform(np.array(data_frame.loc[:,data]).reshape(-1,1))
                    index = 0
                    for date in data_frame.index:
                        data_frame.loc[date,data] = scaled[index][0]
                        index+=1
        logging.info('Data Normalized')
        return data_frame
    
    #takes a scaled data frame and returns that data frame unscaled 
    def inverse_scaling(self,data_frame):
        df = pd.DataFrame()
        logging.info('Unscaling Data')
        if self.scaler == None:
            self.scaler = load(open('scaler.pkl','rb'))
        for data in data_frame:
            if not data[0]=='day':
                scaled = self.scaler[data].inverse_transform(np.array(data_frame.loc[:,data]).reshape(-1,1))
                for i in range(len(scaled)):
                    df.loc[i,data] = scaled[i][0]
        return df
                    
    #obtain OHLCV bar data for securities returns a DataFrame for the past 
    #{window_size} {bar_length} indexed by symbol, type and day
    # performs data aquisition in batches of 200 symbols at a time
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
            formatted_time = algo_time
        else:
            formatted_time = self.api.get_clock().timestamp.astimezone('EST')     
        delta = pd.Timedelta(window_size,'D')
        logging.info('Getting Bars')
        while index < len(symbols):
            symbol_batch = symbols[index:index+batch_size]
            logging.info(str.format('Getting Bars for indicies {}:{}',index,index+batch_size))
            # Retrieve data for this batch of symbols
            bars = self.api.get_barset(
                symbols=symbol_batch,
                timeframe=bar_length,
                limit = window_size,
                end=formatted_time.isoformat(),
                start=(formatted_time - delta).isoformat()
                )
            logging.info('Bars Recieved')
            index+=batch_size
            #start threads here           
            barsdf = bars.df
            data_frame = data_frame.join(barsdf, how='outer')
            #join threads here
        return data_frame
    
    # Wait for market to open.
    # Checks the time every minute while the market is not open.
    def await_market_open(self):
        clock = self.api.get_clock()
        openingTime = clock.next_open.astimezone('EST')
        closingTime = clock.next_close.astimezone('EST')
        if openingTime<closingTime:
            while pd.Timestamp.now('EST')<=(openingTime+pd.Timedelta(15,'m')):
                time.sleep(60)
            logging.info('Market Opened+15 minutes')
        else:
            logging.info(str.format('Await started during market hours next_open = {}, next_close = {}',openingTime,closingTime))
                
    
    #waits for midday +-30 minutes checking the time every minute.
    def await_midday(self):
        today = pd.Timestamp.today()
        offset = pd.Timedelta(random.randint(-30,30),'m')
        while (pd.Timestamp.now('EST')<=(pd.Timestamp(today.year,today.month,today.day,12).tz_localize('EST')+offset)):
            time.sleep(60)
            
    #create and train the neural network initially with an 80/20 training/validation split.
    # input shape (timesteps, 5*Symbols + 7)
    def train_neural_network(self, data_frame, window_size = None, epochs = 10, save_model = True):
        if window_size == None:
            window_size = self.window_size
        data_frame = self.preprocess(data_frame,window_size)
        l = int(len(data_frame)*0.8)
        train = data_frame.iloc[:l]
        validation = data_frame.iloc[l:]
        train_generator = self.create_generator(train)
        val_generator = self.create_generator(validation)
        self.neural_network = Sequential()
        self.neural_network.add(LSTM(64,input_shape = (self.seq_length,2532)))
        self.neural_network.add(Dense(10, activation = 'relu'))
        self.neural_network.add(Dense(1010, activation = 'relu'))
        self.neural_network.compile(
            'adam', loss = 'MSLE',
            metrics = [MeanSquaredError(),RootMeanSquaredError()])
        history = self.neural_network.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs = epochs,
            validation_data = val_generator,
            use_multiprocessing=True)
        if save_model:
            self.neural_network.save('Network')
        return history
    
    #read the symbols from the universe file
    def read_universe(self):
        symbols = list()
        with open('universe','r') as universe_file:
            for line in universe_file:
                symbols.append(line.strip())
        return symbols

    #takes a data frame and runs it through a preprocessing regimen:
        # adding missing days
        # replacing infinite values and empty columns with 0
        # interpolating across missing days
        # backfilling missing values at the start of the data frame
        # recalculating as a percentage change.
        # adding a day of the week signal by one hot encoding.
    def preprocess(self, data_frame, window_size, initial = False, algo_time = None):
        if algo_time == None:
            algo_time = pd.Timestamp.today('America/New_York')
        add_df = pd.DataFrame(index = pd.date_range(start = algo_time - pd.Timedelta(window_size,'D') ,end = algo_time).normalize().difference(data_frame.index), columns = data_frame.columns, dtype = 'float64')
        df = pd.concat([data_frame,add_df])
        df.sort_index(inplace = True)
        df.replace([np.inf,-np.inf], method='bfill',inplace = True)
    
        bad_cols = [col for col in df.columns if df[col].isnull().all()]
        df.loc[:,bad_cols] = 0
        
        df.interpolate(method = 'time', inplace = True)
        df.bfill(inplace = True)
        df = df.pct_change().iloc[1:]
        #Convert weekday into One-Hot categories
        oneHotEncoder = OneHotEncoder(categories= 'auto')
        time_data_frame = pd.DataFrame(
            oneHotEncoder.fit_transform(
                df.index.day_name().to_numpy().reshape(-1,1)).toarray(),
                index = df.index) 
        time_data_frame.columns = pd.MultiIndex.from_arrays(
            [['day','day','day','day','day','day','day'],oneHotEncoder.get_feature_names()])
        df = time_data_frame.join(df)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        #Scale data in columns
        df = self.scale_data(df,initial)
        return df
    

