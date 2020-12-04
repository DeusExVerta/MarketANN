# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:33:43 2020

@author: zeroa
"""
import AutoTrader
import ctypes

if __name__ == "__main__":
    try:
        at = AutoTrader.AutoTrader()
        at.run()
    except AutoTrader.AutoTrader.AccountRestrictedError as are:
        ctypes.windll.user32.MessageBoxW(0,str.format("{},/n{}",are.message,are.account),"Account Restricted Error")
    except AutoTrader.tradeapi.rest.APIError as ex:
        ctypes.windll.user32.MessageBoxW(0,str(ex),str(ex.__class__.__name__))
    except Exception as ex:
        ctypes.windll.user32.MessageBoxW(0,str(ex),str(ex.__class__.__name__))
    finally:
        ctypes.windll.user32.MessageBoxW(0,"please check your internet connection and account information and try again.","Application Terminated")
        