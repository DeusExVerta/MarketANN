# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:33:43 2020

@author: zeroa
"""
import sys
with open('stdout.txt', 'w') as stdoutfile:
    sys.stdout = stdoutfile
with open('stderr.txt', 'w') as stderrfile:
    sys.stderr = stderrfile

import AutoTrader
import ctypes
import traceback

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
        with open('Traceback.txt','w') as tbfile:
            traceback.print_tb(ex.__traceback__,file = tbfile)
    finally:
        ctypes.windll.user32.MessageBoxW(0,"Please check your internet connection and account information and try running the script again.","Application Terminated")
        