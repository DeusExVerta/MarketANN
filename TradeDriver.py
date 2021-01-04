# -*- coding: utf-8 -*-
"""
This project contains software developed by the Apache software foundation and
Alpaca, this software is used in accordance with the Apache License 2.0 

This project is an automated stock trading bot that operates on Alpaca

@author: Gary J Howard
"""
#this file is a driver for AutoTrader 

#redirect stdout and stderr so the script can run as pythonw
#this is neccessary due to a bug with pythonw
import sys
with open('stdout.txt', 'w') as stdoutfile:
    sys.stdout = stdoutfile
    with open('stderr.txt', 'w') as stderrfile:
        sys.stderr = stderrfile
    
        import AutoTrader
        import pyautogui as gui
        import traceback
        
        if __name__ == "__main__":
            try:
                at = AutoTrader.AutoTrader()
                at.run()
            except AutoTrader.AutoTrader.AccountRestrictedError as are:
                gui.alert(str.format("{},/n{}",are.message,are.account),"Account Restricted Error")
            except AutoTrader.tradeapi.rest.APIError as ex:
                gui.alert(str(ex),str(ex.__class__.__name__))
            except Exception as ex:
                gui.alert(str(ex),str(ex.__class__.__name__))
                with open('Traceback.txt','w') as tbfile:
                    traceback.print_tb(ex.__traceback__,file = tbfile)
            finally:
                gui.alert("Please check your internet connection and account information and try running the script again.","Application Terminated")
                    
        