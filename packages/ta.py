import pandas as pd
import numpy as np
import talib

def add_ATR(df, timeperiod=14, inplace=False):
    if inplace:
        df['ATR_{}'.format(timeperiod)] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=timeperiod)
    else:
        temp = df.copy()
        temp['ATR_{}'.format(timeperiod)] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=timeperiod)
        return temp

def add_RSI(df, timeperiod=14, inplace=False):
    if inplace:
        df['RSI_{}'.format(timeperiod)] = talib.RSI(df['Close'], timeperiod=timeperiod)
    else:
        temp = df.copy()
        temp['RSI_{}'.format(timeperiod)] = talib.RSI(df['Close'], timeperiod=timeperiod)
        return temp

def add_SMA(df, timeperiod=200, inplace=False):
    if inplace:
        df['SMA_{}'.format(timeperiod)] = talib.SMA(df['Close'], timeperiod)
    else:
        temp = df.copy()
        temp['SMA_{}'.format(timeperiod)] = talib.SMA(df['Close'], timeperiod)
        return temp

def add_EMA(df, timeperiod=9, inplace=False):
    if inplace:
        df['EMA_{}'.format(timeperiod)] = talib.EMA(df['Close'], timeperiod)
    else:
        temp = df.copy()
        temp['EMA_{}'.format(timeperiod)] = talib.EMA(df['Close'], timeperiod)
        return temp

def add_OBV(df, inplace=False):
    if inplace:
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    else:
        temp = df.copy()
        temp['OBV'] = talib.OBV(df['Close'], df['Volume'])
        return temp