"""
cryptowatch:    https://docs.cryptowat.ch/rest-api/markets/ohlc
coinapi:        https://docs.coinapi.io/#ohlcv
alphavantage:   https://www.alphavantage.co/documentation/

"""

import cryptowatch as cw
import pandas as pd
import config

cw.api_key = config.CRYPTOWATCH_API_KEY

def get_df_from_cryptowatch(exchange='kraken', pair='btcusd'):
    market_list = cw.markets.get("{}:{}".format(exchange, pair), ohlc=True, periods=["1d"]).of_1d

    data = {
        'Date'          : [],
        'Open'          : [], 
        'High'          : [],
        'Low'           : [], 
        'Close'         : [],
        'Volume'        : [], 
        'QuoteVolume'   : []
        }

    for i in market_list:
        for j in range(7):
            data[list(data)[j]].append(i[j])

    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume']).set_index('Date')
    df.index = pd.to_datetime(df.index, unit='s')
    return df