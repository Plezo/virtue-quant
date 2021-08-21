"""
Used to do all things related to models, from preparing the dataframe to generating models.
"""

# import tensorflow as tf
import os
import shutil
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from itertools import product
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from packages import ta

def add_TA(df):
    """Adds technical indicator columns to dataframe

    Args:
        df: The dataframe we're adding the indicators to

    Returns:
        Dataframe passed in with indicators added to it

    """

    ta.add_ATR(df, inplace=True)
    ta.add_OBV(df, inplace=True)
    # ta.add_RSI(df, inplace=True)
    return df

def prepare_for_model(df, day_predicting, num_days, features=['Close', 'Volume'], target='Close'):
    """Formats dataframe for model creation

    Args:
        df: Dataframe to prepare
        day_predicting: The day we are predicting
        num_days: Number of days to train on
        features (optional): Feature columns we want to train
        target (optional): The column used to determine the target values

    Returns:
        Formatted dataframe

    """

    features.insert(0, 'Target')
    day_predicting = pd.to_datetime(day_predicting)
    prepped_df = df.copy()

    prepped_df.insert(0, 'Target', prepped_df[target.title()].shift(-1))
    prepped_df = prepped_df.loc[:, [i for i in prepped_df.columns if i in features]]
    prepped_df.rename(columns={'QuoteVolume': 'Volume'}, inplace=True)
    prepped_df = prepped_df[prepped_df.index < day_predicting].tail(num_days + 1)

    return prepped_df

def get_train_test(df):
    """Splits dataframe into training and testing subsets

    Args:
        df: The dataframe we're getting the train-test from

    Returns:
        Training/Testing predictors and target values

    """

    Xtrain = df.loc[:, df.columns != 'Target'][:-1]
    Xtest = df.loc[:, df.columns != 'Target'].iloc[[-1]]
    ytrain = df[['Target']][:-1]
    ytest = df[['Target']].iloc[[-1]]
    return Xtrain, Xtest, ytrain, ytest