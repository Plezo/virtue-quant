"""
Used to do all things related to models, from preparing the dataframe to generating models.

TODO

--------------------------------------------------------------------------------------------------------------
1. Implement Predictions function
2. Consider switching to a log regression giving probability of profit 0-1
--------------------------------------------------------------------------------------------------------------

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

def prepare_for_model(df, day_predicting, num_days, target='Close'):
    """Formats dataframe for model creation

    Args:
        df: Dataframe to prepare
        day_predicting: The day we are predicting
        num_days: Number of days to train on
        target (optional): The column used to determine the target values

    Returns:
        Formatted dataframe

    """

    day_predicting = pd.to_datetime(day_predicting)
    prepped_df = df.copy()

    prepped_df.insert(0, 'Target', prepped_df[target.title()].shift(-1))
    prepped_df = prepped_df.loc[:, [i for i in prepped_df.columns if i not in ['High', 'Low', 'Volume', 'Open']]]
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

def generate_model(asset_name, df, day_predicting, num_days: int, force_overwrite=False):
    """Summary line.

    Parameters:
        arg1: 
        arg2: Description of arg2

    Returns:
        Description of return value

    """
    
    parent_path = os.path.join('models', asset_name + '_' + day_predicting + "-linear-" + str(num_days))

    if os.path.isfile(os.path.join(parent_path, 'completed.txt')):
        print("This set of models is completely trained already.")
        print("Set force_overwrite to True if you want to overwrite it.")
    else:
        os.makedirs(parent_path, exist_ok=True)
        shutil.rmtree(parent_path)

    os.makedirs(parent_path, exist_ok=force_overwrite)
    
    model = LinearRegression()
    tqdm(model.fit(df.loc[:, df.columns != 'Target'], df['Target']), total=len(df))
    pickle.dump(model, open(os.path.join(parent_path, asset_name) + ".pkl", 'wb'))

    # try:
    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         list(tqdm(pool.imap(generate_model_helper, product([asset_name], [df], [end_date], [num_days], [parent_path])), total=len(df)))

    #     with open(os.path.join(parent_path, 'completed.txt'), 'w') as file:
    #         file.write("This set of models is completely trained.")

    # except Exception as e:
    #     print("Error:", e)
    #     print("Deleting new directory at:", parent_path)
    #     print("An error occured.")
    #     shutil.rmtree(parent_path)
    #     exit()

# def generate_model_helper(p):
#     asset_name, df, end_date, num_days, parent_path = p
#     path_to_save = os.path.join(parent_path, asset_name)

#     model = LinearRegression()
#     model.fit(df.loc[:, df.columns != 'Target'], df['Target'])
#     pickle.dump(model, open(path_to_save + ".pkl", 'wb'))

def load_model(asset_name, end_date, num_days):
    parent_path = os.path.join('models', asset_name + '_' + end_date + "-linear-" + str(num_days))
    return pickle.load(open(os.path.join(parent_path, asset_name) + ".pkl", 'rb'))

def get_predictions(lrmodel, asset_name, end_date, num_days, message=False):
    parent_path = os.path.join('models', asset_name + '_' + end_date + "-linear-" + str(num_days))

    if os.path.isfile(os.path.join(parent_path, 'predictions.pkl')):
        if message:
            print("Loading saved predictions...")
        return pd.read_pickle(os.path.join(parent_path, 'predictions.pkl'))
    else:
        print("Predictions not detected. Calculating and saving...")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(prediction_helper, product([common_features], investible_universe, [lrmodels], [end_date], [num_days], [pycaret])), total=len(investible_universe)))

        predictions = pd.concat(results, axis=0)
        predictions.to_pickle(os.path.join(parent_path, 'predictions.pkl'))
        return predictions