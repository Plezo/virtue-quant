"""
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
from itertools import product
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from packages import ta

def add_TA(df):
    ta.add_ATR(df, inplace=True)
    ta.add_OBV(df, inplace=True)
    return df

def prepare_for_model(df, end_date, num_days):
    end_date = pd.to_datetime(end_date)

    df.drop(columns={'High', 'Low', 'Volume'}, inplace=True)
    df['Open'] = df['Open'].shift(-1)
    df.rename(columns={'Open': 'Target', 'QuoteVolume': 'Volume'}, inplace=True)
    df = df[df.index.year <= end_date.year].tail(num_days)

    return df

def generate_model(asset_name, df, end_date, num_days, force_overwrite=False):
    parent_path = os.path.join('models', asset_name + '_' + end_date + "-linear-" + str(num_days))

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