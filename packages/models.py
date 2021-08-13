"""
TODO

------------------------------------------------------------------------------------------------------------
1. In prepare_for_model add functionality to the end_date and num_days params.
2. Implement the add_TA function (Eventually scale it so you can pass in specific indicators as param)
3. Fix generate_model function to implement the two helper functions above it
------------------------------------------------------------------------------------------------------------


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

def prepare_for_model(df, end_date, num_days):
    df = df[['Close', 'QuoteVolume', 'Open']]
    df['Open'] = df['Open'].shift(-1)
    df.rename(columns={'Open': 'Target', 'QuoteVolume': 'Volume'})


def add_TA(df, inplace=True):
    pass


def generate_model(asset_name, df, end_date, num_days, force_overwrite=False):

    # Consider either leaving this, or just prepping for model beforehand
    prepped_df = prepare_for_model(df, end_date, num_days)


    parent_path = os.path.join('models', asset_name + '_' + end_date + "-linear-" + str(num_days))

    if os.path.isfile(os.path.join(parent_path, 'completed.txt')):
        print("This set of models is completely trained already.")
        print("Set force_overwrite to True if you want to overwrite it.")
    else:
        os.makedirs(parent_path, exist_ok=True)
        shutil.rmtree(parent_path)

    os.makedirs(parent_path, exist_ok=force_overwrite)

    try:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            list(tqdm(pool.imap(generate_model_helper, product([asset_name], [prepped_df], [end_date], [num_days], [parent_path])), total=len(prepped_df)))

        with open(os.path.join(parent_path, 'completed.txt'), 'w') as file:
            file.write("This set of models is completely trained.")

    except Exception as e:
        print("Error:", e)
        print("Deleting new directory at:", parent_path)
        print("An error occured.")
        shutil.rmtree(parent_path)
        exit()

def generate_model_helper(p):
    asset_name, prepped_df, end_date, num_days, parent_path = p
    path_to_save = os.path.join(parent_path, asset_name)

    model = LinearRegression()
    model.fit(X, y)
    pickle.dump(model, open(path_to_save + ".pkl", 'wb'))