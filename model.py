import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from schema import feature_defaults
from schema import feature_names
from schema import target

data_path = './data/'
dataset_file = data_path + 'so_survey_results_public.csv'
train_file = data_path + 'train.csv'
eval_file = data_path + 'eval.csv'
test_file = data_path + 'test.csv'

# |       train      |  eval |  test |
# |       80%        |  10%  |  10%  |
train_size = 0.8  # of the whole data
eval_size = 0.5  # of eval+test data

data = {}


def init_data(train_df, eval_df, test_df):
    print('Train size: {}\nEval size: {}\nTest size: {}'.format(len(train_df), len(eval_df), len(test_df)))
    data['train'] = train_df
    data['eval'] = eval_df
    data['test'] = test_df


def preprocess_data(overwrite=False):
    if not overwrite and (os.path.isfile(train_file) and os.path.isfile(eval_file) and os.path.isfile(test_file)):
        print('Found train/eval/test data')
        train_df = pd.read_csv(train_file)
        eval_df = pd.read_csv(eval_file)
        test_df = pd.read_csv(test_file)
        init_data(train_df, eval_df, test_df)
        return

    print('Reading data...')
    df = pd.read_csv(dataset_file)
    print('Original dataset size: {}'.format(len(df)))

    # drop missing targets
    df.dropna(subset=[target], inplace=True)

    # impute defaults
    df.fillna(feature_defaults, inplace=True)

    print('Filtered dataset size: {}'.format(len(df)))

    # train/eval/test split
    eval_test_size = 1.0 - train_size
    train_df, eval_test_df = train_test_split(df, test_size=eval_test_size)
    eval_df, test_df = train_test_split(eval_test_df, test_size=eval_size)

    init_data(train_df, eval_df, test_df)

    # write to files
    train_df.to_csv(train_file)
    eval_df.to_csv(eval_file)
    test_df.to_csv(test_file)


def input_fn(dataset_key):
    df = data[dataset_key]

    x = df[feature_names]
    y = df[target]

    return tf.estimator.inputs.pandas_input_fn(
        x,
        y,
        batch_size=32,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=4,
        target_column=target
    )


if __name__ == '__main__':
    preprocess_data()
