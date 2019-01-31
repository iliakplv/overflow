import json

import pandas as pd
from sklearn.model_selection import train_test_split

# |       train      |  eval |  test |
# |       80%        |  10%  |  10%  |
train_size = 0.8  # of the whole data
eval_size = 0.5  # of eval+test data

schema_path = './schema/'
schema_defaults_file = schema_path + 'schema-defaults.json'
schema_full_file = schema_path + 'schema-full.json'

data_path = './data/'
dataset_file = data_path + 'so_survey_results_public.csv'
train_file = data_path + 'train.csv'
eval_file = data_path + 'eval.csv'
test_file = data_path + 'test.csv'

# target, features:name/default_value/unique_values
schema = {}

# train/eval/test
data = {}


def preprocess():
    print('Reading default schema...')
    with open(schema_defaults_file) as f:
        default_schema = json.load(f)

    target = default_schema['target']['name']
    feature_defaults = {feature['name']: feature['default_value'] for feature in default_schema['features']}

    print('Reading data...')
    df = pd.read_csv(dataset_file)
    print('Original dataset size: {}'.format(len(df)))

    df.dropna(subset=[target], inplace=True)
    df.fillna(feature_defaults, inplace=True)
    print('Filtered dataset size: {}'.format(len(df)))

    # todo create schema

    # train/eval/test split
    eval_test_size = 1.0 - train_size
    train_df, eval_test_df = train_test_split(df, test_size=eval_test_size)
    eval_df, test_df = train_test_split(eval_test_df, test_size=eval_size)
    print('Train size: {}\nEval size: {}\nTest size: {}'.format(len(train_df), len(eval_df), len(test_df)))

    print('Writing train/eval/test data...')

    train_df.to_csv(train_file)
    eval_df.to_csv(eval_file)
    test_df.to_csv(test_file)


def load():
    # todo load schema
    train_df = pd.read_csv(train_file)
    eval_df = pd.read_csv(eval_file)
    test_df = pd.read_csv(test_file)
    data['train'] = train_df
    data['eval'] = eval_df
    data['test'] = test_df


def get_train_data():
    return data['train']


def get_eval_data():
    return data['eval']


def get_test_data():
    return data['test']


def get_target():
    return schema['target']['name']


def get_feature_names():
    return [feature['name'] for feature in schema['features']]


def get_feature_defaults():
    return [feature['default_value'] for feature in schema['features']]


if __name__ == '__main__':
    preprocess()
