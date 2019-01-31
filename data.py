import json

import pandas as pd
from sklearn.model_selection import train_test_split

# |       train      |  eval |  test |
# |       80%        |  10%  |  10%  |
train_size = 0.8  # of the whole data
eval_size = 0.5  # of eval+test data

schema_dir = './schema/'
schema_defaults_file = schema_dir + 'schema-defaults.json'
schema_full_file = schema_dir + 'schema-full.json'

data_dir = './data/'
dataset_file = data_dir + 'so_survey_results_public.csv'
train_file = data_dir + 'train.csv'
eval_file = data_dir + 'eval.csv'
test_file = data_dir + 'test.csv'

# target, features:name/default_value/unique_values
schema = {}

# train/eval/test
data = {}


def preprocess():
    print('Reading default schema...')
    with open(schema_defaults_file) as f:
        default_schema = json.load(f)

    target = default_schema['target']
    target_name = target['name']
    feature_names = [feature['name'] for feature in default_schema['features']]
    feature_defaults = {feature['name']: feature['default_value'] for feature in default_schema['features']}

    print('Reading data...')
    df = pd.read_csv(dataset_file)
    print('Original dataset size: {}'.format(len(df)))

    df.dropna(subset=[target_name], inplace=True)
    df.fillna(feature_defaults, inplace=True)
    print('Filtered dataset size: {}'.format(len(df)))

    for k, v in default_schema.items():
        schema[k] = v

    print('Target: {}'.format(target_name))
    target_values = df[target_name].unique().tolist()
    target['values'] = target_values
    for value in target_values:
        print('\t{}'.format(value))

    print('Features (total: {})'.format(len(feature_names)))
    print('[ name: default ]')

    for feature in schema['features']:
        values = df[feature['name']].unique().tolist()
        feature['values'] = values
        print('\t{}: {}'.format(feature['name'], feature['default_value']))
    print('Writing schema...')
    with open(schema_full_file, 'w') as f:
        json.dump(schema, f)

    print('Performing data split...')
    eval_test_size = 1.0 - train_size
    train_df, eval_test_df = train_test_split(df, test_size=eval_test_size)
    eval_df, test_df = train_test_split(eval_test_df, test_size=eval_size)
    print('Train size: {}\nEval size: {}\nTest size: {}'.format(len(train_df), len(eval_df), len(test_df)))

    print('Writing train/eval/test data...')
    train_df.to_csv(train_file)
    eval_df.to_csv(eval_file)
    test_df.to_csv(test_file)


def load(load_test=False):
    print('Loading schema...')
    with open(schema_full_file) as f:
        schema_json = json.load(f)
    for k, v in schema_json.items():
        schema[k] = v

    print('Loading train data...')
    data['train'] = pd.read_csv(train_file)
    print('Loading eval data...')
    data['eval'] = pd.read_csv(eval_file)
    if load_test:
        print('Loading test data...')
        data['test'] = pd.read_csv(test_file)


def get_train_data():
    return data['train']


def get_eval_data():
    return data['eval']


def get_test_data():
    return data['test']


def get_target_name():
    return schema['target']['name']


def get_target_values():
    return schema['target']['values']


def get_feature_names():
    return [feature['name'] for feature in schema['features']]


def get_feature_values(feature_name):
    for feature in schema['features']:
        if feature['name'] == feature_name:
            return [value for value in feature['values']]
    raise Exception('Feature {} not found'.format(feature_name))


if __name__ == '__main__':
    preprocess()
