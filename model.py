import time
import tensorflow as tf

import data

models_dir = './models/'

params = {
    'pandas_input_fn': True,
    'classifier_type': 'dnn',  # or 'linear'
    'batch_size': 32,
    'epochs': 1,
    'shuffle_data': False
}


def param(name):
    return params[name]


def input_fn_dataset(df):
    features = df[data.get_feature_names()]
    labels = df[data.get_target_name()]

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if param('shuffle_data'):
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(param('epochs'))
    dataset = dataset.batch(param('batch_size'))
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def input_fn_pandas(df):
    x = df[data.get_feature_names()]
    y = df[data.get_target_name()]

    return tf.estimator.inputs.pandas_input_fn(
        x,
        y,
        batch_size=param('batch_size'),
        num_epochs=param('epochs'),
        shuffle=param('shuffle_data'),
        queue_capacity=1000,
        num_threads=4,
        target_column=data.get_target_name()
    )


def get_input_fn(df):
    if param('pandas_input_fn'):
        return input_fn_pandas(df)
    else:
        return lambda: input_fn_dataset(df)


def get_feature_columns():
    feature_names = data.get_feature_names()
    classifier_type = param('classifier_type')
    if classifier_type == 'linear':
        return [
            tf.feature_column.categorical_column_with_vocabulary_list(name, data.get_feature_values(name))
            for name in feature_names
        ]
    elif classifier_type == 'dnn':
        return [
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(name, data.get_feature_values(name))
            )
            for name in feature_names
        ]
    raise Exception('Unsupported classifier type: {}'.format(classifier_type))


def get_classifier():
    classifier_type = param('classifier_type')
    model_dir = models_dir + classifier_type
    if classifier_type == 'linear':
        return tf.estimator.LinearClassifier(
            feature_columns=get_feature_columns(),
            model_dir=model_dir,
            n_classes=len(data.get_target_values()),
            label_vocabulary=data.get_target_values())
    elif classifier_type == 'dnn':
        return tf.estimator.DNNClassifier(
            feature_columns=get_feature_columns(),
            hidden_units=[10, 10],
            n_classes=len(data.get_target_values()),
            label_vocabulary=data.get_target_values(),
            model_dir=model_dir)
    raise Exception('Unsupported classifier type: {}'.format(classifier_type))


def print_dict(title, d):
    print(title)
    for k, v in d.items():
        print('\t{}: {}'.format(k, v))


def train_evaluate():
    print_dict('Experiment params:', params)

    classifier = get_classifier()

    print('Training ({})...'.format(param('classifier_type')))
    train_start = time.time()
    classifier.train(get_input_fn(data.get_train_data()))
    train_end = time.time()
    print('Training completed in {} seconds'.format(train_end - train_start))

    print('Evaluating...')
    result = classifier.evaluate(get_input_fn(data.get_eval_data()))
    print_dict('Evaluation result:', result)


if __name__ == '__main__':
    data.load(load_test=False)
    train_evaluate()
