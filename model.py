import time
import tensorflow as tf

import data

model_dir = './model/'

pandas_input_fn = False

batch_size = 32
epochs = 1
shuffle_data = False


def input_fn_dataset(df):
    features = df[data.get_feature_names()]
    labels = df[data.get_target_name()]

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if shuffle_data:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def input_fn_pandas(df):
    x = df[data.get_feature_names()]
    y = df[data.get_target_name()]

    return tf.estimator.inputs.pandas_input_fn(
        x,
        y,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=shuffle_data,
        queue_capacity=1000,
        num_threads=4,
        target_column=data.get_target_name()
    )


def get_input_fn(df):
    if pandas_input_fn:
        return input_fn_pandas(df)
    else:
        return lambda: input_fn_dataset(df)


def print_result(result):
    print('Evaluation result:')
    for k, v in result.items():
        print('\t{}: {}'.format(k, v))


def train_evaluate():
    feature_names = data.get_feature_names()
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(name, data.get_feature_values(name))
        for name in feature_names
    ]

    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                               model_dir=model_dir,
                                               n_classes=len(data.get_target_values()),
                                               label_vocabulary=data.get_target_values())

    print('Training...')
    train_start = time.time()
    classifier.train(get_input_fn(data.get_train_data()))
    train_end = time.time()
    print('Training completed in {} seconds'.format(train_end - train_start))

    print('Evaluating...')
    result = classifier.evaluate(get_input_fn(data.get_eval_data()))
    print_result(result)


if __name__ == '__main__':
    data.load(load_test=False)
    train_evaluate()
