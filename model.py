import tensorflow as tf
import pandas as pd

dataset_file = './data/so_survey_results_public.csv'

# TODO
feature_names = ['Hobby', 'OpenSource', 'Country']
record_defaults = [['No'], ['No']]
target = ''


def input_fn(file_path, perform_shuffle=False, repeat_count=1, batch_size=32):
    def decode_csv(line):
        # TODO features and target
        parsed_line = tf.decode_csv(line, record_defaults)
        label = parsed_line[-1:]
        del parsed_line[-1]
        features = parsed_line
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)
               .skip(1)  # skip header
               .map(decode_csv, num_parallel_calls=4))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def input_fn_pandas():
    df = pd.read_csv(dataset_file)
    x = df[feature_names]
    y = df[target]

    return tf.estimator.inputs.pandas_input_fn(
        x,
        y,
        batch_size=32,
        num_epochs=1,
        shuffle=None,
        queue_capacity=1000,
        num_threads=4,
        target_column=target
    )
