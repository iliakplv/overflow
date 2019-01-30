import pandas as pd
import tensorflow as tf

from schema import feature_defaults
from schema import feature_names
from schema import target

dataset_file = './data/so_survey_results_public.csv'


def input_fn():
    df = pd.read_csv(dataset_file)

    df.dropna(subset=[target], inplace=True)
    df.fillna(feature_defaults, inplace=True)

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
    input_fn()
