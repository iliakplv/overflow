import tensorflow as tf

import data

model_dir = './model/'


def create_input_fn(df):
    x = df[data.get_feature_names()]
    y = df[data.get_target_name()]

    return tf.estimator.inputs.pandas_input_fn(
        x,
        y,
        batch_size=32,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=4,
        target_column=data.get_target_name()
    )


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

    classifier.train(create_input_fn(data.get_train_data()))

    result = classifier.evaluate(data.get_eval_data())

    print(result)


if __name__ == '__main__':
    data.load(load_test=False)
    train_evaluate()
