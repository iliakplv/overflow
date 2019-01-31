import tensorflow as tf

import data


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


def train_evaluate():
    # next_batch = input_fn('train')

    # feature_columns = [tf.feature_column.categorical_column_with_identity(k) for k in feature_names]
    # feature_columns = [tf.feature_column.categorical_column_with_vocabulary_list(k) for k in feature_names]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir=model_path)

    classifier.train(input_fn=lambda: input_fn('train'))

    # evaluate_result = classifier.evaluate(input_fn=lambda: input_fn('eval'))
    #
    # print('Evaluation results')
    # for key in evaluate_result:
    #     print('  {}, was: {}'.format(key, evaluate_result[key]))


if __name__ == '__main__':
    pass
