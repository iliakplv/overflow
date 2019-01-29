import tensorflow as tf

dataset_file = './data/so_survey_results_public.csv'

feature_names = ['Hobby', 'OpenSource', 'Country']  # TODO
record_defaults = [['No'], ['No']]  # TODO


def input_fn(file_path, perform_shuffle=False, repeat_count=1, batch_size=32):
    def decode_csv(line):
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
