from typing import Sequence
import dask.dataframe as dd
import tensorflow as tf

def csv_to_input(inpath, column_names: Sequence[str]):
    df = dd.read_csv(inpath).compute()[column_names].fillna('')
    return df.T.values


def load_data(doc_path: str = None, asp_path: str = None):
    """
    Function that takes in a path to the document data and to the aspect data and returns a dictionary
    suitable for fitting the model.

    Args:
        doc_path: path to the document dataset
        asp_path: path to the aspect dataset

    Returns:
        x,y: dictionaries suitable for training and validating models

    """
    if doc_path is not None and asp_path is not None:
        doc_x, doc_y = csv_to_input(doc_path, ['text', 'polarity'])
        *asp_x, asp_y = csv_to_input(asp_path, ['context_left', 'target', 'context_right', 'polarity'])
        x = {'asp': asp_x, 'doc': doc_x}
        y = {'asp': tf.one_hot(asp_y + 1, 3, dtype='int32'), 'doc': tf.one_hot(doc_y + 1, 3, dtype='int32')}
        return x, y

    if doc_path is not None and asp_path is None:
        doc_x, doc_y = csv_to_input(doc_path, ['text', 'polarity'])
        x = {'doc': doc_x}
        y = {'doc': tf.one_hot(doc_y + 1, 3, dtype='int32')}
        return x, y

    if doc_path is None and asp_path is not None:
        *asp_x, asp_y = csv_to_input(asp_path, ['context_left', 'target', 'context_right', 'polarity'])
        x = {'asp': asp_x}
        y = {'asp': tf.one_hot(asp_y + 1, 3, dtype='int32')}
        return x, y