import os

import numpy as np

from interactive_naive_bayes.naive_bayes.preprocessing import preprocess


def test_preprocess():
    processed = preprocess(get_test_data_path())

    assert np.array_equal(processed.categories, np.array([0, 1, 2, 3, 4, 5, 6, 7, 7]))
    assert len(processed.vocabulary) == 3


def test_preprocess_with_mask():
    processed = preprocess(get_test_data_path(), word_mask=("banana",))

    assert np.array_equal(processed.categories, np.array([0, 1, 2, 3, 4, 5, 6, 7, 7]))
    assert len(processed.vocabulary) == 2


def get_test_data_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "test.csv")
