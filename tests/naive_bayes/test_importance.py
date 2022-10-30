import numpy as np

from interactive_naive_bayes.naive_bayes.importance import adjust_category_smoothing


def test_adjust_smoothing():
    unique_word_count = 3
    old_smoothing = np.ones(unique_word_count)
    target_likelihood = 0.5
    word_idx = 0
    documents = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ]
    )

    new_smoothing = adjust_category_smoothing(
        old_smoothing, target_likelihood, word_idx, documents
    )

    np.testing.assert_allclose(
        new_smoothing,
        np.array([6.0, 1.0, 1.0]),
    )
