import numpy as np

from interactive_naive_bayes.ui import get_word_importance


def test_get_word_importance():
    document = np.array([2, 0, 1])
    category = 0
    likelihood = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    vocabulary = ("a", "b", "c")
    assert tuple(
        get_word_importance(document, category, likelihood, vocabulary, top=2)
    ) == ({"word": "c", "importance": 0.3}, {"word": "a", "importance": 0.1})
