import numpy as np

from interactive_naive_bayes.naive_bayes.classifier import train


def test_train():
    categories = np.array([1, 1, 0, 0, 0, 2, 2, 2, 2, 2])
    documents = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
            [2, 1, 2],
            [1, 2, 2],
        ]
    )

    model = train(categories, documents)

    np.testing.assert_allclose(
        model.prior,
        np.array([0.3, 0.2, 0.5]),
    )

    np.testing.assert_allclose(
        model.likelihood,
        np.array([[0.375, 0.25, 0.375], [0.4, 0.4, 0.2], [0.32, 0.32, 0.36]]),
    )
