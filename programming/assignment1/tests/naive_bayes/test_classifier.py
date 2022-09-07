import numpy as np
import pytest
from interactive_naive_bayes.naive_bayes.classifier import Model, predict, train


def test_train():
    targets = np.array([0, 0, 1], dtype=np.ushort)
    samples = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.ushort,
    )

    model = train(targets, samples)

    assert model.prior == pytest.approx({0: 2 / 3, 1: 1 / 3})

    assert model.likelihood[0][0] == pytest.approx({0: 1 / 4, 1: 3 / 4})
    assert model.likelihood[0][1] == pytest.approx({0: 1 / 2, 1: 1 / 2})
    assert model.likelihood[0][2] == pytest.approx({0: 1 / 2, 1: 1 / 2})
    assert model.likelihood[1][0] == pytest.approx({0: 2 / 3, 1: 1 / 3})
    assert model.likelihood[1][1] == pytest.approx({0: 1 / 3, 1: 2 / 3})
    assert model.likelihood[1][2] == pytest.approx({0: 1 / 3, 1: 2 / 3})


def test_predict():
    model = Model(
        prior={0: 2 / 3, 1: 1 / 3},
        likelihood={
            0: {
                0: {0: 1 / 4, 1: 3 / 4},
                1: {0: 1 / 2, 1: 1 / 2},
                2: {0: 1 / 2, 1: 1 / 2},
            },
            1: {
                0: {0: 2 / 3, 1: 1 / 3},
                1: {0: 1 / 3, 1: 2 / 3},
                2: {0: 1 / 3, 1: 2 / 3},
            },
        },
    )

    assert predict(np.array([1, 1, 0]), model) == 0
    assert predict(np.array([1, 0, 1]), model) == 0
    assert predict(np.array([0, 1, 1]), model) == 1
