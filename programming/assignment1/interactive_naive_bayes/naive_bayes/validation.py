import time
from typing import Callable

import numpy as np
import numpy.typing as npt

from interactive_naive_bayes.naive_bayes.classifier import (
    Category,
    Count,
    Model,
    predict,
    train,
)


def validate(
    folds: int,
    categories: npt.NDArray[Category],
    documents: npt.NDArray[Count],
    smoothing: npt.NDArray[Count],
    on_progress: Callable[[float], None] = lambda _: None,
):
    assert len(documents) == len(categories)

    indices = np.arange(len(categories))
    np.random.shuffle(indices)
    groups = np.array_split(indices, folds)

    accuracies: dict[float, Model] = {}

    for idx, group in enumerate(groups):
        st = time.time()
        test_categories = categories[group]
        test_documents = documents[group]

        train_categories = np.delete(categories, group)
        train_documents = np.delete(documents, group, axis=0)

        model = train(train_categories, train_documents, smoothing)
        accuracy = _test(test_categories, test_documents, model)

        accuracies[accuracy] = model

        on_progress((idx + 1) / len(groups))

    best_accuracy = max(accuracies.keys())

    return accuracies[best_accuracy], best_accuracy


def _test(
    categories: npt.NDArray[Category], documents: npt.NDArray[Count], model
) -> float:
    correct = 0
    for category, document in zip(categories, documents):
        (result, _) = predict(document, model)
        if result == category:
            correct += 1

    return correct / len(categories)
