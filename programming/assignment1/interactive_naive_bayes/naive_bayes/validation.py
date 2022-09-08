import numpy as np
import numpy.typing as npt

from interactive_naive_bayes.naive_bayes.classifier import (
    Category,
    HasWord,
    Model,
    predict,
    train,
)


def validate(folds: int, targets: npt.NDArray[Category], samples: npt.NDArray[HasWord]):
    assert len(targets) == len(samples)

    indices = np.arange(len(targets))
    np.random.shuffle(indices)
    groups = np.array_split(indices, folds)

    accuracies = []
    for group in groups:
        test_targets = targets[group]
        test_samples = samples[group]

        train_targets = np.delete(targets, group)
        train_samples = np.delete(samples, group, axis=0)

        model = train(train_targets, train_samples)
        accuracy = test(test_targets, test_samples, model)

        accuracies.append(accuracy)

    # TODO: Increase performance
    print(accuracies)


# TODO: Increase performance
def test(
    targets: npt.NDArray[Category], samples: npt.NDArray[HasWord], model: Model
) -> float:
    correct = 0
    for sample, target in zip(samples, targets):
        if predict(sample, model) == target:
            correct += 1

    return correct / len(samples)
