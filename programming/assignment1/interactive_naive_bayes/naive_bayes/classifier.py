import dataclasses

import numpy as np
import numpy.typing as npt

Category = np.ushort
HasWord = np.ushort

Likelihood = dict[Category, dict[int, dict[HasWord, float]]]


@dataclasses.dataclass
class Model:
    prior: dict[Category, float]
    likelihood: Likelihood


def train(targets: npt.NDArray[Category], samples: npt.NDArray[HasWord]) -> Model:
    return Model(prior=get_prior(targets), likelihood=get_likelihood(targets, samples))


def get_prior(targets: npt.NDArray[Category]) -> dict[Category, float]:
    unique, counts = np.unique(targets, return_counts=True)
    return dict(zip(unique, counts / len(targets)))


# TODO: Maybe use numba to speed up this function
def get_likelihood(
    targets: npt.NDArray[Category], samples: npt.NDArray[HasWord]
) -> Likelihood:
    categories = np.unique(targets)
    smoothing_values = np.array([0, 1])

    likelihood: Likelihood = {}
    for category in categories:
        likelihood[category] = {}
        (indices,) = np.nonzero(targets == category)
        for feature in range(samples.shape[1]):
            values = np.append(samples[indices, feature], smoothing_values)
            unique, counts = np.unique(values, return_counts=True)
            likelihood[category][feature] = dict(zip(unique, counts / len(values)))

    return likelihood


def predict(sample: npt.NDArray[HasWord], model: Model) -> Category:
    max_posterior = 0.0
    predicted_category = None
    for category, prior in model.prior.items():
        posterior = prior * np.prod(
            tuple(
                model.likelihood[category][feature][value]
                for feature, value in enumerate(sample)
            )
        )
        if posterior > max_posterior:
            max_posterior = float(posterior)
            predicted_category = category

    assert predicted_category is not None
    return predicted_category
