import dataclasses

import numpy as np
import numpy.typing as npt

Category = np.uint
Count = np.uint


@dataclasses.dataclass
class Model:
    # Shape: (1, num_categories)
    prior: npt.NDArray[np.floating]
    # Shape: (num_categories, num_features)
    likelihood: npt.NDArray[np.floating]


def train(categories: npt.NDArray[Category], documents: npt.NDArray[Count]) -> Model:
    assert len(categories) == len(documents)
    return Model(get_prior(categories), get_likelihood(categories, documents))


def get_prior(categories: npt.NDArray[Category]) -> npt.NDArray[np.floating]:
    unique, counts = np.unique(categories, return_counts=True)
    prior = np.zeros(len(unique))
    prior[unique] = counts / len(categories)
    return prior


def get_likelihood(
    categories: npt.NDArray[Category], documents: npt.NDArray[Count]
) -> npt.NDArray[np.floating]:
    assert len(categories) == len(documents)

    unique = np.unique(categories)

    category_likelihoods = []
    for category in unique:
        indices = np.nonzero(categories == category)
        word_counts = documents[indices].sum(axis=0)
        all_counts = word_counts.sum()
        words_likelihoods = (word_counts + 1) / (all_counts + len(word_counts))
        category_likelihoods.append(words_likelihoods)

    return np.vstack(category_likelihoods)


def predict(document: npt.NDArray[Count], model: Model) -> Category:
    pass
