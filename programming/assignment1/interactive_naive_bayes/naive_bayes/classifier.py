import dataclasses

import numpy as np
import numpy.typing as npt
import scipy.special

Category = np.int_
Count = np.uint


@dataclasses.dataclass
class Model:
    # Shape: (1, num_categories)
    prior: npt.NDArray[np.floating]
    # Shape: (num_categories, num_features)
    likelihood: npt.NDArray[np.floating]


def train(categories: npt.NDArray[Category], documents: npt.NDArray[Count]) -> Model:
    assert len(categories) == len(documents)
    return Model(_get_prior(categories), _get_likelihood(categories, documents))


def _get_prior(categories: npt.NDArray[Category]) -> npt.NDArray[np.floating]:
    unique, counts = np.unique(categories, return_counts=True)
    prior = np.zeros(len(unique))
    prior[unique] = counts / len(categories)
    return prior


def _get_likelihood(
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


# Use log-sum-exp trick to avoid underflow. See:
# https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes/253319#253319
def predict(document: npt.NDArray[Count], model: Model) -> tuple[Category, float]:
    log_normalizer = _get_log_normalizer(document, model)
    probabilities = []
    for category, prior in enumerate(model.prior):
        log_posterior = np.log(prior) + np.sum(
            np.log(model.likelihood[category]) * document
        )

        log_probability = log_posterior - log_normalizer
        probabilities.append(np.exp(log_probability))

    return np.argmax(probabilities), float(max(probabilities))


def _get_log_normalizer(document: npt.NDArray[Count], model: Model):
    return scipy.special.logsumexp(
        np.fromiter(
            (
                np.log(prior) + np.sum(np.log(model.likelihood[category]) * document)
                for category, prior in enumerate(model.prior)
            ),
            dtype=np.float64,
        )
    )
