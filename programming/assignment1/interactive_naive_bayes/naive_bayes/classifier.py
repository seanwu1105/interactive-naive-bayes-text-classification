import dataclasses

import numba
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


def train(
    categories: npt.NDArray[Category],
    documents: npt.NDArray[Count],
    smoothing: npt.NDArray[Count],
) -> Model:
    assert len(categories) == len(documents)
    return Model(
        _get_prior(categories), _get_likelihood(categories, documents, smoothing)
    )


def _get_prior(categories: npt.NDArray[Category]) -> npt.NDArray[np.floating]:
    unique, counts = np.unique(categories, return_counts=True)
    prior = np.zeros(len(unique))
    prior[unique] = counts / len(categories)
    return prior


def _get_likelihood(
    categories: npt.NDArray[Category],
    documents: npt.NDArray[Count],
    smoothing: npt.NDArray[Count],
) -> npt.NDArray[np.floating]:
    assert len(categories) == len(documents)

    unique = np.unique(categories)

    if smoothing is None:
        smoothing = np.ones((len(unique), documents.shape[1]), dtype=np.float64)

    category_likelihoods = []
    for category in unique:
        indices = np.nonzero(categories == category)
        word_counts = np.sum(documents[indices], axis=0)
        words_likelihoods = (word_counts + smoothing[category]) / (
            np.sum(word_counts) + np.sum(smoothing[category])
        )
        category_likelihoods.append(words_likelihoods)

    return np.vstack(category_likelihoods)


# Use log-sum-exp trick to avoid underflow. See:
# https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes/253319#253319
def predict(document: npt.NDArray[Count], model: Model) -> tuple[Category, float]:
    log_posteriors = _get_log_posteriors(document, model.prior, model.likelihood)
    log_normalizer = scipy.special.logsumexp(log_posteriors)
    probabilities = tuple(
        np.exp(log_posterior - log_normalizer) for log_posterior in log_posteriors
    )

    return np.argmax(probabilities), float(max(probabilities))


@numba.njit(parallel=True, cache=True, nogil=True, fastmath=True)
def _get_log_posteriors(
    document: npt.NDArray[Count],
    priors: npt.NDArray[np.floating],
    likelihoods: npt.NDArray[np.floating],
) -> list[np.floating]:
    return [
        np.log(prior) + np.sum(np.log(likelihood) * document)
        for prior, likelihood in zip(priors, likelihoods)
    ]
