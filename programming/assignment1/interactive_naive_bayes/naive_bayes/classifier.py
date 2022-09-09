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


def get_prior(categories: npt.NDArray[Category]) -> dict[Category, float]:
    pass


def get_likelihood(
    categories: npt.NDArray[Category], documents: npt.NDArray[Count]
) -> npt.NDArray[np.floating]:
    assert len(categories) == len(documents)


def predict(document: npt.NDArray[Count], model: Model) -> Category:
    pass
