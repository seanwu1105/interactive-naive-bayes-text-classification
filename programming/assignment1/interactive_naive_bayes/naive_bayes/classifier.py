import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Model:
    prior: npt.NDArray[np.floating]
    likelihood: npt.NDArray[np.floating]


Category = np.ushort
HasWord = np.ushort


def train(targets: npt.NDArray[Category], features: npt.NDArray[HasWord]) -> Model:
    print(targets)


def predict(feature: npt.NDArray[HasWord], model: Model) -> Category:
    pass
