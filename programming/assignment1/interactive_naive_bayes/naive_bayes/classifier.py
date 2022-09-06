import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Model:
    prior: npt.NDArray[np.floating]
    likelihood: npt.NDArray[np.floating]


Category = np.ushort
WordCount = np.uint


def train(target: npt.NDArray[Category], features: npt.NDArray[WordCount]) -> Model:
    pass


def predict(feature: npt.NDArray[WordCount], model: Model) -> Category:
    pass
