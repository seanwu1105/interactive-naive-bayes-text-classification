from typing import TypedDict

import numpy as np
import numpy.typing as npt

from interactive_naive_bayes.naive_bayes.classifier import Category


class WordImportance(TypedDict):
    word: str
    importance: float


def get_word_importance(
    document: npt.NDArray,
    category: Category,
    likelihood: npt.NDArray[np.floating],
    vocabulary: tuple[str, ...],
    top=10,
) -> tuple[WordImportance, ...]:
    return tuple(
        reversed(
            tuple(
                {"word": vocabulary[idx], "importance": likelihood[category][idx]}
                for idx in np.argsort(likelihood[category] * (document != 0))[-top:]
                if document[idx] != 0
            )
        )
    )
