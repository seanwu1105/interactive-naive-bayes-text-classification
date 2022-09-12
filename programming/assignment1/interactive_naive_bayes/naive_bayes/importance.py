from typing import TypedDict

import numpy as np
import numpy.typing as npt

from interactive_naive_bayes.naive_bayes.classifier import Category, Count


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


def adjust_category_smoothing(
    # old_smoothing shape: (1, unique_word_count)
    old_smoothing: npt.NDArray[Count],
    target_likelihood: float,
    word_idx: int,
    # documents shape: (document_in_category_count, unique_word_count)
    documents: npt.NDArray[Count],
) -> npt.NDArray[Count]:
    # The likelihood of the word in an arbitrary category is defined as:
    #              word_count[word] + smoothing[word]
    # P(word) = ----------------------------------------
    #               sum(word_count) + sum(smoothing)
    #
    # We can rewrite the above as:
    #                           WC[w] + S[w]
    # P(w) = ------------------------------------------------
    #              sum(WC) + (sum(S) - S[w]) + S[w]
    #
    # To achieve the target likelihood, we need to solve for S[w]:
    #             P(w) * (sum(WC) + (sum(S) - S[w])) - WC[w]
    # S[w] = -------------------------------------------------------
    #                             1 - P(w)
    word_count = np.sum(documents[:, word_idx])
    all_count = np.sum(documents)
    all_smoothing = np.sum(old_smoothing)
    new_word_smoothing = (
        target_likelihood * (all_count + all_smoothing - old_smoothing[word_idx])
        - word_count
    ) / (1 - target_likelihood)
    new_smoothing = old_smoothing.copy()
    new_smoothing[word_idx] = new_word_smoothing
    return new_smoothing
