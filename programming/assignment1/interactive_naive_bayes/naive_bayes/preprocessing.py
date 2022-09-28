import collections
import dataclasses
import functools
import os
import string

import nltk
import nltk.corpus
import numpy as np
import numpy.typing as npt
import pandas as pd

from interactive_naive_bayes.naive_bayes.classifier import Category, Count


@dataclasses.dataclass
class ProcessedData:
    categories: npt.NDArray[Category]
    category_labels: tuple[str, ...]
    documents: npt.NDArray[Count]  # TODO: Use a sparse matrix
    vocabulary: tuple[str, ...]
    vocabulary_indices: dict[str, int]
    smoothing: npt.NDArray[Count]


def _get_default_data_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "dbpedia_8K.csv")


def preprocess(
    filename=_get_default_data_path(),
    word_mask: tuple[str, ...] | None = None,
    old_data: ProcessedData | None = None,
) -> ProcessedData:
    df = pd.read_csv(filename)

    assert df["label"].unique().size == len(_TARGET_LABELS)

    df["content"] = (
        df["content"]
        .map(_remove_punctuation)
        .map(lambda s: s.lower().split())
        .map(lambda words: (word for word in words if word not in _get_stopwords()))
        .map(
            lambda words: (
                word for word in words if word_mask is None or word not in word_mask
            )
        )
        .map(collections.Counter)
    )

    vocabulary: tuple[str, ...] = tuple(
        set(key for counters in df["content"].values for key in counters.keys())
    )

    vocabulary_indices = {word: i for i, word in enumerate(vocabulary)}

    documents = np.zeros((len(df), len(vocabulary)), dtype=Count)

    for i, counter in enumerate(df["content"].values):
        for word, count in counter.items():
            documents[i, vocabulary_indices[word]] = count

    smoothing = np.ones((len(_TARGET_LABELS), len(vocabulary)), dtype=Count)
    if old_data is not None:
        for category in range(len(_TARGET_LABELS)):
            for word in vocabulary:
                old_word_smoothing = old_data.smoothing[category][
                    old_data.vocabulary_indices[word]
                ]
                smoothing[category][vocabulary_indices[word]] = old_word_smoothing

    return ProcessedData(
        categories=df["label"].to_numpy(dtype=Category),
        category_labels=_TARGET_LABELS,
        documents=documents,
        vocabulary=vocabulary,
        vocabulary_indices=vocabulary_indices,
        smoothing=smoothing,
    )


def to_document(text: str, vocabulary_indices: dict[str, int]) -> npt.NDArray[Count]:
    words = _remove_punctuation(text).lower().split()
    document = np.zeros(len(vocabulary_indices), dtype=Count)
    for word in words:
        if word in vocabulary_indices:
            document[vocabulary_indices[word]] += 1
    return document


def _remove_punctuation(text: str) -> str:
    return "".join(
        map(lambda c: c if c in string.ascii_letters + string.digits else " ", text)
    )


@functools.cache
def _get_stopwords():
    nltk.download("stopwords", os.path.abspath(".venv/lib/nltk_data"))
    return set(nltk.corpus.stopwords.words("english"))


_TARGET_LABELS: tuple[str, ...] = (
    "Company",
    "Education Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean Of Transportation",
    "Building",
    "Natural Place",
)
