import collections
import dataclasses
import functools
import os
import string

import nltk.corpus
import numpy as np
import numpy.typing as npt
import pandas as pd

from interactive_naive_bayes.naive_bayes.classifier import Category, Count


@dataclasses.dataclass
class ProcessedData:
    categories: npt.NDArray[Category]
    category_labels: tuple[str, ...]
    documents: npt.NDArray[Count]
    vocabulary: tuple[str, ...]
    vocabulary_indices: dict[str, int]


def get_default_data_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "dbpedia_8K.csv")


def preprocess(filename=get_default_data_path()) -> ProcessedData:
    df = pd.read_csv(filename)

    assert df["label"].unique().size == len(TARGET_LABELS)

    df["content"] = (
        df["content"]
        .map(remove_punctuation)
        .map(lambda s: s.lower().split())
        .map(
            lambda words: collections.Counter(
                filter(lambda w: w not in get_stopwords(), words)
            )
        )
    )

    vocabulary: tuple[str, ...] = tuple(
        set(key for counters in df["content"].values for key in counters.keys())
    )

    vocabulary_indices = {word: i for i, word in enumerate(vocabulary)}

    documents = np.zeros((len(df), len(vocabulary)), dtype=Count)

    for i, counter in enumerate(df["content"].values):
        for word, count in counter.items():
            documents[i, vocabulary_indices[word]] = count

    return ProcessedData(
        categories=df["label"].to_numpy(dtype=Category),
        category_labels=TARGET_LABELS,
        documents=documents,
        vocabulary=vocabulary,
        vocabulary_indices=vocabulary_indices,
    )


def remove_punctuation(text: str) -> str:
    return "".join(
        map(lambda c: c if c in string.ascii_letters + string.digits else " ", text)
    )


@functools.cache
def get_stopwords():
    nltk.download("stopwords", os.path.abspath(".venv/lib/nltk_data"))
    return set(nltk.corpus.stopwords.words("english"))


TARGET_LABELS: tuple[str, ...] = (
    "Company",
    "Education Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean Of Transportation",
    "Building",
    "Natural Place",
)
