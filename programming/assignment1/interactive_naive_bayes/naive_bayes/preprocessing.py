import dataclasses
import itertools
import os
import string

import nltk
import nltk.corpus
import numpy as np
import numpy.typing as npt
import pandas as pd
from interactive_naive_bayes.naive_bayes.classifier import Category, HasWord


@dataclasses.dataclass
class ProcessedData:
    targets: npt.NDArray[Category]
    target_labels: tuple[str, ...]
    features: npt.NDArray[HasWord]
    feature_labels: tuple[str, ...]


def get_default_data_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "dbpedia_8K.csv")


def preprocess(filename=get_default_data_path()) -> ProcessedData:
    df = pd.read_csv(filename)

    assert df["label"].unique().size == len(TARGET_LABELS)

    df["content"] = (
        df["content"]
        .map(remove_punctuation)
        .map(lambda s: s.lower().split())
        .map(lambda words: tuple(filter(lambda w: w not in STOPWORDS, words)))
    )

    feature_labels: tuple[str, ...] = tuple(
        set(itertools.chain.from_iterable(df["content"].values))
    )

    feature_indices = {label: i for i, label in enumerate(feature_labels)}

    sets: list[set[str]] = df["content"].map(set).to_list()

    features = np.zeros((len(df), len(feature_labels)), dtype=HasWord)

    for i, words in enumerate(sets):
        for word in words:
            features[i, feature_indices[word]] = 1

    return ProcessedData(
        targets=df["label"].to_numpy(dtype=np.uint),
        target_labels=TARGET_LABELS,
        features=features,
        feature_labels=feature_labels,
    )


def remove_punctuation(text: str) -> str:
    return "".join(
        map(lambda c: c if c in string.ascii_letters + string.digits else " ", text)
    )


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

nltk.download("stopwords", os.path.abspath(".venv/lib/nltk_data"))

STOPWORDS = set(nltk.corpus.stopwords.words("english"))
