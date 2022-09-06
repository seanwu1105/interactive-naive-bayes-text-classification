# 1. Read file
# 2. Use columns: category, content
# 3. Make sure there is no missing data or empty strings (or nan)
# 4. Get all unique words
# 5. Transform content into the frequency of each word
# 6. Data would be Dict[int, Dict[str, int]]
# 7. Transform data to be np.array[int] (len=categories) and np.array[int, int]
#    (len=[categories, words])
import dataclasses
import itertools
import os
import string
from collections import Counter

import nltk
import nltk.corpus
import numpy as np
import numpy.typing as npt
import pandas as pd
from interactive_naive_bayes.naive_bayes.classifier import Category, WordCount


@dataclasses.dataclass
class ProcessedData:
    targets: npt.NDArray[Category]
    target_labels: tuple[str, ...]
    features: npt.NDArray[WordCount]
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

    counters: list[dict[str, int]] = df["content"].map(Counter).to_list()

    features = np.zeros((len(df), len(feature_labels)), dtype=WordCount)

    for i, counter in enumerate(counters):
        for word, count in counter.items():
            features[i, feature_indices[word]] = count

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
