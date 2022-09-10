# pylint: disable=invalid-name
import json
import threading
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QmlElement

from interactive_naive_bayes.naive_bayes.classifier import Category, Model, predict
from interactive_naive_bayes.naive_bayes.preprocessing import (
    ProcessedData,
    preprocess,
    to_document,
)
from interactive_naive_bayes.naive_bayes.validation import validate

QML_IMPORT_NAME = "InteractiveNaiveBayes.Ui"
QML_IMPORT_MAJOR_VERSION = 1


class WordImportance(TypedDict):
    word: str
    importance: float


class State(TypedDict):
    accuracy: float
    loadingLabel: str
    predictionResult: str
    confidence: float
    wordImportance: tuple[WordImportance, ...]


@QmlElement
class Bridge(QObject):
    stateChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processed: ProcessedData | None = None
        self.model: Model | None = None
        self._state: State = {
            "accuracy": 0.0,
            "loadingLabel": "Initializing",
            "predictionResult": "",
            "confidence": 0.0,
            "wordImportance": (),
        }
        threading.Thread(target=self._train).start()

    def _train(self):
        if self.processed is None:
            self._set_state({**self._state, "loadingLabel": "Preprocessing"})
            self.processed = preprocess()

        self._set_state({**self._state, "loadingLabel": "Training"})
        self.model, accuracy = validate(
            10,
            self.processed.categories,
            self.processed.documents,
        )
        self._set_state({**self._state, "accuracy": accuracy, "loadingLabel": ""})

    @Property(str, notify=stateChanged)
    def state(self):
        return json.dumps(self._state)

    def _set_state(self, state: State):
        self._state = state
        self.stateChanged.emit()

    @Slot(str)
    def predict(self, value):
        assert self.model is not None
        assert self.processed is not None

        document = to_document(value, self.processed.vocabulary_indices)
        category, confidence = predict(document, self.model)
        result = self.processed.category_labels[category]
        word_importance = get_word_importance(
            document, category, self.model.likelihood, self.processed.vocabulary
        )
        self._set_state(
            {
                **self._state,
                "predictionResult": result,
                "confidence": confidence,
                "wordImportance": word_importance,
            }
        )

    @Slot(str)
    def addWord(self, value):
        print(value)

    @Slot(str)
    def removeWord(self, value):
        print(value)

    @Slot(str, float)
    def setWordImportance(self, value, importance):
        print(value, importance)


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
