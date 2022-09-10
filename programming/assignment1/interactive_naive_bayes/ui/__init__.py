# pylint: disable=invalid-name
import threading
from typing import TypedDict

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QmlElement

from interactive_naive_bayes.naive_bayes.classifier import Model, predict
from interactive_naive_bayes.naive_bayes.preprocessing import (
    ProcessedData,
    preprocess,
    to_document,
)
from interactive_naive_bayes.naive_bayes.validation import validate

QML_IMPORT_NAME = "InteractiveNaiveBayes.Ui"
QML_IMPORT_MAJOR_VERSION = 1


class State(TypedDict):
    accuracy: float
    loadingLabel: str
    predictionResult: str
    confidence: float


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
        }
        threading.Thread(target=self.train).start()

    @Property("QVariantMap", notify=stateChanged)
    def state(self):
        return self._state

    def set_state(self, state: State):
        self._state = state
        self.stateChanged.emit()

    @Slot(str)
    def predict(self, value):
        def _predict(model: Model):
            if self.processed is None:
                return
            category, confidence = predict(
                to_document(value, self.processed.vocabulary_indices), model
            )
            result = self.processed.category_labels[category]
            self.set_state({**self._state, "predictionResult": result})
            self.set_state({**self._state, "confidence": confidence})

        assert self.model is not None

        return _predict(self.model)

    def train(self):
        if self.processed is None:
            self.set_state({**self._state, "loadingLabel": "Preprocessing"})
            self.processed = preprocess()

        self.set_state({**self._state, "loadingLabel": "Training"})
        self.model, accuracy = validate(
            10, self.processed.categories, self.processed.documents
        )
        self.set_state({**self._state, "accuracy": accuracy, "loadingLabel": ""})
