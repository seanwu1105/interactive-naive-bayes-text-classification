# pylint: disable=invalid-name
import threading
from typing import Any, Callable, TypedDict

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QmlElement

from interactive_naive_bayes.naive_bayes.classifier import Model, predict, train
from interactive_naive_bayes.naive_bayes.preprocessing import (
    ProcessedData,
    preprocess,
    to_sample,
)

QML_IMPORT_NAME = "InteractiveNaiveBayes.Ui"
QML_IMPORT_MAJOR_VERSION = 1


class State(TypedDict):
    predictionResult: str
    loadingLabel: str


@QmlElement
class Bridge(QObject):
    stateChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processed: ProcessedData | None = None
        self.model: Model | None = None
        self._state: State = {
            "loadingLabel": "",
            "predictionResult": "",
        }
        self.train()

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
            result = self.processed.target_labels[
                predict(to_sample(value, self.processed.label_indices), model)
            ]
            self.set_state({**self._state, "predictionResult": result})

        if self.model is None:
            return self.train(_predict)

        return _predict(self.model)

    def train(self, callback: None | Callable[[Model], Any] = None):
        def _train():
            if self.processed is None:
                self.set_state({**self._state, "loadingLabel": "Preprocessing"})
                self.processed = preprocess()

            self.set_state({**self._state, "loadingLabel": "Training"})
            self.model = train(self.processed.targets, self.processed.samples)
            self.set_state({**self._state, "loadingLabel": ""})

            if callback is not None:
                callback(self.model)

        threading.Thread(target=_train).start()
