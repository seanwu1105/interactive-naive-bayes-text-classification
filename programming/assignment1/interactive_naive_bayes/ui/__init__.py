# pylint: disable=invalid-name
import threading
from typing import Any, Callable

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


@QmlElement
class Bridge(QObject):
    stateChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.processed: ProcessedData | None = None
        self.model: Model | None = None
        self._predictionResult = ""

    @Property(str, notify=stateChanged)
    def predictionResult(self):
        return self._predictionResult

    @Slot(str)
    def predict(self, value):
        def _predict(model: Model):
            if self.processed is None:
                return
            self._predictionResult = self.processed.target_labels[
                predict(to_sample(value, self.processed.label_indices), model)
            ]
            self.stateChanged.emit()

        if self.model is None:
            return self.train(_predict)

        return _predict(self.model)

    def train(self, cb: Callable[[Model], Any]):
        def _train():
            if self.processed is None:
                self.processed = preprocess()
            self.model = train(self.processed.targets, self.processed.samples)
            cb(self.model)

        threading.Thread(target=_train).start()
