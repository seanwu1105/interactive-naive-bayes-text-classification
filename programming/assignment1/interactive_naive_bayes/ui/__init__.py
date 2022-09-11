# pylint: disable=invalid-name
import json
import threading
from typing import TypedDict

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QmlElement

from interactive_naive_bayes.naive_bayes.classifier import Model, predict
from interactive_naive_bayes.naive_bayes.importance import (
    WordImportance,
    get_word_importance,
)
from interactive_naive_bayes.naive_bayes.preprocessing import (
    ProcessedData,
    preprocess,
    to_document,
)
from interactive_naive_bayes.naive_bayes.validation import validate

QML_IMPORT_NAME = "InteractiveNaiveBayes.Ui"
QML_IMPORT_MAJOR_VERSION = 1


class State(TypedDict):
    text: str
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
        self._word_mask: tuple[str, ...] = ()
        self._processed: ProcessedData | None = None
        self._model: Model | None = None
        self._state: State = {
            "text": "",
            "accuracy": 0.0,
            "loadingLabel": "Initializing",
            "predictionResult": "",
            "confidence": 0.0,
            "wordImportance": (),
        }
        threading.Thread(target=self._train).start()

    @Property(str, notify=stateChanged)
    def state(self):
        return json.dumps(self._state)

    def _set_state(self, state: State):
        self._state = state
        self.stateChanged.emit()

    @Slot(str)
    def setText(self, value: str):
        if self._state["text"] == value:
            return
        self._set_state({**self._state, "text": value})

    @Slot()
    def predict(self):
        assert self._model is not None
        assert self._processed is not None

        document = to_document(self._state["text"], self._processed.vocabulary_indices)
        category, confidence = predict(document, self._model)
        result = self._processed.category_labels[category]
        word_importance = get_word_importance(
            document, category, self._model.likelihood, self._processed.vocabulary
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
    def addWord(self, value: str):
        def task():
            self._train()
            self.predict()

        self._word_mask = tuple(word for word in self._word_mask if word != value)
        threading.Thread(target=task).start()

    @Slot(str)
    def removeWord(self, value: str):
        def task():
            self._train()
            self.predict()

        self._word_mask += (value,)
        threading.Thread(target=task).start()

    @Slot(str, float)
    def setWordImportance(self, value: str, importance: float):
        print(value, importance)

    def _train(self):
        self._set_state({**self._state, "loadingLabel": "Preprocessing"})
        self._processed = preprocess(word_mask=self._word_mask)

        self._set_state({**self._state, "loadingLabel": "Training"})
        self._model, accuracy = validate(
            10,
            self._processed.categories,
            self._processed.documents,
        )
        self._set_state({**self._state, "accuracy": accuracy, "loadingLabel": ""})
