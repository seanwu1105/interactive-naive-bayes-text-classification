# pylint: disable=invalid-name
import dataclasses
import json
import threading
from typing import TypedDict

import numpy as np
from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QmlElement

from interactive_naive_bayes.naive_bayes.classifier import Model, predict
from interactive_naive_bayes.naive_bayes.importance import (
    WordImportance,
    adjust_category_smoothing,
    get_word_importance,
)
from interactive_naive_bayes.naive_bayes.preprocessing import (
    ProcessedData,
    preprocess,
    to_document,
)
from interactive_naive_bayes.naive_bayes.validation import test, validate

QML_IMPORT_NAME = "InteractiveNaiveBayes.Ui"
QML_IMPORT_MAJOR_VERSION = 1


class State(TypedDict):
    text: str
    accuracy: float
    loadingLabel: str
    predictionResult: str
    confidence: float
    wordImportance: tuple[WordImportance, ...]
    progress: float


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
            "progress": 0.0,
        }
        threading.Thread(target=self._train).start()

    # Ignore type error due to: https://bugreports.qt.io/browse/PYSIDE-1675
    @Property(str, notify=stateChanged)  # type: ignore[operator, arg-type]
    def state(self):
        return json.dumps(self._state)

    def _set_state(self, state: State):
        self._state = state
        self.stateChanged.emit()

    @Slot(str)
    def setText(self, value: str):
        if self._state["text"] == value:
            return
        # Ignore type error due to https://github.com/python/mypy/issues/4122
        self._set_state({**self._state, "text": value})  # type: ignore[misc]

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
        self._word_mask = tuple(word for word in self._word_mask if word != value)
        threading.Thread(target=self._retrain).start()

    @Slot(str)
    def removeWord(self, value: str):
        self._word_mask += (value,)
        threading.Thread(target=self._retrain).start()

    @Slot(str, float)
    def setWordImportance(self, value: str, importance: float):
        assert self._processed is not None

        category = self._processed.category_labels.index(
            self._state["predictionResult"]
        )

        word_idx = self._processed.vocabulary_indices[value]

        document_indices = np.nonzero(self._processed.categories == category)
        documents = self._processed.documents[document_indices]

        new_category_smoothing = adjust_category_smoothing(
            old_smoothing=self._processed.smoothing[category],
            target_likelihood=importance,
            word_idx=word_idx,
            documents=documents,
        )
        new_smoothing = np.copy(self._processed.smoothing)
        new_smoothing[category] = new_category_smoothing

        self._processed = dataclasses.replace(self._processed, smoothing=new_smoothing)

        threading.Thread(target=self._retrain).start()

    def _retrain(self):
        self._train()
        self.predict()

    def _train(self):
        self._set_state(
            {**self._state, "loadingLabel": "Preprocessing", "progress": 0.0}
        )

        if self._processed is None:
            self._processed = preprocess(word_mask=self._word_mask)
        stacked = np.column_stack(
            (self._processed.categories, self._processed.documents)
        )
        np.random.shuffle(stacked)

        train_size = int(len(stacked) * 0.8)
        train_categories, train_documents = (
            stacked[:train_size, 0],
            stacked[:train_size, 1:],
        )
        test_categories, test_documents = (
            stacked[train_size:, 0],
            stacked[train_size:, 1:],
        )

        self._set_state({**self._state, "loadingLabel": "Training"})
        self._model, _ = validate(
            10,
            train_categories,
            train_documents,
            self._processed.smoothing,
            on_progress=lambda progress: self._set_state(
                {**self._state, "progress": progress}
            ),
        )

        accuracy = test(test_categories, test_documents, self._model)

        self._set_state({**self._state, "accuracy": accuracy, "loadingLabel": ""})
