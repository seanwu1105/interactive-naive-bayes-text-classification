# pylint: disable=invalid-name

from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtQml import QmlElement

QML_IMPORT_NAME = "InteractiveNaiveBayes.Ui"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class Bridge(QObject):
    stateChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._myText = "Hello World!"

    @Property(str, notify=stateChanged)
    def myText(self):
        return self._myText

    @Slot(str)
    def setMyText(self, value):
        self._myText = value
        self.stateChanged.emit()
