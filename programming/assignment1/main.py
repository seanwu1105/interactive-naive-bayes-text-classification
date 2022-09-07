import sys

from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from interactive_naive_bayes.ui import rc_resources

rc_resources  # pylint: disable=pointless-statement

if __name__ == "__main__":
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    # TODO: use rcc
    engine.load("interactive_naive_bayes/ui/main.qml")
    rootObjects = engine.rootObjects()
    if not rootObjects:
        sys.exit("Engine loading failed")
    sys.exit(app.exec())
