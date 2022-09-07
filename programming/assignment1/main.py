import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load("interactive_naive_bayes/ui/main.qml")
    rootObjects = engine.rootObjects()
    if not rootObjects:
        sys.exit("Engine loading failed")
    sys.exit(app.exec())
