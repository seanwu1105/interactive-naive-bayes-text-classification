import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCharts

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: "Interactive Naive Bayes Text Classifier"

    GridLayout {
        anchors.fill: parent
        columns: 2

        // TODO: Buggy
        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width / 3
            contentHeight: textArea.implicitHeight

            TextArea {
                id: textArea
                Layout.fillWidth: true
                Layout.fillHeight: true
                placeholderText: "Enter text to classify"
                wrapMode: TextEdit.Wrap
            }

            ScrollBar.vertical: ScrollBar {}
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Text {
                text: "Model Accuracy: 75%"
            }

            ChartView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Word Importance"
                legend.visible: false
                antialiasing: true

                BarSeries {
                    BarSet { values: [2, 2, 3, 4, 5, 6] }
                }
            }
        }

        Button {
            Layout.fillWidth: true
            text: "Predict"
        }

        Text {
            Layout.fillWidth: true
            text: "Prediction Result: Person (81%)"
        }
    }
}