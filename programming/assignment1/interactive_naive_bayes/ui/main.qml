import QtQuick
import QtQuick.Controls
import QtQuick.Controls
import QtQuick.Layouts
import QtCharts
import InteractiveNaiveBayes.Ui

ApplicationWindow {
    Bridge {
        id: bridge
    }

    visible: true
    width: 640
    height: 480
    title: "Interactive Naive Bayes Text Classifier"

    GridLayout {
        anchors.fill: parent
        columns: 2

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width / 3
            background: Rectangle {
                color: "ghostwhite"
            }

            TextArea {
                placeholderText: "Text Document"
            }
        }
        
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 8

            Text {
                // text: "Model Accuracy: 75%"
                text: bridge.myText
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
            onClicked: {
                bridge.setMyText("Hello New World!")
            }
        }

        Text {
            Layout.fillWidth: true
            text: "Prediction Result: Person (81%)"
        }
    }
}