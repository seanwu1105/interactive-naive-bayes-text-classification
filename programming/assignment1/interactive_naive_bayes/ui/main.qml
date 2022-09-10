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
    width: 1080
    height: 480
    title: "Interactive Naive Bayes Text Classifier"

    GridLayout {
        anchors.fill: parent
        columns: 2

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width / 2
            background: Rectangle {
                color: "ghostwhite"
            }

            TextArea {
                id: textArea
                placeholderText: "Text Document"
            }
        }
        
        ColumnLayout {
            Layout.margins: 8

            Text {
                text: `Model Accuracy: ${Number.parseFloat(bridge.state.accuracy * 100).toFixed(2)}%`
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
            enabled: bridge.state.loadingLabel.length === 0
            onClicked: {
                bridge.predict(textArea.text)
            }
        }

        RowLayout {
            Layout.preferredWidth: parent.width / 2

            Text {
                Layout.fillWidth: true
                text: (bridge.state.loadingLabel.length > 0
                       ? bridge.state.loadingLabel
                       : bridge.state.predictionResult.length > 0
                       ? `Prediction Result: ${bridge.state.predictionResult} (${Number.parseFloat(bridge.state.confidence * 100).toFixed(2)}%)`
                       : "")
            }

            BusyIndicator {
                Layout.preferredHeight: 40 // Same as button height
                visible: bridge.state.loadingLabel.length > 0
                running: true
            }
        }
    }
}