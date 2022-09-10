import QtQuick
import QtQuick.Controls
import QtQuick.Controls
import QtQuick.Layouts
import QtCharts
import InteractiveNaiveBayes.Ui

ApplicationWindow {
    id: app

    Bridge {
        id: bridge
    }
    
    property var state: JSON.parse(bridge.state)

    visible: true
    width: 1080
    height: 720
    title: "Interactive Naive Bayes Text Classifier"

    ColumnLayout {
        anchors.fill: parent

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height / 4
            background: Rectangle {
                color: "ghostwhite"
            }

            TextArea {
                id: textArea
                placeholderText: "Text Document"
            }
        }

        Button {
            Layout.fillWidth: true
            text: "Predict"
            enabled: app.state.loadingLabel.length === 0
            onClicked: {
                bridge.predict(textArea.text)
            }
        }
        
        ColumnLayout {
            Layout.margins: 8
            Layout.preferredHeight: parent.height / 2

            Text {
                text: `Model Accuracy: ${Number.parseFloat(app.state.accuracy * 100).toFixed(2)}%`
            }

            ChartView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Word Importance"
                legend.visible: false
                antialiasing: true

                BarSeries {
                    axisX: BarCategoryAxis {
                        categories: app.state.wordImportance.map(i => i.word)
                        labelsAngle: 45
                    }
                    axisY: ValueAxis {
                        max: app.state.wordImportance.length === 0 ? 1 : Math.max(...app.state.wordImportance.map(i => i.importance))
                        min: 0
                    }
                    BarSet { values: app.state.wordImportance.map(i => i.importance) }
                }
            }
        }
    }

    footer: Frame {
        background: Rectangle {
            color: "whitesmoke"
        }
        contentWidth: parent.width
        contentHeight: 40
        topPadding: 0
        bottomPadding: 0

        RowLayout {
            anchors.fill: parent
            Text {
                Layout.fillWidth: true
                text: (app.state.loadingLabel.length > 0
                    ? app.state.loadingLabel
                    : app.state.predictionResult.length > 0
                    ? `Prediction Result: ${app.state.predictionResult} (${Number.parseFloat(app.state.confidence * 100).toFixed(2)}%)`
                    : "")
            }

            BusyIndicator {
                Layout.preferredHeight: 40
                visible: app.state.loadingLabel.length > 0
                running: true
            }
        }
    }
}