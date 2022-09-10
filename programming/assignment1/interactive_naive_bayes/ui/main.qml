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
                id: chartView
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Word Importance"
                legend.visible: false
                antialiasing: true
                property real maxImportance: app.state.wordImportance.length === 0 ? 1 : Math.max(...app.state.wordImportance.map(i => i.importance))
                property var draggedBarIndex: undefined
                property var rightClickedBarIndex: undefined

                BarSeries {
                    BarSet {
                        id: barSet
                        values: app.state.wordImportance.map(i => i.importance)
                    }

                    axisX: BarCategoryAxis {
                        categories: app.state.wordImportance.map(i => i.word)
                        labelsAngle: 45
                    }
                    axisY: ValueAxis {
                        max: chartView.maxImportance
                        min: 0
                    }
                }

                MouseArea {
                    id: mouseArea
                    anchors.fill: parent
                    enabled: app.state.wordImportance.length > 0
                    acceptedButtons: Qt.LeftButton | Qt.RightButton
                    onClicked: (event) => {
                        if (event.button !== Qt.RightButton) return

                        const point = chartView.mapToValue(Qt.point(event.x, event.y))
                        if (!isInChartArea(point)) return
                        
                        chartView.rightClickedBarIndex = Math.round(point.x)
                        contextMenu.popup()
                    }
                    onPressed: (event) => {
                        if (event.button !== Qt.LeftButton) return
                        
                        const point = chartView.mapToValue(Qt.point(event.x, event.y))
                        if (!isInChartArea(point)) return

                        chartView.draggedBarIndex = Math.round(point.x)
                    }
                    onPositionChanged: (event) => {
                        if (mouseArea.pressedButtons !== Qt.LeftButton) return
                        if (chartView.draggedBarIndex === undefined) return

                        const point = chartView.mapToValue(Qt.point(event.x, event.y))
                        if (!isInChartArea(point)) return

                        barSet.replace(chartView.draggedBarIndex, point.y)
                    }
                    onReleased: (event) => {
                        if (event.button !== Qt.LeftButton) return
                        if (chartView.draggedBarIndex === undefined) return

                        const word = app.state.wordImportance[chartView.draggedBarIndex].word

                        chartView.draggedBarIndex = undefined

                        const point = chartView.mapToValue(Qt.point(event.x, event.y))

                        bridge.setWordImportance(word, barSet.at(chartView.draggedBarIndex))
                    }

                    Menu {
                        id: contextMenu
                        MenuItem {
                            text: "Add Word"
                            onClicked: {
                                const word = app.state.wordImportance[chartView.rightClickedBarIndex].word
                                bridge.addWord(word)
                            }
                        }
                        MenuItem {
                            text: "Remove Word"
                            onClicked: {
                                const word = app.state.wordImportance[chartView.rightClickedBarIndex].word
                                bridge.removeWord(word)
                            }
                        }
                    }

                    function isInChartArea(point) {
                        return point.x >= -0.5
                            && point.x <= app.state.wordImportance.length - 0.5
                            && point.y >= 0
                            && point.y <= chartView.maxImportance
                    }
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