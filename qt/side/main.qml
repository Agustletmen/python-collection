import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Column {
        anchors.centerIn: parent
        spacing: 12

        // Text {
        //     text: qsTr("Hello World")
        //     font.pointSize: 28
        //     horizontalAlignment: Text.AlignCenter
        // }

        // Button {
        //     text: qsTr("Click me")
        //     onClicked: console.log("Button clicked!")
        // }

        ListView {
            anchors.fill: parent // 如果你想让列表视图填充父级空间
            // 或者指定宽度和高度
            // width: 200; height: 200
            model: listModel
            delegate: Text {
                text: modelData
                font.pixelSize: 20
                anchors.horizontalCenter: parent.horizontalCenter // 居中对齐文本
            }
        }

        // 定义一个数据模型
        ListModel {
            id: listModel
            ListElement { name: "Apple" }
            ListElement { name: "Banana" }
            ListElement { name: "Orange" }
        }
    }
}