```qml
元素名 {
    属性名1: 属性值1
    属性名2: 属性值2
    ...

    子元素1 {
        ...
    }
    子元素2 {
        ...
    }
    ...
}    
```

使用 Connections 元素来连接信号和槽
```qml
Rectangle {
    width: 100
    height: 100
    color: "red"
 
    signal clicked()
 
    MouseArea {
        anchors.fill: parent
        onClicked: parent.clicked()
    }
}
 
Rectangle {
    width: 100
    height: 100
    color: "blue"
 
    Connections {
        target: redRect
        onClicked: {
            console.log("Red rectangle clicked!")
        }
    }
}
```


https://blog.csdn.net/BenBierBa/article/details/130421025