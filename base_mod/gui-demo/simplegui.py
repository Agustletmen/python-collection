import PySimpleGUI as sg

layout = [
    [sg.Text("性别："), sg.Radio("男", group_id="gender", key="-MALE-", default=True), sg.Radio("女", group_id="gender", key="-FEMALE-")],
    [sg.Text("爱好："), sg.Checkbox("编程", key="-CODE-"), sg.Checkbox("阅读", key="-READ-"), sg.Checkbox("运动", key="-SPORT-")],
    [sg.Text("城市："), sg.Combo(["北京", "上海", "广州", "深圳"], key="-CITY-", default_value="北京")],
    [sg.Button("提交")]
]

window = sg.Window("表单示例", layout, size=(350, 150))

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "提交":
        gender = "男" if values["-MALE-"] else "女"
        hobbies = []
        if values["-CODE-"]: hobbies.append("编程")
        if values["-READ-"]: hobbies.append("阅读")
        if values["-SPORT-"]: hobbies.append("运动")
        city = values["-CITY-"]
        sg.popup(
            f"性别：{gender}\n"
            f"爱好：{','.join(hobbies) if hobbies else '无'}\n"
            f"城市：{city}"
        )

window.close()