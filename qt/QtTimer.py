# https://zhuanlan.zhihu.com/p/643738570
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from pathlib import Path


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(610, 374)
        self.formLayout = QtWidgets.QFormLayout(Form)
        self.formLayout.setObjectName("formLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_lcd1_start = QtWidgets.QPushButton(parent=Form)
        self.pushButton_lcd1_start.setObjectName("pushButton_lcd1_start")
        self.horizontalLayout.addWidget(self.pushButton_lcd1_start)
        self.pushButton_lcd1_pause = QtWidgets.QPushButton(parent=Form)
        self.pushButton_lcd1_pause.setObjectName("pushButton_lcd1_pause")
        self.horizontalLayout.addWidget(self.pushButton_lcd1_pause)
        self.pushButton_lcd1_stop = QtWidgets.QPushButton(parent=Form)
        self.pushButton_lcd1_stop.setObjectName("pushButton_lcd1_stop")
        self.horizontalLayout.addWidget(self.pushButton_lcd1_stop)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.lcdNumber_1 = QtWidgets.QLCDNumber(parent=Form)
        self.lcdNumber_1.setMinimumSize(QtCore.QSize(0, 100))
        self.lcdNumber_1.setLineWidth(1)
        self.lcdNumber_1.setDigitCount(12)
        self.lcdNumber_1.setObjectName("lcdNumber_1")
        self.lcdNumber_1.display("00:00:00:000")
        self.verticalLayout.addWidget(self.lcdNumber_1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.comboBox_lcd2_hour = QtWidgets.QComboBox(parent=Form)
        self.comboBox_lcd2_hour.setObjectName("comboBox_lcd2_hour")
        self.comboBox_lcd2_hour.setToolTip("时")  # 设置提示
        self.comboBox_lcd2_hour.addItems(map(str, range(13)))
        self.horizontalLayout_2.addWidget(self.comboBox_lcd2_hour)
        self.comboBox_lcd2_minute = QtWidgets.QComboBox(parent=Form)
        self.comboBox_lcd2_minute.setObjectName("comboBox_lcd2_minute")
        self.comboBox_lcd2_minute.addItems(map(str, range(60)))
        self.comboBox_lcd2_minute.setToolTip("分")  # 设置提示
        self.comboBox_lcd2_minute.setCurrentText("1")  # 设置默认值
        self.horizontalLayout_2.addWidget(self.comboBox_lcd2_minute)
        self.pushButton_countdown_start = QtWidgets.QPushButton(parent=Form)
        self.pushButton_countdown_start.setObjectName("pushButton_countdown_start")
        self.horizontalLayout_2.addWidget(self.pushButton_countdown_start)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.lcdNumber_2 = QtWidgets.QLCDNumber(parent=Form)
        self.lcdNumber_2.setMinimumSize(QtCore.QSize(0, 100))
        self.lcdNumber_2.setDigitCount(8)
        self.lcdNumber_2.setObjectName("lcdNumber_2")
        self.lcdNumber_2.display("00:00:00")
        self.verticalLayout.addWidget(self.lcdNumber_2)
        self.label_countdown_info = QtWidgets.QLabel(parent=Form)
        self.label_countdown_info.setObjectName("label_countdown_info")
        self.verticalLayout.addWidget(self.label_countdown_info)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_browse_audio = QtWidgets.QPushButton(parent=Form)
        self.pushButton_browse_audio.setObjectName("pushButton_browse_audio")
        self.horizontalLayout_3.addWidget(self.pushButton_browse_audio)
        self.pushButton_play_audio = QtWidgets.QPushButton(parent=Form)
        self.pushButton_play_audio.setObjectName("pushButton_6")
        self.pushButton_stopaudio = QtWidgets.QPushButton(parent=Form)
        self.pushButton_stopaudio.setObjectName("pushButton_7")
        self.horizontalLayout_3.addWidget(self.pushButton_play_audio)
        self.horizontalLayout_3.addWidget(self.pushButton_stopaudio)
        self.horizontalSlider_volume = QtWidgets.QSlider(parent=Form)
        self.horizontalSlider_volume.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.horizontalSlider_volume.setObjectName("horizontalSlider_volume")
        self.horizontalSlider_volume.setMaximum(100)  # 不设置的话，默认最大值是99
        self.horizontalSlider_volume.setValue(50)  # 设置当前滑动条位置
        self.horizontalLayout_3.addWidget(self.horizontalSlider_volume)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_currentaudio_info = QtWidgets.QLabel(parent=Form)
        self.label_currentaudio_info.setObjectName("label_currentaudio_info")
        self.verticalLayout.addWidget(self.label_currentaudio_info)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=Form)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.horizontalLayout_4.addWidget(self.plainTextEdit)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.horizontalLayout_4)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.lasttimepause_counting = 0  # 因为有暂停的需求，所以需要一个变量储存上次的计时
        self.current_working_dir = Path(__file__).cwd()  # 获取当前py文件所在路径
        self.media_play = QMediaPlayer()  # 创建QMediaPlayer()，这个类可播放音频视频
        self.audio_output = QAudioOutput()  # 创建一个QAudioOutput()，QMediaPlayer()要播放视频还得承载一个视频类
        self.media_play.setAudioOutput(self.audio_output)  # 设置QMediaPlayer()的音频输出处理
        self.qdeadlinetimer = QtCore.QDeadlineTimer()  # 创建一个QtCore.QDeadlineTimer()对象，它可设置过期时间，检测是否过期等等
        self.QElapsedTimer = QtCore.QElapsedTimer()  # 创建一个计时QElapsedTimer对象
        self.qtimer_1 = QtCore.QTimer()  # 分别创建两个定时器，可按指定间隔发出信号
        self.qtimer_2 = QtCore.QTimer()
        self.qfontmetrics = QtGui.QFontMetrics(self.label_currentaudio_info.font())  # 创建一个可采用省略号处理过长文本的对象，避免因文本过长导致布局变形
        self.read_exist_audiopath_config()  # 读取音频文件路径配置文件
        self.pushButton_lcd1_pause.setEnabled(False)  # 禁用计时暂停、停止按钮，避免在未开始计时相应的函数调用
        self.pushButton_lcd1_stop.setEnabled(False)  # 禁用计时暂停、停止按钮，避免在未开始计时相应的函数调用
        self.pushButton_lcd1_start.clicked.connect(
            lambda: self.start_counting_time(Form))  # 以下都使用了临时函数lambda传参，Form在这类应由外面传入，但qt有些函数需要它，故继续传下去
        self.pushButton_lcd1_pause.clicked.connect(lambda: self.pause_counting_time(Form))
        self.pushButton_lcd1_stop.clicked.connect(lambda: self.stop_counting_time(Form))
        self.qtimer_1.timeout.connect(lambda: self.counting_time(Form))
        self.qtimer_2.timeout.connect(lambda: self.countdown_time_display(Form))
        self.pushButton_countdown_start.clicked.connect(lambda: self.start_countdown_time(Form))
        self.pushButton_play_audio.clicked.connect(lambda: self.play_audio(Form))
        self.pushButton_stopaudio.clicked.connect(lambda: self.stop_play_audio(Form))
        self.pushButton_browse_audio.clicked.connect(lambda: self.explore_audio(Form))
        self.horizontalSlider_volume.valueChanged.connect(
            lambda: self.audio_output.setVolume(self.horizontalSlider_volume.value() / 100))  # 音量条即时更新音量值
        self.media_play.mediaStatusChanged.connect(lambda: self.playing_status(Form))  # 使用mediaStatusChanged捕获播放器的状态

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "计时小工具"))
        self.pushButton_lcd1_start.setText(_translate("Form", "开始"))
        self.pushButton_lcd1_pause.setText(_translate("Form", "暂停"))
        self.pushButton_lcd1_stop.setText(_translate("Form", "停止"))
        self.pushButton_countdown_start.setText(_translate("Form", "开始倒计时"))
        self.label_countdown_info.setText(_translate("Form", "上次倒计时结束于:"))
        self.pushButton_browse_audio.setText(_translate("Form", "设置音频"))
        self.pushButton_play_audio.setText(_translate("Form", "测试"))
        self.pushButton_stopaudio.setText(_translate("Form", "停止"))
        self.label_currentaudio_info.setText(_translate("Form", "当前铃声:"))
        self.plainTextEdit.setPlaceholderText(_translate("Form", "简易笔记本:"))  # 设置背景文字

    def start_counting_time(self, Form):  # 定义开始计时需要的一些操作
        if not self.qtimer_1.isActive():  # 为避免用户多次点击出现重新计时
            self.QElapsedTimer.start()  # QElapsedTimer是qt中无关系统时间的计时器，不会因为修改系统时间变化而变化。start/restart方法好像差不多,它没有stop的方法
            self.pushButton_lcd1_stop.setEnabled(True)
            self.pushButton_lcd1_pause.setEnabled(True)
            self.qtimer_1.start(50)  # 设置定时器发送信号间隔时间，这里1000是1秒

    def counting_time(self, Form):
        if self.lasttimepause_counting == 0:  # 判断是否存在上次计时暂存
            y = self.QElapsedTimer.elapsed()
            z = QtCore.QTime.fromMSecsSinceStartOfDay(y % 86400000).toString(
                "hh:mm:ss:zzz")  # QTime.fromMSecsSinceStartOfDay()秒后返回是3位的，没发现有显示秒后两位的办法。因为这方法超一天毫秒总数就无返回，所以用%86400000取余的办法实现循环计时，一天是86,400秒，不过谁会计时一整天呢
            self.lcdNumber_1.display(z)

        else:  # 如果出现暂停又继续计时的情况，QElapsedTimer会重新从0计算，所以要加回上次计时结果
            y = self.lasttimepause_counting + self.QElapsedTimer.elapsed()
            z = QtCore.QTime.fromMSecsSinceStartOfDay(y % 86400000).toString("hh:mm:ss:zzz")
            self.lcdNumber_1.display(z)

    def pause_counting_time(self, Form):
        if self.qtimer_1.isActive():  # 若用户多次点击暂停，会出现self.lasttimepause_counting继续加的问题，需要判断定时器是否还在运行作对应处理
            self.lasttimepause_counting += self.QElapsedTimer.elapsed()  # 暂停可能是多次，需用+=累加
            self.qtimer_1.stop()  # 停止定时器
            self.pushButton_lcd1_start.setText('继续')

    def stop_counting_time(self, Form):
        self.lasttimepause_counting = 0  # 上次计时暂存归0
        self.qtimer_1.stop()  # 停止定时器
        self.pushButton_lcd1_start.setText('开始')
        self.pushButton_lcd1_pause.setEnabled(False)
        self.pushButton_lcd1_stop.setEnabled(False)

    def start_countdown_time(self, Form):
        x = QtCore.QTime.fromString(
            f'{self.comboBox_lcd2_hour.currentText().zfill(2)}:{self.comboBox_lcd2_minute.currentText().zfill(2)}:00')  # 使用QtCore.QTime.fromString返回一个Qtime对对象，方便根据时分返回总毫秒数，fromString不能识别1：1：1这样的时间格式，需使用字符串的zfill方法补位为01：01：01这样的格式
        if (y := x.msecsSinceStartOfDay()) != 0:  # 若倒计时时间为零，倒计时无意义,这里用了海象运算符，并使用括号包，提高其优先级，否则逻辑不对
            self.qdeadlinetimer.setRemainingTime(y)  # 设置self.qdeadlinetimer过期时间
            self.qtimer_2.start(100)  # 以0.1秒间隔启用定时器

    def countdown_time_display(self, Form):  # 倒计时显示函数
        if not self.qdeadlinetimer.hasExpired():  # 未过期时更新lcd2的显示内容
            x = QtCore.QTime.fromMSecsSinceStartOfDay(self.qdeadlinetimer.remainingTime()).toString("hh:mm:ss")
            self.lcdNumber_2.display(x)
        else:
            self.qtimer_2.stop()
            self.label_countdown_info.setText(
                f"上次倒计时结束于: {QtCore.QDateTime().currentDateTime().toString('dd hh:mm:ss')}")  # 考虑到有可能跨日，使用QtCore.QDateTime()把日期也获取
            self.play_audio(Form)

    def play_audio(self, Form):
        self.audio_output.setVolume(self.horizontalSlider_volume.value() / 100)
        try:
            self.media_play.play()  # 为避免程序在播放出错时崩溃，使用try包起来，QMediaPlayer播放音频有几个步骤，有错误不能马上捕获的
        except:
            QtWidgets.QMessageBox.information(Form, '错误', '无法播放！')

    def stop_play_audio(self, Form):
        self.media_play.stop()

    def explore_audio(self, Form):
        self.audio_path, y = QtWidgets.QFileDialog.getOpenFileName(Form,
                                                                   "选择音频文件",
                                                                   "./",
                                                                   "音频文件(*.mp3 *.wav *.aac *.flac *.wma)")  # 创建一个文件打开对话框，其中y是返回过滤器
        if self.audio_path:  # 因为存在打开了对话框但不选择然后关闭的情形，这样将无返回值的
            audio_path = Path(self.audio_path)
            if (a := len(audio_path.parts)) >= (b := len(self.current_working_dir.parts)):
                if audio_path.parts[:b] == self.current_working_dir.parts:  # 使用工作路径长度截取文件路径，并比较，判断是否可视为相对路径
                    self.audio_path = str(audio_path.relative_to(self.current_working_dir))  # 取得基于当前工作目录得到的相对路径
            displaystr = self.qfontmetrics.elidedText(f"当前铃声:{self.audio_path}", QtCore.Qt.TextElideMode.ElideLeft, 320)
            self.label_currentaudio_info.setText(displaystr)
            self.label_currentaudio_info.setToolTip(f"当前铃声:{self.audio_path}")
            self.media_play.setSource(QtCore.QUrl.fromLocalFile(self.audio_path))
            with open('audio_path.txt', encoding='utf-8', mode='w') as e:
                e.write(self.audio_path)

    def read_exist_audiopath_config(self):
        if Path("audio_path.txt").is_file():  # 检查是否存在对应文件
            with open('audio_path.txt', encoding='utf-8', mode='r') as e:
                check_path = Path(e.read())
                if check_path.is_file():  # 检查是否存在对应文件
                    self.audio_path = str(check_path)
                    displaystr = self.qfontmetrics.elidedText(f"当前铃声:{self.audio_path}",
                                                              QtCore.Qt.TextElideMode.ElideLeft, 320)
                    self.label_currentaudio_info.setText(displaystr)
                    self.label_currentaudio_info.setToolTip(f"当前铃声:{self.audio_path}")
                    self.media_play.setSource(QtCore.QUrl.fromLocalFile(self.audio_path))

    def playing_status(self, Form):
        # print(self.media_play.mediaStatus())
        if self.media_play.mediaStatus().value == 7:  # mediaStatus()是一个类，而无效子类的值是7
            QtWidgets.QMessageBox.information(Form, '错误', '无法播放！')


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
