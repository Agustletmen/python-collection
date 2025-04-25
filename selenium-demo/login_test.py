import unittest
from selenium import webdriver
import time
from base import Base


class TestKlogin(unittest.TestCase):
    # 定位手机号
    username = ("name", "loginName")
    # 定位密码
    psw = ("name", "loginPassWord")
    # 定位登录按钮
    login_button = ("id", "loginBtn")
    # 定位帮助中心
    help = ("xpath", "//*[contains(text(),'帮助中心')]")
    print("help", help)
    # 定位提示信息
    message = ("className", "toast-message")
    print("message", message)

    @classmethod
    def setUpClass(cls):
        cls.driver = webdriver.Firefox()
        cls.baselei = Base(cls.driver)

    def setUp(self):
        self.driver.get("url地址")

    def tearDown(self):
        # 清空cookies
        self.driver.delete_all_cookies()

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_01_login_success(self):
        '''正确输入手机号和密码'''
        self.baselei.send(self.username, "13000000000")
        self.baselei.send(self.psw, "666666")
        self.baselei.click(self.login_button)
        result1 = self.baselei.is_element_exist(self.help)
        self.assertTrue(result1)

    def test_02_bushuru(self):
        '''不输入，点击登录'''
        self.baselei.click(self.login_button)
        result2 = self.baselei.find(self.message).text
        print("我是内容test02：", result2)
        exp2 = "请输入手机号码"
        self.assertEqual(result2, exp2)


    def test_03_shuruname(self):
        '''只输入手机号，不输入密码'''
        self.baselei.send(self.username, "12343657")
        self.baselei.click(self.login_button)
        result3 = self.baselei.find(self.message).text
        print("我是内容test03：", result3)
        exp3 = "请输入密码"
        self.assertTrue(result3 == exp3)

    def test_04_shurupsw(self):
        '''只输入密码，不输入手机号'''
        self.baselei.send(self.psw, "123465")
        self.baselei.click(self.login_button)
        result4 = self.baselei.find(self.message).text
        print("我是内容test04:", result4)
        exp4 = "请输入手机号码"
        self.assertTrue(result4 == exp4)

    def test_05_shurufail(self):
        '''输入错误的账号和密码'''
        self.baselei.send(self.username, "4334668")
        self.baselei.send(self.psw, "325465")
        self.baselei.click(self.login_button)
        result5 = self.baselei.find(self.message).text
        print("我是内容test05", result5)
        exp5 = "账号不存在"
        self.assertEqual(result5, exp5)


if __name__ == '__main__':
    unittest.main()
