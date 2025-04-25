import time
import unittest
from selenium import webdriver

username = '1527248287@qq.com'
password = 'xxxxx'


# 登录类，用于被测试
class Login:
    def __init__(self):
        self.driver = webdriver.Edge(r'msedgedriver.exe')
        self.driver.maximize_window()
        self.driver.implicitly_wait(8)  #隐式等待
        self.driver.get("https://www.scholat.com/login.html")

    def login(self, username, password, screenshot=True):
        if isinstance(username, str):
            time.sleep(1)
            self.driver.find_element_by_id('j_username').send_keys(username)
        if isinstance(password, str):
            time.sleep(1)
            self.driver.find_element_by_id('j_password_ext').send_keys(password)
        time.sleep(1)
        self.driver.find_element_by_id("login").click()
        res = self.driver.find_element_by_id('sp_err').text
        if screenshot:
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            self.driver.get_screenshot_as_file("screenshot%s.png" % now)
        self.driver.delete_all_cookies()
        self.driver.quit()
        return res


class Test(unittest.TestCase):
    # unittest使用的前置条件
    def setUp(self):
        pass

    # 测试用例，必须时test_开头
    # 正确输入手机号和密码，点击登录
    def test_case01(self):
        Login().login(username, "xxxxxx")
        pass

    # 不输入手机号和密码，点击登录
    def test_case02(self):
        pass

    # 只输入手机号不输入密码，点击登录
    def test_case03(self):
        pass

    # 只输入密码不输入手机号，点击登录
    def test_case04(self):
        pass

    # 输入错误的用户名和密码
    def test_case05(self):
        pass

    # unittest使用的后置条件
    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
