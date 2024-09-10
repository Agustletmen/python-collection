from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver


class Base():

    def __init__(self, driver):
        self.driver = driver

    def find(self, locator):
        '''locator = ("id","kw")，查找某元素'''
        element = WebDriverWait(self.driver, 10, 1).until(lambda x: x.find_element(*locator))
        return element

    def send(self, locator, text):
        '''输入框传值'''
        self.find(locator).send_keys(text)

    def click(self, locator):
        '''点击事件'''
        self.find(locator).click()

    def is_element_exist(self, locator):
        '''判断元素是否存在'''
        els = self.find(locator)
        count = len(els)  # 计算元素个数
        if len(els) < 1:
            return False
        else:
            print("定位到的元素个数：%s" % count)
            return True
