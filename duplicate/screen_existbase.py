from script.screen_test import grab_screen
import numpy as np
import time


def exist_base():
    
    window1 = (292,614,318,685)
    # 首先捕获窗口的屏幕
    prev_screen = grab_screen(window1)

    # while True:
    # 持续捕获窗口的屏幕内容
    curr_screen = grab_screen(window1)

    # 判断当前屏幕内容与之前的屏幕内容是否相同，如果发生了变化，则返回 True
    if not np.array_equal(prev_screen, curr_screen):
        return True

    # 更新上一次的屏幕内容
    prev_screen = curr_screen

    # 你可能想在此处插入一些延迟，以避免过度占用CPU
    # time.sleep(0.1)
    
    # 如果在函数退出（例如通过按键中断）时还没有发现变化，就返回 False
    return False

if exist_base():
    print("你基地炸了")
