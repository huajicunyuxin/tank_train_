#运行该主程序，首先停5秒，然后运行screen_start.py文件，即开始游戏
#然后继续停3秒，之后就可以有agent来操作
# -*- coding: utf-8 -*-

import cv2
import random
from DQN_Class import DQN
import os
import pandas as pd
from getkeys import key_check
import directkey

import time
import keyboard as kb
import numpy as np
from pynput.keyboard import Controller, Key, Listener
from script.screen_start_pause import start_game
from script.screen_enemy import enemy_fun
from script.screen_mytank import mytank
from script.screen_num_rec import num_rec
from script.screen_test import grab_screen

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#停止判断
stop_flag = False

def on_press(key):
    global stop_flag
    if key == Key.esc:
        stop_flag = not stop_flag

if __name__ == '__main__':
    window_global = (10,91,640,685)# 全局窗口
    window_base = (296,620,315,685)# 基地窗口
    
    keyboard = Controller()
    episode = 1 #轮次
    flag = 0 # 判断输赢
    # 创建一个全局的停止标志
    
    
    # 启动键盘监听器
    listener = Listener(on_press=on_press)
    listener.start()  
    
    while True:
        # 创建一个键盘控制器
        start_game()# 首先运行开始游戏程序
        print("第",episode,"轮")
        episode+=1
        time.sleep(2)
        # 首先捕获基地窗口的屏幕
        prev_screen = grab_screen(window_base)

        while True:
            
            # 检查停止标志
            if not stop_flag:
                print("游戏暂停")
                while not stop_flag:
                    time.sleep(0.1)
                print("游戏开始")
            
            # 持续捕获窗口的屏幕内容
            curr_screen = grab_screen(window_base)
            #如果我方坦克不存在并且此时我方坦克数量为0，则为输
            
            if num_rec() == '0': 
                print("1.你要寄了，你要寄了")
                time.sleep(1)
                screen_global = grab_screen(window_global)
                mask = np.all(screen_global[:,:,:3] == (0, 74, 156), axis=-1)
                found_pixels = np.sum(mask)
                if mytank() is False and found_pixels > 0:
                    print("你全军覆没了，游戏失败")
                    time.sleep(12)
                    keyboard.press('r')
                    keyboard.release('r')
                    break
                
            
            elif enemy_fun() == 0:
                print("3.现在是enemy_fun分支")
                time.sleep(3)
                print("游戏通关，即将进入下一关")
                time.sleep(3)
                break
            
            #如果基地被毁，则为输
            elif not np.array_equal(prev_screen, curr_screen):
                print("2.现在是exist_base分支")
                print("你基地炸了，游戏失败")
                time.sleep(10)
                keyboard.press('r')
                keyboard.release('r')
                break
    # 停止监听器
    listener.stop() 
