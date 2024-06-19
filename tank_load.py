import cv2
import time
import numpy as np
from DQN_Class_test import DQN
from getkeys import key_check
import directkey
from script.screen_start_pause import start_game
from script.screen_mytank import mytank
from script.screen_num_rec import num_rec
from script.screen_test import grab_screen

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 停止判断函数
def pause_game(paused):
    keys = key_check()
    if 'ESC' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            if 'ESC' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

def take_action(action):
    if action == 0:  # 不做任何操作
        pass
    elif action == 1:
        directkey.go_forward()  # w
    elif action == 2:
        directkey.go_back()  # s
    elif action == 3:
        directkey.go_left()  # a
    elif action == 4:
        directkey.go_right()  # d
    elif action == 5:
        directkey.attack()  # j

window_enemy = (859,429,940,447)# 敌军数量窗口
window_self = (693,655,715,700)# 我方坦克数量窗口
window_global = (247,400,661,817)# 全局窗口
window_base = (447,800,461,817)# 基地窗口

WIDTH = 132
HEIGHT = 132
action_size = 6  # 动作空间维度

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"



# 加载并运行模型的主程序
def main():
    paused = True
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)

    while True:
        start_game()  # 首先运行开始游戏程序

        # 获取灰度全局窗口
        screen_gray = cv2.cvtColor(grab_screen(window_global), cv2.COLOR_BGR2GRAY)
        # 获取状态
        station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
        station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]
        
        prev_screen = grab_screen(window_base)
        while True:
            curr_screen = grab_screen(window_base)
            next_self_tank = num_rec(window_self)  # 当前我方坦克的数量
            next_enemy_tank = num_rec(window_enemy)  # 观察敌方坦克变化
            
            # 终止条件: 基地被毁、敌人数量为0或我方坦克数量为0
            if not np.array_equal(prev_screen, curr_screen):
                print("基地被炸，游戏失败")
                break
            elif next_enemy_tank == 0:
                print("游戏通关，进入下一关")
                break
            elif next_self_tank == 1:
                print("我方全军覆没，游戏失败")
                break

            # 进行预测并采取动作
            action = agent.Choose_Action(station)
            take_action(action)

            # 获取下一状态
            screen_gray = cv2.cvtColor(grab_screen(window_global), cv2.COLOR_BGR2GRAY)
            next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
            next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]
            
            station = next_station
            
            paused = pause_game(paused)

if __name__ == '__main__':
    main()