#运行该主程序，首先停5秒，然后运行screen_start.py文件，即开始游戏
#然后继续停3秒，之后就可以有agent来操作
# -*- coding: utf-8 -*-

import cv2
import random
from DQN_Class_o import DQN
import pandas as pdd
from getkeys import key_check
import directkey

import time
import keyboard as kb
import numpy as np
from pynput.keyboard import Controller, Key, Listener
from script.screen_start_pause import start_game
from script.screen_mytank import mytank
from script.screen_num_rec import num_rec
from script.screen_test import grab_screen
from script.pvz_cheat import num_enemy

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#停止判断
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
            # pauses game and can get annoying.
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
    if action == 0: #表示什么都不做
        pass
    elif action == 1:
        directkey.go_forward() # w
    elif action == 2:
        directkey.go_back() # s
    elif action == 3:
        directkey.go_left() # a
    elif action == 4:
        directkey.go_right() # d
    elif action == 5:
        directkey.attack() # j
        
def action_judge(self_tank, enemy_tank, next_self_tank, next_enemy_tank, isbase_flag, flag, stop, emergence_break):
    if isbase_flag == 1: #基地被毁,游戏结束
        reward = -100
        print("基地炸了扣100分")
        return reward, stop, emergence_break
    if next_enemy_tank < enemy_tank: #敌人数量减少
        reward = 500
        print("奖励500分")
        return reward, stop, emergence_break  
    if next_self_tank < self_tank or flag == 0: #我方数量减少
        reward = -30
        print("你没了一条命扣30分")
        return reward, stop, emergence_break          
    if next_enemy_tank == 1 and stop == 0: #此时地图上还剩下1个敌人，说明游戏快结束了
        print("奖励一次100分")
        reward = 100
        stop = 1 #表示仅此奖励一次
        emergence_break = 100
        return reward, stop, emergence_break
    # 如果其他条件都不满足, 返回默认值
    return 0, stop, emergence_break  # 返回默认reward为0

window_enemy = (859,429,940,447)# 敌军数量窗口
window_self = (693,655,715,700)# 我方坦克数量窗口
window_global = (247,400,661,817)# 全局窗口
window_base = (447,800,461,817)# 基地窗口
DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
Big_BATCH_SIZE = 32
EPISODES = 19000
paused = False
target_step = 0#用来计算什么时候更新target网络
num_step = 0#更新网络次数
action_size = 6#动作空间维度
WIDTH = 132
HEIGHT = 132
UPDATE_STEP = 50

if __name__ == '__main__':

    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    keyboard = Controller()# 创建一个键盘控制器
    
    for episode in range(1, EPISODES+1):

        start_game()# 首先运行开始游戏程序
        print("第",episode,"轮")
        
        #获取灰度全局窗口
        screen_gray = cv2.cvtColor(grab_screen(window_global),cv2.COLOR_BGR2GRAY)
        #获取状态
        station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        
        # 捕获基地窗口的屏幕
        prev_screen = grab_screen(window_base)
        
        flag = 999 # 判断输赢
        isbase_flag = 0 #判断是否是因为基地被毁而输的
        total_reward = 0#总奖励
        done = 0#终止条件
        stop = 0#防止奖励多次
        emergence_break = 0#紧急停止训练
        self_tank = num_rec(window_self)#当前我方坦克的数量
        enemy_tank = num_enemy()#观察敌方坦克变化
        circle_switch = 0#循环穿透开关
        pre_action = 0
        
        #清空key列表
        clearn = key_check()
        
        station = np.array(station).reshape(-1,HEIGHT,WIDTH,1)[0]
        while True:
            
            
            # 持续捕获基地窗口的屏幕内容
            curr_screen = grab_screen(window_base)
            
            next_self_tank = num_rec(window_self)#当前我方坦克的数量
            #check_sleep = 1 if next_self_tank > 0 else 0
            next_enemy_tank = num_enemy()#观察敌方坦克变化
            
            
            #以下均为终止条件
            # #如果基地被毁，则为输
            if not np.array_equal(prev_screen, curr_screen):
                print("你基地炸了，游戏失败")
                flag = 0 #输了
                isbase_flag = 1
                done = 1
                    
            elif next_enemy_tank == 0:
                print("游戏通关，即将进入下一关")
                flag = 1 #赢了
                done = 1
            
            #如果我方坦克不存在并且此时我方坦克数量为0，则为输
            elif next_self_tank == 0:  #加速训练，测试只能有两条命
                #print("1.你要寄了，你要寄了")
                #if check_sleep == 0:
                #    time.sleep(0.2)
                #screen_global = grab_screen(window_global)
                #mask = np.all(screen_global[:,:,:3] == (12, 77, 203), axis=-1)
                #found_pixels = np.sum(mask)
                #if mytank() is False and found_pixels > 0:
                print("你全军覆没了，游戏失败")
                flag = 0 #输了
                done = 1   
             
            if stop == 2:
                action = agent.Choose_Action(station)
                take_action(action)

            #这里是训练过程逻辑
            if emergence_break != 100:
                action = agent.Choose_Action(station)
                take_action(action)
                
                #获取下一状态
                screen_gray = cv2.cvtColor(grab_screen(window_global),cv2.COLOR_BGR2GRAY)
                next_station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
                next_station = np.array(next_station).reshape(-1,HEIGHT,WIDTH,1)[0]
                
                target_step+=1
                #获取反馈
                reward, stop, emergence_break = action_judge(self_tank, enemy_tank,
                                                                next_self_tank, next_enemy_tank,
                                                                isbase_flag, flag, stop, emergence_break)
                
                if action != 0 and pre_action == action:#鼓励探索并且保证之前的动作与之前要不一致，即防止兜圈
                    reward+=3
                elif action != 0 and pre_action != action:
                    reward+=2
                #保存数据到经验池
                agent.Store_Data(station, action, reward, next_station, done)
                
                if len(agent.replay_buffer) > Big_BATCH_SIZE:
                    num_step += 1
                    # save loss graph
                    # print('train')
                    agent.Train_Network(Big_BATCH_SIZE, num_step)
                if target_step % UPDATE_STEP == 0:
                    agent.Update_Target_Network()
                    #更新q_target网络
                    
                    
                self_tank = next_self_tank#当前我方坦克的数量
                enemy_tank = next_enemy_tank#观察敌方坦克变化
                station = next_station
                pre_action = action
                total_reward += reward
                
                #print("。。。。。。。。。。。。。。。。。。。。。。")
            
            elif emergence_break==100 and stop == 1: #表示游戏快要干碎敌人了，但还没结束，此时要保存数据，停止训练，等待游戏结束
                stop += 1 #保证跳过这两个判断，保存模型一次即可
                #保存模型
                agent.save_model()
            
            paused = pause_game(paused)
                
            if done == 1:
                break
            
        if episode % 10 == 0:
            agent.save_model()
            print("epsilon为",agent.epsilon)
            
        print('episode: ', episode, '预估平均奖励:', total_reward/target_step)


            