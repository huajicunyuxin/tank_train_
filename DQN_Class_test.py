
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np
import os

#经验回放池的大小
replay_size = 1500
#batch的大小
mini_batch_size = 16
big_batch_size = 128

#dqn超参数
#折扣因素
GAMMA = 0.9
#epsilon参数
INITIAL_EPSILON = 0.41
FINAL_EPSILON = 0.01

class DQN():
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        #输入状态
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        
        #输入动作
        self.action_dim = action_space
        
        #创建经验回放池
        self.replay_buffer = deque()
        
        #创建q神经网络
        self.create_Q_network()
        
        #更新权重方法adam
        self.create_updating_method()
        
        #epsilon设置
        self.epsilon = INITIAL_EPSILON
        
        #模型路径
        self.model_path = model_file + "/save_model.ckpt"
        #模型文件
        self.model_file = model_file
        #模型日志
        self.log_file = log_file

        #创建一个交互式的 TensorFlow 会话
        self.session = tf.InteractiveSession()
        if os.path.exists(self.model_file):
            print("模型存在，加载模型ing\n")
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, self.model_path)
        else:
            print("模型不存在，创建新模型ing\n")
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        
        #神经网络视图，把框图保存到文件中，加载到浏览器中观看
        self.writer = tf.summary.FileWriter(self.log_file, self.session.graph)
        #路径中不要有中文字符，否则加载不进来
        self.merged = tf.summary.merge_all()       
        #把所有summary合并在一起，就是把所有loss,w,b这些的数据打包

    # 初始化w权重
    def weight_value(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)#几维张量，标准差为0.1
        return tf.Variable(initial)
    
    # 初始化b偏离值
    def b_value(self, shape):
        initial = tf.constant(0.01,shape=shape)#所有数值为0.01，shape维的张量
        return tf.Variable(initial)
    
    # 二维卷积运算
    """
    input（在你给的例子里是 x）：一个4维的张量，应该具有 [batch, in_height, in_width, in_channels] 的形状。
    filter（在你的例子里是 W）：一个4维的张量，应该具有 [filter_height, filter_width, in_channels, out_channels] 的形状。
    strides：一个4元素的列表，定义了输入张量每个维度的滑动步长。在大多数情况下，都会设置为 [1, stride, stride, 1]，意味着我们不会在批次和通道维度上进行滑动，只会在高度和宽度维度上进行滑动。
    padding：一个字符串，要么是 'VALID'，要么是 'SAME'。如果是 'VALID'，那么不会在输入的各边界上进行填充操作。如果是 'SAME'，那么会在输入的各边界上进行填充操作，以确保输出的高度和宽度与输入相同。   
    """    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    #最大池化
    """
    value（这里是 x）：一个4维输入张量，通常形状是 [batch, height, width, channels]。
    ksize：一个长度为4的列表，定义了池化窗口的大小。通常形式为 [1, height, width, 1]，表示在每个高度和宽度上进行池化。
    strides：一个长度为4的列表，定义了池化窗口移动的步长。常见形式是 [1, stride, stride, 1]，意为高度和宽度维度上每次的移动步长。
    padding：一个字符串，'SAME' 或 'VALID'，指定了填充方式。'VALID' 表示不用零进行填充，'SAME' 表示用零进行填充，以保持输入和输出的空间维度一致。
    
    """
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def create_Q_network(self):
        #首个维度是 None，表示这个占位符可以接收任意数量的样本。
        with tf.name_scope('inputs'):
            #在当前命名范围下创建一个占位符 state_input。这是一个用来接收输入数据的节点。
            self.state_input = tf.placeholder("float", [None, self.state_h, self.state_w, 1])

        with tf.variable_scope('current_net'):
            
            """
            5, 5：卷积核的尺寸（高和宽）。
            1：输入特征图（feature map）的数量，也就是上一层网络的输出深度。
            32：表示卷积核的数量，也就是这一层网络的输出深度。
            """
            
            #卷积层
            #第一层神经网络参数定义
            W_conv1 = self.weight_value([5,5,1,32])
            b_conv1 = self.b_value([32])
            #第二层神经网络参数定义
            W_conv2 = self.weight_value([5,5,32,64])
            b_conv2 = self.b_value([64])
            #第一层卷积操作
            h_conv1 = tf.nn.relu(self.conv2d(self.state_input, W_conv1) + b_conv1)   
            #第一层最大池化
            h_pool1 = self.max_pool_2x2(h_conv1)
            #第二层卷积操作
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
            #第二层最大池化
            h_pool2 = self.max_pool_2x2(h_conv2) 
            
            #全连接层
            #第一层全连接层神经网络参数定义
            #平铺（flatten）之后的矩阵是一个一维向量
            W1 = self.weight_value([int((self.state_w/4) * (self.state_h/4) * 64), 512])
            b1 = self.b_value([512])
            #第二层全连接层神经网络参数定义
            W2 = self.weight_value([512, 256])
            b2 = self.b_value([256])
            #第三层全连接层神经网络参数定义
            W3 = self.weight_value([256, self.action_dim])
            b3 = self.b_value([self.action_dim])
            
            #把卷积层第二层池化后输出的矩阵平铺，-1表示“自动计算
            h_conv2_flat = tf.reshape(h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])
            #矩阵相乘
            h_layer_one = tf.nn.relu(tf.matmul(h_conv2_flat, W1) + b1)
            #dropout防止过拟合
            #第一层全连接层输出
            h_layer_one = tf.nn.dropout(h_layer_one, 1)
            #第一层输出与生成的随机变量进行矩阵相乘
            h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)
            # dropout防止过拟合
            #第二层全连接层输出
            h_layer_two = tf.nn.dropout(h_layer_two, 1)
            #第二层输出与生成的随机变量进行矩阵相乘
            Q_value = tf.matmul(h_layer_two, W3) + b3
            #dropout，最后输出的就是q值
            self.Q_value = tf.nn.dropout(Q_value, 1)
            
        #target网络
        with tf.variable_scope('target_net'):
            #卷积层
            t_W_conv1 = self.weight_value([5,5,1,32])
            t_b_conv1 = self.b_value([32])
            t_W_conv2 = self.weight_value([5,5,32,64])
            t_b_conv2 = self.b_value([64])
            #卷积操作
            t_h_conv1 = tf.nn.relu(self.conv2d(self.state_input, t_W_conv1) + t_b_conv1)#卷积
            t_h_pool1 = self.max_pool_2x2(t_h_conv1)#池化
            t_h_conv2 = tf.nn.relu(self.conv2d(t_h_pool1, t_W_conv2) + t_b_conv2)#卷积
            t_h_pool2 = self.max_pool_2x2(t_h_conv2)#池化
            
            #全连接层
            t_W1 = self.weight_value([int((self.state_w/4) * (self.state_h/4) * 64), 512])
            t_b1 = self.b_value([512])
            t_W2 = self.weight_value([512, 256])
            t_b2 = self.b_value([256])
            t_W3 = self.weight_value([256, self.action_dim])
            t_b3 = self.b_value([self.action_dim])
            
            t_h_conv2_flat = tf.reshape(t_h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])#卷积输出的矩阵进行平铺
            t_h_layer_one = tf.nn.relu(tf.matmul(t_h_conv2_flat, t_W1) + t_b1)
            t_h_layer_one = tf.nn.dropout(t_h_layer_one, 1)#dropout
            t_h_layer_two = tf.nn.relu(tf.matmul(t_h_layer_one, t_W2) + t_b2)
            t_h_layer_two = tf.nn.dropout(t_h_layer_two, 1)#dropout
            #最后一层输出q_target值，一般来说会几轮才会更新
            target_Q_value = tf.matmul(t_h_layer_two, t_W3) + t_b3
            self.target_Q_value = tf.nn.dropout(target_Q_value, 1)#最后输出q_target
            
        #这行代码获取了所有 "current_net" 中的变量（即参数），并将它们存储在 e_params 列表中。
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        #这行代码获取了"target_net" 中的所有变量，并将它们保存在 t_params 列表中。
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        #下面的代码是要把"current_net" 中的变量复制到"target_net" 中
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_updating_method(self):
        #动作输入
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        #先用一个占位符表示target_Q_value的值
        self.y_input = tf.placeholder("float", [None])
        #提取Q_action的值,求和可以直接求得标量值
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        #求均方误差
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        #画图
        tf.summary.scalar('loss',self.cost)
        #放进优化器里进行迭代更新
        with tf.name_scope('train_loss'):
            # use the loss to optimize the network
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def Choose_Action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
        self.state_input: [state]
        })[0]
        
        self.epsilon = 0.1
        if random.random() <= self.epsilon:
            #如果低于epslion值，则给一个随机动作上去
            #self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            #如果大于epslion值，则选择q值最大的动作
            #self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)
        
    def Store_Data(self, state, action, reward, next_state, done):
        #首先创建了一个与动作空间维度相同、全零的向量(one_hot_action)，然后在执行的动作的索引处设置值为1。这是对执行的动作进行的one-hot编码，标记了在各可能动作中哪一个被执行。
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        #经验池的长度大于replay_size时，就把第一个经验弹出
        if len(self.replay_buffer) > replay_size:
            self.replay_buffer.popleft()
    
    def Train_Network(self, BATCH_SIZE, num_step):
        #1.从经验池里获取经验数据
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        #状态
        state_batch = [data[0] for data in minibatch]
        #动作
        action_batch = [data[1] for data in minibatch]
        #奖励
        reward_batch = [data[2] for data in minibatch]
        #下一状态
        next_state_batch = [data[3] for data in minibatch]
        
        #2.计算target_Q_value值
        #表示target_Q_value的batch
        y_batch = []
        #以state_input为键，输入next_state_batch，即下一状态的q值，形成Q_value_batch
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done: #终态的时候，即使奖励即为q值
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        
        #3.运行优化器进行梯度下降
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            #y即为更新后的Q值,与Q_action构成损失函数更新网络
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        if num_step % 100 == 0:
            #保存计算图
            result = self.session.run(self.merged,feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
            })
            # 把merged的数据放进writer中才能画图
            self.writer.add_summary(result, num_step)

    def Update_Target_Network(self):
        # 更新 target Q netowrk
        self.session.run(self.target_replace_op)

    def save_model(self):
        self.save_path = self.saver.save(self.session, self.model_path)
        print("Save to path:", self.save_path)