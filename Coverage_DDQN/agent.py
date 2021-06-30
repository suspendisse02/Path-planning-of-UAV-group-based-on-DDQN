from __future__ import print_function, division
import os
import time
import random
import numpy as np
from Environment import *
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow as tf
import matplotlib.pyplot as plt

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir)
        self.env = environment
        self.max_iterations = 100000             # 最大步数限制
        self.num_UAVs = self.env.num_UAVs      # 16架无人机
        self.num_actions = 4                   # 动作集
        self.actions_by_uavs = np.zeros(self.num_UAVs)  # 16架无人机的动作
        self.reward_list = []
        self.cost = []
        self.learning_rate = 0.0005               # 学习率
        self.learning_rate_minimum = 0.0001     # 学习率衰减的最小值
        self.learning_rate_decay = 0.8          # 学习率衰减比例0.96
        self.learning_rate_decay_step = 500000  # 学习率减缓轮数
        self.target_q_update_step = 100         # 更新Q-target的频率
        self.discount = 0.5                     # 折扣因子(由0.5改为0.8),可以看的更远
        self.double_q = True                    # 使用DDQN解决overestimated问题
        self.build_dqn()
        self.training = True

    # 状态:16架无人机的横纵坐标(x0,x1...x16,y0,y1...y16), shape=32
    def get_state(self):
        # ===============
        #  Get State from the environment
        # =============
        position_x = []
        position_y = []
        for i in range(self.num_UAVs):
            position_x.append((self.env.UAVs[i].position[0]-270)/500)
            position_y.append((self.env.UAVs[i].position[1]-270)/500)
        x = np.array(position_x)
        y = np.array(position_y)
        state = np.concatenate((x, y))  # 将状态元素拼接起来
        return state

    def predict(self, s_t,  iter, test_ep=False):
        # ==========================
        #  Select actions
        # ======================
        ep = 1/(iter/1000000 + 1)                          # 动态调整贪婪系数ε，随着步数的增多探索的概率逐渐减小
        if random.random() < ep and test_ep == False:    # random.random()生成0和1之间的随机浮点数
            action = np.random.randint(0, 64)    # 探索，action为随机整数
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]  # 开发,q_action.eval在build_dqn()中定义，即输入当前状态s_t返回动作action。
        return action

    def observe(self, iter, prestate, state, reward, action):
        # -----------
        # Collect Data for Training
        # 观测的环境信息有：前一时刻状态、当前状态、奖励、行为
        # ---------
        self.memory.add(prestate, state, reward, action)  # add the state and the action and the reward to the memory
        if iter > 0:
            if iter % 10 == 0:                              # 10个为一批进行训练一次DQN
                self.q_learning_mini_batch(iter)            # training a mini batch
                self.save_weight_to_pkl()
            if iter % self.target_q_update_step == self.target_q_update_step - 1:  # 第99个iteration，更新q_target网络参数
                self.update_target_q_network()

    def train(self):
        num_game, self.update_count, ep_reward = 0., 0., 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        self.env.new_random_game()  # 重新随机生成初始位置
        for iter in (range(0, self.max_iterations)):
            if iter == 0:  # initialize set some varibles
                num_game, self.update_count, ep_reward = 0., 0., 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
            # training
            if (iter % 2000 == 1) and (iter > 0):  # 重启两次，2001,4001
                self.env.new_random_game()
            # print("training_iter: ", iter)
            self.training = True
            for i in range(self.num_UAVs):
                state_old = self.get_state()  # 获取当前状态
                action = self.predict(state_old, iter)  # 当前状态下预测动作
                # self.env.UAVs[i].direction = int(action)
                self.env.UAVs[i].direction = int(np.floor(action/self.num_UAVs))
                reward_train = self.env.act_for_training(i)
                state_new = self.get_state()
            self.observe(iter, state_old, state_new, reward_train, action)

            # testing
            if (iter % 20 == 0) and (iter > 0):  # 在第2000步的整数倍时test 10次
                self.training = False
                number_of_game = 10
                if (iter % 10000 == 0) and (iter > 0):  # 在第10000步的整数倍时test 50次
                    number_of_game = 50
                if iter == 38000:
                    number_of_game = 100
                for game_idx in range(number_of_game):
                    self.env.new_random_game()
                    test_sample = 1000
                    Reward_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        self.env.add_new_UAVs_by_number(self.num_UAVs)
                        for i in range(self.num_UAVs):  # 0~15
                            state_old = self.get_state()
                            action = self.predict(state_old, 0, True)
                            # self.env.UAVs[i].direction = int(action)
                            self.env.UAVs[i].direction = int(np.floor(action / self.num_UAVs))
                            self.env.renew_positions_test(i)
                    # self.env.visual_roads()
                    '''
                        if i % (self.num_UAVs/8) == 1:  # 2个2个更新,3/5/7/9/11/13/15
                            reward_asyn = self.env.act_asyn(i)
                            Reward_list.append(np.sum(reward_asyn))
                    '''
                self.save_weight_to_pkl()

        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def q_learning_mini_batch(self, iter):
        # Training the DQN model
        # ------
        s_t, s_t_plus_1, action, reward = self.memory.sample()
        # print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])
        if self.double_q:  # double Q learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})  # s_t+1状态下在q_eval网络中最大值函数对应的动作a^
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1,
                                                                       self.target_q_idx: [[idx, pred_a] for idx, pred_a
                                                                                           in enumerate(pred_action)]})
            # a^动作在q_target网络中的q函数
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            target_q_t = self.discount * q_t_plus_1_with_pred_action + reward  # 目标q值计算公式
        else:  # 普通DQN
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)  # 直接选取s_t+1状态在q_target网络中最大值函数q_max
            target_q_t = self.discount * max_q_t_plus_1 + reward  # 将q_max带入公式
        # training the network
        _, q_t, loss, w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t,
                                                                                  self.action: action, self.s_t: s_t,
                                                                                  self.learning_rate_step: iter})

        print('loss is ', loss)
        self.cost.append(loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def build_dqn(self):
        # --- Building the DQN -------
        self.w = {}
        self.t_w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)  # 从截断的正态分布中输出随机值,mean=0是要生成的随机值的均值，stddev=0.002是要生成的随机值的标准偏差。
        activation_fn = tf.nn.relu
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 32  # 输入无人机的状态是16架无人机的横纵坐标
        n_output = 4  # 输出每个无人机的动作,所以是16×4=64

        def encoder(x):
            # tf.truncated_normal(shape,mean,stddev)截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差stddev，则重新生成
            weights = {
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1)),
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),
            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            return layer_4, weights

        # Q-eval网络
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, n_input])
            self.q, self.w = encoder(self.s_t)  # 向encoder()输入状态，输出q值和网络参数
            self.q_action = tf.argmax(self.q, dimension=1)  # 取最大的q值下的动作

        # Q-target网络
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])

        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32', None, name='action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(self.learning_rate, self.learning_rate_step,
                                                                          self.learning_rate_decay_step,
                                                                          self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(
                self.loss)

        tf.initialize_all_variables().run()
        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def save_weight_to_pkl(self):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network()

    def play(self, n_step=100, n_episode=100, test_ep=None, render=False):
        number_of_game = 100
        self.load_weight_from_pkl()
        self.training = False
        for game_idx in range(number_of_game):
            self.env.new_random_game()
            test_sample = 200
            print('test game idx:', game_idx)
            print('The number of UAV is ', self.num_UAVs)
            average_steps = np.zeros(16)
            for k in range(test_sample):
                for u in range(self.num_UAVs):
                    state_old = self.get_state()
                    action = self.predict(state_old, 0, True)
                    # self.env.UAVs[u].direction = int(action)
                    self.env.UAVs[u].direction = int(np.floor(action / self.num_UAVs))
                    self.env.renew_positions_test(u)
        self.env.visual_roads()
        print("success_times:", self.env.task_success)
            # plt.bar(np.arrange(len(self.env.step_numbers)), average_steps)


"""
def main(_):
    up_lanes = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
                240 + 20 / 2 + 20, 320 + 20 / 2, 320 + 20 + 20 / 2, 400 + 20 / 2, 400 + 20 / 2 + 20]
    down_lanes = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 / 2 - 20, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                  320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 - 20 / 2, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2,
                  490]
    left_lanes = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
                  240 + 20 + 20 / 2, 320 + 20 / 2, 320 + 20 / 2 + 20, 400 + 20 / 2, 400 + 20 / 2 + 20]
    right_lanes = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 - 20 / 2, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                   320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 / 2 - 20, 400 - 20 / 2, 480 - 20 / 2 - 20,
                   480 - 20 / 2, 490]
    width = 500
    height = 500
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
    '''
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto()) as sess:
        config = []
        agent = Agent(config, Env, sess)
        # agent.play()
        agent.train()
        agent.play()

if __name__ == '__main__':
    tf.app.run()
"""


