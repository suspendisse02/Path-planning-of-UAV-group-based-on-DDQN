from __future__ import division
import time
import numpy as np
import math
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


#  UAV simulator: include all the information for a UAV
class UAV:
    def __init__(self, start_position, start_direction, step_number):
        self.position = start_position    # 初始位置
        self.direction = start_direction  # 初始方向
        # self.velocity = velocity        # 速度设置
        self.step_number = step_number    # 一架无人机达到覆盖点的步数
        # self.trajectory = trajectory            # 轨迹记录


# Enviroment Simulator: Provide states and rewards to agents.
# Evolve to new state based on the actions taken by the UAVs.
class Environ(tk.Tk, object):
    def __init__(self, num_uavs, down_lane, up_lane, left_lane, right_lane, width, height):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height
        self.UAVs = []
        self.num_UAVs = num_uavs                                # 16架无人机
        self.step_numbers = np.zeros(self.num_UAVs)             # 每架飞机的步数
        self.trajectories = [[] for _ in range(self.num_UAVs)]  # 每架飞机的轨迹
        # self.reward_idx = np.zeros(self.num_UAVs)             # 每架飞机的奖励
        # self.punish_idx = np.zeros(self.num_UAVs)             # 每架飞机的惩罚
        self.task_success = 0                                   # 完成100%覆盖率的次数
        self.n_step = 0  # ?
        self.success = False

        super(Environ, self).__init__()
        self.title('Environ')
        ww = 500  # 画布宽
        wh = 500  # 画布高
        self.geometry("%dx%d" % (ww, wh))

    # UAV(起始位置,起始方向,起始速度,飞行步数，出界次数，轨迹)
    def add_new_UAVs(self, start_position, start_direction, step_number):
        self.UAVs.append(UAV(start_position, start_direction, step_number))

    # 随机添加16架无人机(开始位置 开始方向 步数)
    def add_new_UAVs_by_number(self, n):
        x_index = [[1, 11], [11, 15], [15, 23], [1, 11], [15, 23], [1, 11], [11, 15], [15, 23]]
        y_index = [[1, 11], [1, 11], [1, 11], [11, 15], [11, 15], [15, 23], [15, 23], [15, 23]]
        for i in range(n):
            index = i // 2
            start_direction = np.random.randint(4)  # 初始方向在0~3中随机选择
            position_x = 10 * (2 * np.random.randint(x_index[index][0], x_index[index][1]) - 1) + 30
            position_y = 10 * (2 * np.random.randint(y_index[index][0], y_index[index][1]) - 1) + 30
            start_position = [position_x, position_y]
            step_number = 0
            trajectory_start = [start_position]
            self.trajectories[i] = trajectory_start
            self.add_new_UAVs(start_position, start_direction, step_number)

    # 根据UAVs速度和方向调整其位置,training时不画出轨迹
    def renew_positions_train(self, i):
        # ========================================================
        # This function update the position of each UAV
        # ========================================================
        target = np.zeros([4, 4])   # 覆盖区域用矩阵表示
        if i < self.num_UAVs:
            step_distance = 20      # 更新的距离是网格的长度                                                                      '
            if self.UAVs[i].direction == 0:  # down
                position_new = [self.UAVs[i].position[0], self.UAVs[i].position[1] + step_distance]
            elif self.UAVs[i].direction == 1:  # up
                position_new = [self.UAVs[i].position[0], self.UAVs[i].position[1] - step_distance]
            elif self.UAVs[i].direction == 2:  # right
                position_new = [self.UAVs[i].position[0] + step_distance, self.UAVs[i].position[1]]
            else:
                assert self.UAVs[i].direction == 3  # left
                position_new = [self.UAVs[i].position[0] - step_distance, self.UAVs[i].position[1]]

            self.UAVs[i].position = position_new

        punish, reward = self.adjust_positions_train()

        for j in range(self.num_UAVs):
            if (230 < self.UAVs[j].position[0] < 310) and (230 < self.UAVs[j].position[1] < 310):
                x1 = (self.UAVs[j].position[0] - 230) // 20
                y1 = (self.UAVs[j].position[1] - 230) // 20
                target[x1, y1] = 1
        cover_rate = np.sum(target == 1) / target.size  # 计算覆盖率：标记为1的数目/16
        if cover_rate == 1:  # 完成100%覆盖率则重新随机生成起始位置和方向
            self.success = True
            self.task_success += 1  # 记录百分百覆盖次数
            print("task success in training")
            self.new_random_game()
        else:
            self.success = False

        return punish, reward

    # testing时，更新位置时画出轨迹
    def renew_positions_test(self, i):
        target = np.zeros([4, 4])  # 覆盖区域用矩阵表示
        if i < self.num_UAVs:
            step_distance = 20  # 更新的距离是网格的长度                                                                      '
            if self.UAVs[i].direction == 0:  # down
                position_new = [self.UAVs[i].position[0], self.UAVs[i].position[1] + step_distance]
            elif self.UAVs[i].direction == 1:  # up
                position_new = [self.UAVs[i].position[0], self.UAVs[i].position[1] - step_distance]
            elif self.UAVs[i].direction == 2:  # right
                position_new = [self.UAVs[i].position[0] + step_distance, self.UAVs[i].position[1]]
            else:
                assert self.UAVs[i].direction == 3  # left
                position_new = [self.UAVs[i].position[0] - step_distance, self.UAVs[i].position[1]]

            self.UAVs[i].position = position_new
            self.UAVs[i].step_number += 1
            self.step_numbers[i] += 1
            self.trajectories[i].append(position_new)  # 给第i架飞机的轨迹赋值

        punish, reward = self.adjust_positions_test()

        for j in range(self.num_UAVs):
            if (230 < self.UAVs[j].position[0] < 310) and (230 < self.UAVs[j].position[1] < 310):
                x1 = (self.UAVs[j].position[0] - 230) // 20
                y1 = (self.UAVs[j].position[1] - 230) // 20
                target[x1, y1] = 1
        cover_rate = np.sum(target == 1) / target.size  # 计算覆盖率：标记为1的数目/16
        # print("coverage_rate", cover_rate)
        if cover_rate == 1:  # 完成100%覆盖率则重新随机生成起始位置和方向
            self.success = True
            self.task_success += 1  # 记录百分百覆盖次数
            print("task success in testing")
            self.new_random_game()
        else:
            self.success = False

        return punish, reward

    # 在renew_position之前先判断是否出界,taining时不计入步数
    def adjust_positions_train(self):
        position_reward = 0
        position_punish = 0
        for i in range(self.num_UAVs):
            if self.UAVs[i].position[0] < 30 or self.UAVs[i].position[0] > 470 or self.UAVs[i].position[1] < 30 or self.UAVs[i].position[1] > 470:
                if self.UAVs[i].position[0] >= 30 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] + 20]
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1] + 20]
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] >= 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1]]
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] >= 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1]]
                if self.UAVs[i].position[0] <= 470 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] + 20]
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1] + 20]
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] <= 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1]]
                if self.UAVs[i].position[0] <= 470 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] - 20]
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1] - 20]
                if self.UAVs[i].position[0] >= 30 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] - 20]
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] <= 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1]]
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1] - 20]

                position_punish += 1
            else:
                position_reward += 2
        return position_punish/self.num_UAVs, position_reward/self.num_UAVs

    # 在renew_position之前先判断是否出界,testing时，若出界则计入返回步数
    def adjust_positions_test(self):
        position_reward = 0
        position_punish = 0
        for i in range(self.num_UAVs):
            if self.UAVs[i].position[0] < 30 or self.UAVs[i].position[0] > 470 or self.UAVs[i].position[1] < 30 or self.UAVs[i].position[1] > 470:
                if self.UAVs[i].position[0] >= 30 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] + 20]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1] + 20]
                    self.UAVs[i].step_number += 2
                    self.step_numbers[i] += 2
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] >= 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1]]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] >= 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1]]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] <= 470 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] + 20]
                    self.UAVs[i].direction = 0
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] < 30:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1] + 20]
                    self.UAVs[i].step_number += 2
                    self.step_numbers[i] += 2
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] <= 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1]]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] <= 470 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] - 20]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] > 470 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] - 20, self.UAVs[i].position[1] - 20]
                    self.step_numbers[i] += 2
                if self.UAVs[i].position[0] >= 30 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0], self.UAVs[i].position[1] - 20]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] <= 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1]]
                    self.UAVs[i].step_number += 1
                    self.step_numbers[i] += 1
                if self.UAVs[i].position[0] < 30 and self.UAVs[i].position[1] > 470:
                    self.UAVs[i].position = [self.UAVs[i].position[0] + 20, self.UAVs[i].position[1] - 20]
                    self.UAVs[i].step_number += 2
                    self.step_numbers[i] += 2

                position_punish += 1
            else:
                position_reward += 2
        return position_punish/self.num_UAVs, position_reward/self.num_UAVs

    def coverage_reward(self):
        target = np.zeros([4, 4])  # 覆盖区域用矩阵表示
        for j in range(self.num_UAVs):
            if (230 < self.UAVs[j].position[0] < 310) and (230 < self.UAVs[j].position[1] < 310):
                x1 = (self.UAVs[j].position[0] - 230) // 20
                y1 = (self.UAVs[j].position[1] - 230) // 20
                target[x1, y1] = 1
        cover_rate = np.sum(target == 1) / target.size  # 计算覆盖率：标记为1的数目/16
        coverage_reward = 10 * (cover_rate - 0.5)
        return coverage_reward

    def visual_roads(self):
        self.canvas = tk.Canvas(self, bg='white', height=500, width=500)
        for i in range(len(self.up_lanes[2:])):
            x0, y0, x1, y1, = self.up_lanes[i+2], 30, self.up_lanes[i+2], 470
            self.canvas.create_line(x0, y0, x1, y1, dash=(4, 4))
        for j in range(len(self.down_lanes[:-2])):
            x0, y0, x1, y1 = self.down_lanes[j], 30, self.down_lanes[j], 470
            self.canvas.create_line(x0, y0, x1, y1, dash=(4, 4))
        for k in range(len(self.right_lanes[:-2])):
            x0, y0, x1, y1 = 30, self.right_lanes[k], 470, self.right_lanes[k]
            self.canvas.create_line(x0, y0, x1, y1, dash=(4, 4))
        for l in range(len(self.left_lanes[2:])):
            x0, y0, x1, y1 = 30, self.left_lanes[l+2], 470, self.left_lanes[l+2]
            self.canvas.create_line(x0, y0, x1, y1, dash=(4, 4))
        for m in range(2):
            x0, y0, x1, y1, = self.up_lanes[m], 0, self.up_lanes[m], self.height
            self.canvas.create_line(x0, y0, x1, y1, fill='red', dash=(4, 4))
        for n in range(2):
            x0, y0, x1, y1, = self.down_lanes[-1-n], 0, self.down_lanes[-1-n], self.height
            self.canvas.create_line(x0, y0, x1, y1, fill='red', dash=(4, 4))
        for p in range(2):
            x0, y0, x1, y1 = 0, self.left_lanes[p], self.width, self.left_lanes[p]
            self.canvas.create_line(x0, y0, x1, y1, fill='red', dash=(4, 4))
        for q in range(2):
            x0, y0, x1, y1 = 0, self.right_lanes[-1-q], self.width, self.right_lanes[-1-q]
            self.canvas.create_line(x0, y0, x1, y1, fill='red', dash=(4, 4))

        self.target = self.canvas.create_rectangle([270 - 40, 270 - 40, 270 + 40, 270 + 40], fill='black')

        # 无人机用红色质点表示
        for u in range(self.num_UAVs):
            self.canvas.create_oval([self.UAVs[u].position[0]-5, self.UAVs[u].position[1]-5, self.UAVs[u].position[0]+5, self.UAVs[u].position[1]+5], fill='red')
        # 轨迹用绿色实线表示
        for i in range(self.num_UAVs):
            for j in range(len(self.trajectories[i])-1):
                x0, y0 = self.trajectories[i][j][0], self.trajectories[i][j][1]
                x1, y1 = self.trajectories[i][j+1][0], self.trajectories[i][j+1][1]
                self.canvas.create_line(x0, y0, x1, y1, fill='green', width=4)

        self.canvas.pack()
        tk.Tk.mainloop(self)
    """
    def test(self):
        self.num_UAVs = 16
        self.add_new_UAVs_by_number(int(self.num_UAVs))
        steps = 1000
        for i in range(steps):
            for j in range(self.num_UAVs):
                self.renew_positions_test(j)
            self.coverage_reward()
            print("test_steps: ", i)
        self.visual_roads()
    """
    def act_for_training(self, i):  # 待改进
        # =============================================
        # This function gives rewards for training
        # ===========================================
        p_punish, p_reward = self.renew_positions_train(i)
        c_reward = self.coverage_reward()
        total_rewards = p_reward + c_reward - p_reward
        return total_rewards

    # 没用到
    def act_asyn(self, i):
        self.n_step += 1
        if self.n_step % 1 == 0:
            p_punish, p_reward = self.renew_positions_train(i)
            c_reward = self.coverage_reward()
            total_rewards = p_reward + c_reward - p_reward
        return total_rewards
    # 没用到
    def act(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1
        next_state = []
        for i in range(self.num_UAVs):
            self.UAVs[i].direction = actions[i]
            self.renew_positions_train(i)
            next_state.append((self.UAVs[i].position))
        return next_state

    def new_random_game(self):
        # make a new game
        self.n_step = 0
        self.num_UAVs = 16
        self.UAVs = []
        self.step_numbers = np.zeros(self.num_UAVs)
        self.trajectories = [[] for _ in range(self.num_UAVs)]  # 每架飞机的轨迹,list
        self.task_success = 0
        self.reward_idx = np.zeros(self.num_UAVs)
        self.punish_idx = np.zeros(self.num_UAVs)  #
        self.add_new_UAVs_by_number(int(self.num_UAVs))
        self.test_time_count = 10
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.002  # 2ms update time for testing
        self.update_time_asyn = 0.0002  # 0.2 ms update one subset of the uavs; for each uav, the update time is 2 ms

if __name__ == "__main__":
    up_lane = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
               240 + 20 / 2 + 20, 320 + 20 / 2, 320 + 20 + 20 / 2, 400 + 20 / 2, 400 + 20 / 2 + 20]
    down_lane = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 / 2 - 20, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                 320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 - 20 / 2, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2,
                 490]
    left_lane = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
                 240 + 20 + 20 / 2, 320 + 20 / 2, 320 + 20 / 2 + 20, 400 + 20 / 2, 400 + 20 / 2 + 20]
    right_lane = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 - 20 / 2, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                  320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 / 2 - 20, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2,
                  490]
    width = 500
    height = 500
    Env = Environ(16, down_lane, up_lane, left_lane, right_lane, width, height)




