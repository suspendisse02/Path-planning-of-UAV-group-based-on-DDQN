from __future__ import division, print_function
import numpy as np 
from Environment import *
import matplotlib.pyplot as plt

# This py file using the random algorithm.

def main():
    up_lanes = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
               240 + 20 / 2 + 20, 320 + 20 / 2, 320 + 20 + 20 / 2, 400 + 20 / 2, 400 + 20 / 2 + 20]
    down_lanes = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 / 2 - 20, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                 320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 - 20 / 2, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2,
                 490]
    left_lanes = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
                 240 + 20 + 20 / 2, 320 + 20 / 2, 320 + 20 / 2 + 20, 400 + 20 / 2, 400 + 20 / 2 + 20]
    right_lanes = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 - 20 / 2, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                  320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 / 2 - 20, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2,
                  490]
    width = 500
    height = 500
    Rate = list()
    n = 16
    while n < 17:
        Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
        number_of_game = 50
        n_step = 100
        for game_idx in range(number_of_game):
            print(game_idx)
            Env.new_random_game()
            for i in range(n_step):
                # print(i)
                directions = np.random.randint(0, 4)
                actions = np.concatenate((directions[..., np.newaxis]))
                next_position = Env.act(actions)
            print(next_position)
            # print('percentage here is ', percent)
    '''
    plt.plot(Rate, n_Rate, 'y^-', label='Random')
    plt.xlim([0, 100])
    plt.xlabel("Time left for V2V transmission (s)")
    plt.ylabel("Probability of power selection")
    plt.legend()
    plt.grid()
    plt.show()
    '''

main()
