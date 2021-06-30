from __future__ import division, print_function  # 导入future后，用的都是python3的版本
import random
import tensorflow as tf
from agent import Agent
from Environment import *
flags = tf.app.flags  # 在执行main函数之前首先进行flags的解析,也就是说TensorFlow通过设置flags来传递tf.app.run()所需要的参数

# Model 添加命令行参数
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS  # 实例化，从对应的命令行取出参数

# Set random seed
tf.set_random_seed(FLAGS.random_seed)  # 设置全局随机种子，可以跨Session生成相同的随机数，种子数为变量随机生成次数
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')  # 通过指定分隔符对字符串进行切片，默认分割次数为“-1”即所有
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  up_lanes = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
              240 + 20 / 2 + 20, 320 + 20 / 2, 320 + 20 + 20 / 2, 400 + 20 / 2, 400 + 20 / 2 + 20]
  down_lanes = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 / 2 - 20, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 - 20 / 2, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2, 490]
  left_lanes = [20 / 2, 20 / 2 + 20, 80 + 20 / 2, 80 + 20 / 2 + 20, 160 + 20 / 2, 160 + 20 + 20 / 2, 240 + 20 / 2,
                240 + 20 + 20 / 2, 320 + 20 / 2, 320 + 20 / 2 + 20, 400 + 20 / 2, 400 + 20 / 2 + 20]
  right_lanes = [80 - 20 / 2 - 20, 80 - 20 / 2, 160 - 20 - 20 / 2, 160 - 20 / 2, 240 - 20 / 2 - 20, 240 - 20 / 2,
                 320 - 20 / 2 - 20, 320 - 20 / 2, 400 - 20 / 2 - 20, 400 - 20 / 2, 480 - 20 / 2 - 20, 480 - 20 / 2, 490]
  width = 500
  height = 500
  Env = Environ(16, down_lanes, up_lanes, left_lanes, right_lanes, width, height)
  # Env.new_random_game()
  # 指定了每个GPU进程中使用显存的上限，均匀地作用于所有GPU
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  config = tf.ConfigProto()  # 配置tf.Session的运算方式
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:  # 使用with tf.Session()创建上下文来执行，当上下文退出时自动释放。
    config = []
    agent = Agent(config, Env, sess)
    # agent.train()
    agent.play()
    # env = Environ()
    # env.mainloop()


if __name__ == '__main__':
    tf.app.run()
