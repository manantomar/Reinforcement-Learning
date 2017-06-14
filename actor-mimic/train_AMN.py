"""
    This script describes the OpenAI gym environment for the source tasks,
    samples actions from either the AMN or the expert networks and provides
    the true policy and the sampled state action pairs to train the AMN

"""

import gym
import numpy as np
import tensorflow as tf
from actor-mimic import AMN
from scipy.misc import imresize
import random
from collections import deque
import cv2

net = AMN()

game1_load_path = './pong_dqn_v4_reg_0.01/network_file_50.pkl'
game2_load_path = './pong_dqn_v4_reg_0.01/network_file_50.pkl'
#-------------------importing the pretrained models----------------------------#

import theano
import pickle

print("unpickling first game...")
pkl = open(game_1_load_path, 'rb')
game_1 = pickle.load(pkl)
pkl.close()

print("unpickling second game...")
pkl = open(game2_load_path, 'rb')
game_2 = pickle.load(pkl)


#-----------------------------------------------------------------------------#

# implementing intially only for two games
game = ['MsPacman-v0', 'Pong-v0'] # add accordingly

num_exp = len(game)

def preprocess(frame):

    gray_image = frame.mean(2)
    reshaped_image = imresize(gray_image, (84,84))
    x = np.reshape(reshaped_image, [84,84,1]).astype(np.float32)
    x *= (1.0 / 255.0) # divide by 255

    return x

def get_num_actions(game_id):


def get_AMN_policy(s_t, num_actions):

    #num_actions = get_num_actions(game_id)
    q_vals = net.build_net.eval(feed_dict = {state : s_t, num_actions : num_actions})

    one_hot = np.zeros(BATCH, num_actions)
    one_hot[:,np.argmax(q_vals, axis=1)] = 1
    AMN_policy = one_hot

    return AMN_policy

def get_true_policy(state_batch, AMN_action_batch, game_id):

    true_policy = []

    if game_id == 0:
        game = game_1
    else:
        game = game_2

    for i,s in enumerate(state_batch):
        game.state_shared.set_value(s)
        true_policy[i] = game._q_vals()

    return true_policy

def rollout(state, action, encode):

    #sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, load_path)
    print("variables restored and loaded...")

    # stores history for all games separately
    replay_memory = []
    s_t = []
    s_t1 = []

    for i in range(num_exp):
        D = deque()
        replay_memory.append(D)
        s_t.append([])
        s_t1.append([])

    num_episodes = np.zeros(num_exp)
    k = 0

    while np.max(num_episodes) < MAX_EPISODES:

        game_id = random.randint(0, num_exp - 1)
        env = gym.make(game[game_id])
        num_actions = env.action_space.n
        ob = env.reset()

        obf = preprocess(ob)
        s_t[game_id] = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (84, 84, 4))
        observations, actions = [], []

        i = 0
        print("num of episodes ", num_episodes[game_id])

        for t in range(10000):
            env.render() #optional

            q_val_AMN = get_AMN_policy(s_t[game_id], num_actions)
            # epsilon greedy policy
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                action[action_index] = 1
            else:
                action_index = np.argmax(q_val_AMN) # create instance from AMN class
                action[action_index] = 1

            ob, reward, done, info = env.step(action_index)

            obf = preprocess(ob)

            s_t1[game_id] = np.append(obf, s_t[:,:,0:3], axis = 2)
            ''' uncomment for training '''

            replay_memory[game_id].append((s_t, action, obf))
            if len(replay_memory[game_id]) > REPLAY_MEMORY:
                replay_memory[game_id].popleft()

            if train == True:

                minibatch = random.sample(replay_memory[game_id], BATCH)
                state_batch = [d[0] for d in minibatch]
                AMN_action_batch = [d[1] for d in minibatch]
                # get true action
                true_action = get_true_policy(state_batch, AMN_action_batch)
                #num_actions = get_num_actions(game_id)
                # minibatch update
                net.train_step(true_action, state_batch, AMN_action_batch, num_actions)

            s_t[game_id] = s_t1[game_id]

            if done:
                num_episodes[game_id] += 1
                break
