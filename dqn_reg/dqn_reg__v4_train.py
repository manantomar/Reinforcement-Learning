"""
    This script describes the OpenAI gym environment for the source tasks,
    samples actions from either the AMN or the expert networks and provides
    the true policy and the sampled state action pairs to train the AMN

After skipping frames we generate the following sequence:
                                                     |-a_t-|       |-a_t+1-|     |-a_t+2-|     |-a_t+3-|
eps_start : action - ob - action - ob - action - ob - action - ob - action - ob - action - ob - action - ob - action - ob
                     |-------------- state t ------------------|
                                   |---------------- state t+1 ---------------|
                                                  |---------------- state t+2 ---------------|
                                                               |---------------- state t+3 ---------------|
                                                                              |---------------- state t+4 ---------------|

"""

import gym
import numpy as np
import tensorflow as tf
from dqn_reg_v4 import net_v4
from scipy.misc import imresize
import random
from collections import deque
import cv2
import itertools

BATCH = 32
MAX_EPISODES = 10
REPLAY_MEMORY = 1000
ACTIONS = 4
epsilon = 0.3

net = net_v4(0.99, 10000, BATCH, 0.5)

def get_minibatch(D, BATCH):

    batch_id = 0
    minibatch = []


    while batch_id < BATCH:
        #print("D size", len(D))
        idx = random.randrange(len(D) - 8)
        range_idx = np.arange(idx, idx + 8)
        action_idx = np.arange(idx + 3, idx + 7)
        end_idx = idx + 3

        state_sample = [s[0] for s in D[idx : idx + 8]]
        action_sample = [s[1] for s in D[idx + 3 : idx + 7]]
        reward_sample = D[idx + 3][2]
        done_sample = D[idx + 3][3]
        print("action sample size", action_sample[0])
        minibatch.append((np.asarray(state_sample).transpose(3,1,2,0), action_sample, reward_sample, done_sample))
        batch_id += 1

    return minibatch

def preprocess(frame):

    gray_image = frame.mean(2)
    reshaped_image = imresize(gray_image, (84,84))
    x = np.reshape(reshaped_image, [84,84,1]).astype(np.float32)
    x *= (1.0 / 255.0) # divide by 255

    return x

def get_policy(D, obf):

    idx = len(D) - 7
    #range_idx = np.arange(idx, idx+7)

    state = [s[0] for s in D[idx : idx+7]]
    state.append(obf)
    #print("state shape input", state[0].shape)

    q_vals = net.q_val(np.asarray(state).transpose(3,1,2,0))

    one_hot = np.zeros((BATCH, ACTIONS))
    one_hot[:,np.argmax(q_vals, axis=1)] = 1
    AMN_policy = one_hot

    return AMN_policy

def rollout():

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())
    #saver.restore(sess, load_path)
    #print("variables restored and loaded...")

    env = gym.make('Pong-v0')
    ACTIONS = env.action_space.n

    # stores history for all games separately
    s_t = []
    s_t1 = []

    D = []
    k = 0
    num_episodes = 0
    train = False

    while num_episodes < MAX_EPISODES:

        ob = env.reset()

        obf = preprocess(ob)
        s_t = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (84, 84, 4))
        observations, actions = [], []
        REWARD = 0
        action_index = random.randrange(ACTIONS)

        i = 0
        print("num of episodes ", num_episodes)

        for t in range(10000):
            env.render() #optional

            ob, reward, done, info = env.step(action_index)

            REWARD += reward
            obf = preprocess(ob)
            #print("D lenght", len(D))

            if i == 3:

                if len(D) > 8:
                    q_val = get_policy(D, obf)
                    # epsilon greedy policy
                    if random.random() <= epsilon:
                        action_index = random.randrange(ACTIONS)
                        #action[action_index] = 1
                    else:
                        action_index = np.argmax(q_val)
                        #action[action_index] = 1

                else:
                    action_index = random.randrange(ACTIONS)

                #s_t1 = np.append(obf, s_t[:,:,0:3], axis = 2)
                D.append((obf, int(action_index), REWARD, done))
                if len(D) > REPLAY_MEMORY:
                    D.pop(0)

            if num_episodes > 2:
                train = True

            if train == True:

                print("training now...")
                minibatch = get_minibatch(D, BATCH)
                print("minibatch collected...")
                state_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]
                reward_batch = [d[2] for d in minibatch]
                done_batch = [d[3] for d in minibatch]
                print("minibatch state shape", np.asarray(state_batch).shape)
                # minibatch update
                net.train(np.asarray(state_batch).reshape(32,84,84,8), np.asarray(action_batch).reshape(32,4),
                                                       np.asarray(reward_batch).reshape(32,1), np.asarray(done_batch).reshape(32,1))

            if i == 3:
                #s_t = s_t1
                i = 0

            i += 1

            if done:
                num_episodes += 1
                break

rollout()
