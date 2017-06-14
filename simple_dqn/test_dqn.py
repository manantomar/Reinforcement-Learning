import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imresize
from collections import deque
import sys
import random

INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.05
REPLAY_MEMORY = 10000
max_episodes = 5
BATCH = 2
GAMMA = 0.99
TRAIN = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def dqn(x):

    #x = tf.reshape(x, (-1, 84, 84, 2))
    x = tf.reshape(x, (-1, 4, 4, 2))
    #print(x.shape)
    conv1 = tf.layers.conv2d(x, 32, [5, 5], padding="same", activation=tf.nn.relu)
    print(conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(pool1, 64, [5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print("passes pool2", pool2.shape)
    #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64 * 9])
    pool2_flat = tf.reshape(pool2, [-1, 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    #print("passes dense", dense.shape)
    action_logits = tf.layers.dense(inputs=dense, units=ACTIONS)
    print("passes logits", action_logits.shape)
    #conv1 = slim.conv2d(x, 10, [5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(conv1, 10, [5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
    #net = slim.flatten(conv2)
    #action_logits = slim.fully_connected(net, ACTIONS, activation_fn=None)

    return action_logits

def preprocess(frame):
    gray_image = frame.mean(2)
    #reshaped_image = imresize(gray_image, (84,84))
    reshaped_image = imresize(gray_image, (4,4))
    #x = np.reshape(reshaped_image, [84,84,1])
    x = np.reshape(reshaped_image, [4,4,1])
    return x

def rollout(sess, max_iter=5000):

    observations, actions, rewards = [], [], []
    ob = env.reset()
    ep_steps = 0
    num_episodes = 0
    epsilon = INITIAL_EPSILON

    ob_now = preprocess(ob)
    ob_prev = None
    t=0

    D = deque()
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    #s = tf.placeholder("float", [None, 84, 84, 2])
    s = tf.placeholder("float", [None, 4, 4, 2])

    readout_action = tf.reduce_sum(tf.multiply(dqn(s), a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    sess.run(tf.initialize_all_variables())

    while True or num_episodes < max_episodes:

        ob_before = ob_now if ob_prev is None else ob_prev
        obf_prev = np.concatenate((ob_before, ob_now), 2)
        ob_prev = ob_now

        action_index = 0
        action = np.zeros(ACTIONS, np.int32)
        print("action is ", action)
        if random.random() <= epsilon:
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        else:
            action_index = np.argmax(dqn(obf_prev.astype(np.float32)))
            action[action_index] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 1000

        ob, reward, done, info = env.step(np.argmax(action))
        ep_steps += 1

        ob_now = preprocess(ob)
        ob_before = ob_now if ob_prev is None else ob_prev
        obf_now = np.concatenate((ob_before, ob_now), 2)

        #observations.append(ob)
        #actions.append(action)
        #rewards.append(reward)

        D.append((obf_prev, action, reward, obf_now, done))
        if len(D) > REPLAY_MEMORY:
                D.popleft()
        if t >= TRAIN:
            #training starts
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            obf_prev_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            obf_now_batch = [d[3] for d in minibatch]

            target_batch = []
            #obf_batch = np.concatenate(ob_before, ob_now), 2)
            for i in range(0, len(minibatch)):

                if minibatch[i][4]:
                     target_batch.append(reward_batch[i])

                else:
                    print("obf_prev_batch shape ", len(obf_prev_batch))
                    target_batch.append(reward_batch[i] + GAMMA*sess.run(tf.reduce_max(dqn(obf_now_batch[i].astype(np.float32)))))

            #print("obf_prev_batch", obf_prev_batch)
            obff = np.zeros((len(obf_prev_batch), 4, 4, 2))
            for i,x in enumerate(obf_prev_batch):
                obff[i] = x
            #readout_t = s.eval(feed_dict = {s : obff})[0]
            print("reward", reward_batch)
            print("target_batch", target_batch[0])
            target = np.zeros((len(target_batch)))
            for i,x in enumerate(target_batch):
                target[i] = x

            #print("reward", reward_batch[1])
            train_step.run(feed_dict = {
                s : obff,
                a : action_batch,
                y : target})


        ob_prev = ob_now
        t += 1

        if done or ep_steps >= max_iter:
            num_episodes += 1
            ep_steps = 0
            ob_prev = None
            ob = env.reset()

env = gym.make('Pong-v0')

#x = tf.placeholder(tf.float32, name )
#sampled_actions = tf.placeholder(tf.int32)
#discounted_reward = tf.placeholder(tf.float32)

#action_logits = dqn(x)
ACTIONS = env.action_space.n
sess = tf.InteractiveSession()
#s, action_logits = dqn()
rollout(sess)
#print(env.action_space.n)
for i_episode in range(1):
    observation = env.reset()
    ob = preprocess(observation)
    obf = []
    print(ob.shape)
    for t in range(10000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        obf.append(preprocess(observation))
        #print(action)
        if done == True:
            print("Episode finished")
            break

#print(obf[0])
#print(len(obf))
#obff = np.zeros((len(obf), 84, 84, 1))
#for i,x in enumerate(obf):
#    obff[i] = x
#print(obff)
