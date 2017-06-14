import os
os.environ['THEANO_FLAGS'] = "device=gpu0"

import theano
import numpy
import pickle

pkl = open('./pong_dqn_v4_reg_0.01/network_file_50.pkl', 'rb')
data = pickle.load(pkl)

import lasagne

params = lasagne.layers.get_all_params(data.l_out)
param_values = {}

id = 0

for p in params:

    if str(p) == "W":
        param_values["w%d" %(id+1)] = p.get_value().T

    if str(p) == "b":
        param_values["w%d" %(id+1)] = p.get_value()
        id += 1

pkl.close()
pkl = open('./pong_dqn_v4_reg_0.01/network_params.pkl', 'wb')
pickle.dump(param_values, pkl)

pkl.close()
