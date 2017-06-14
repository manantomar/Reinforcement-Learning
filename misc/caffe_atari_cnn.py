def add_cnn(n, data, act, batch_size, T, K, num_step, mode='train'):
  # data : batch_size x T x 3 x height x width
  n.x_flat = L.Flatten(data, axis=1, end_axis=2)
  # n.x_flat : batch_size x T*3 x height x width
  n.act_flat = L.Flatten(act, axis=1, end_axis=2)
  if mode == 'train':
    x = L.Slice(n.x_flat, axis=1, ntop=T)
    # x : T layers of size : batch_size x 3 x height x width
    act_slice = L.Slice(n.act_flat, axis=1, ntop=T-1)
    x_set = ()
    label_set = ()
    x_hat_set = ()
    silence_set = ()
    for i in range(T):
      t = tag(i+1)
      # n.tops[x1] : batch_size x 3 x height x width
      # n.tops[x2] : batch_size x 3 x height x width
      n.tops['x'+t] = x[i]
      if i < K:
        # storing just the first four frames in x_set
        x_set += (x[i],)
      if i < T - 1:
        n.tops['act'+t] = act_slice[i]
      if i < K - 1:
        silence_set += (n.tops['act'+t],)
      if i >= K:
        # storing the fifth frame as the label
        label_set += (x[i],)
    # not important for 1 step prediction,
    # produces : batch_size x 3 x height x width
    n.label = L.Concat(*label_set, axis=0)
    # converting to list
    input_list = list(x_set)
    # not important as no. of steps is 1
    for step in range(0, num_step):
      step_tag = tag(step + 1) if step > 0 else ''
      t = tag(step + K)
      tp = tag(step + K + 1)
      input_tuple = tuple(input_list)
      # concatenating all 4 frames together
      n.tops['input'+step_tag] = L.Concat(*input_tuple, axis=1)
      # passing through the feed-forward net
      top = add_conv_enc(n, n.tops['input'+step_tag], tag=step_tag)
      n.tops['x_hat'+tp] = add_decoder(n, top, n.tops['act'+t], flatten=False,
          tag=step_tag)
      # using the predicted values to form the input for the next prediction
      input_list.pop(0)
      input_list.append(n.tops['x_hat'+tp])
  else:
    top = add_conv_enc(n, n.x_flat)
    n.tops['x_hat'+tag(K+1)] = add_decoder(n, top, n.act_flat, flatten=False)
  if mode == 'train':
    x_hat = ()
    # for 1 step prediciton, just runs once for i = 4
    for i in range(K, T):
      t = tag(i+1)
      # prediction for the 5th frame comes from the net
      x_hat += (n.tops['x_hat'+t],)
    # concatenate all predictions
    n.x_hat = L.Concat(*x_hat, axis=0)
    n.silence = L.Silence(*silence_set, ntop=0)
    # takes the predcition for the 5th frame and output label
    # both are of size batch_size x 3 x height x width
    n.l2_loss = L.EuclideanLoss(n.x_hat, n.label)
  return n
