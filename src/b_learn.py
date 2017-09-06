# 

import _config, _lib
import sys, os, fnmatch, datetime, subprocess, random
from autograd import grad
import autograd.numpy as np
from collections import defaultdict
from mylib import util
import edward as ed
import tensorflow as tf
from edward.models import Uniform, Normal
from tensorflow.contrib import slim
import custom_gan_inference

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'a_generate/'
NAME = util.get_fn(__file__)

# Functions
def igp_hat(zs):
  assert (0 < zs).all() and (zs < 1).all()
  true_means_0 = np.array([-1, 1, -1, 3, 4])
  true_means_1 = np.array([1, -1, 2, 3, 0])
  std = 0.15

  # differs in last column, is 2 instead of 0.
  alt_means_1 = np.array([1, -1, 2, 3, 2])

  xs = []
  for z in zs:
    mean = true_means_0 + z * alt_means_1
    draw = np.random.normal(mean, std, len(mean))
    xs.append(draw)
  return np.array(xs)

def generator_loss(z):
  x = igp_hat(z)[0]
  return np.log(discriminator(x))

def discriminator(x):
  if x[0] < 0:
    return 0.10
  if x[0] > 0:
    return 0.90

def generative_network(noise):
  h1 = slim.fully_connected(noise, 32, activation_fn = tf.nn.relu)
  x = slim.fully_connected(h1, 2, activation_fn = tf.sigmoid)
  return x

def discriminative_network(x):
  h1 = slim.fully_connected(x, 16, activation_fn = tf.nn.relu)
  h2 = slim.fully_connected(h1, 16, activation_fn = tf.nn.relu)
  h3 = slim.fully_connected(h2, 16, activation_fn = tf.nn.relu)
  logit = slim.fully_connected(h3, 1, activation_fn = None)
  return logit

def learn(inp_dir, out_dir):
  import code; code.interact(local=dict(globals(), **locals()))

  print 'Setting up GAN...'
  batch_size = 500
  d = 2
  x_ph = tf.placeholder(tf.float32, [batch_size, d])
  with tf.variable_scope('Gen'):
    # noise = Uniform(tf.zeros([batch_size, d]) - 1.0, tf.ones([batch_size, d]))
    noise = Normal(tf.zeros([batch_size, d]), tf.constant(1.0))
    x = generative_network(noise)

  print 'Setting up inference...'
  inference = custom_gan_inference.GANInference(data = {x: x_ph},
                              discriminator = discriminative_network)
  optimizer = tf.train.AdamOptimizer()
  optimizer_d = tf.train.AdamOptimizer()
  inference.initialize(optimizer = optimizer, 
                       optimizer_d = optimizer_d,
                       n_iter = 1200, 
                       n_print = 2,
                       igp_hat_fn = igp_hat)  

  return

@util.time_dec
def main(inp_dir, out_dir, run = True):
  print NAME  
  util.ensure_dir_exists(out_dir)
  if not run:
    print '\tskipped'
    return out_dir

  # Function calls
  learn(inp_dir, out_dir)

  return out_dir


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1], sys.argv[2])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
