# 

import _config, _lib
import sys, os, fnmatch, datetime, subprocess, random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mylib import util


# Default params
DEFAULT_INP_DIR = _config.DATA_DIR
NAME = util.get_fn(__file__)

# Functions

def gen_trimodal(num):
  mid = np.random.normal(loc = 0.5, scale = 0.10, size = num/2)
  low = np.random.exponential(scale = 0.10, size = num/4)
  high = 1 - np.random.exponential(scale = 0.10, size = num/4)
  leftover = np.random.uniform(size = num%4)          
  z = np.append(mid, low)
  z = np.append(z, high)
  z = np.append(z, leftover)  
  return z

def gen_linear(num):
  low = np.random.uniform(low = 0, high = 0.40, size = int(num * 0.45))
  mid = np.random.uniform(low = 0.40, high = 0.65, size = int(num * 0.25))
  high = np.random.uniform(low = 0.65, high = 1.00, size = int(num * 0.3))
  leftover = np.random.uniform(size = num % 20)
  for item in mid:
    if item < 0.53:
      prob = (0.53-item)/0.26
      if np.random.rand() < prob:
        item = (0.53-item) + 0.52
  z = np.append(mid, low)
  z = np.append(z, high)
  z = np.append(z, leftover)
  return z

def gen_gaussian(num):
  z = np.random.normal(loc = 0.5, scale = 0.15, size = num)
  z = np.minimum(1, z)
  z = np.maximum(0, z)
  return z

def gen_mixture_gaussian_uniform(num):
  gp = np.random.normal(loc = 0.30, scale = 0.10, size = num/2)
  gp = np.minimum(1, gp)
  gp = np.maximum(0, gp)

  un = np.random.uniform(low = 0.45, high = 1.00, size = num/2)

  z = np.hstack([gp, un])
  return z

def generate_z(num):

  # z = gen_gaussian(num)
  # z = gen_trimodal(num)
  # z = gen_linear(num)
  z = gen_mixture_gaussian_uniform(num)


  assert len(z) == num
  return z

def plot_z(zs, out_fn):
  plt.hist(zs, bins = np.arange(0, 1, 0.01))
  plt.title('z histogram | genotype')
  plt.xlim([0, 1])
  plt.xlabel('Z')
  plt.savefig(out_fn)
  plt.close()  
  return

def igp_true(zs):
  # also p(x|z)
  # means_0 = np.array([-1, 1, -1, 3, 4])
  # means_1 = np.array([1, -1, 2, 3, 0])
  means_0 = np.array([-50])
  means_1 = np.array([50])
  std = 1.00
  xs = []
  for z in zs:
    mean = (1-z) * means_0 + z * means_1
    # if mean > 0:
      # mean = mean ** 2
    draw = np.random.normal(mean, std, len(mean))
    xs.append(draw)
  return np.array(xs)

def igp_true_2dx(zs):
  # also p(x|z)
  # means_0 = np.array([-1, 1, -1, 3, 4])
  # means_1 = np.array([1, -1, 2, 3, 0])
  means_0 = np.array([-50, -50])
  means_1 = np.array([50, 50])
  std = 1.00
  error = np.random.normal(loc = 0, scale = std, size = 1)
  # corr = 0
  # cov = np.array([[std, corr*std*std], [corr*std*std, std]])
  xs = []
  for z in zs:
    mean = (1-z) * means_0 + z * means_1
    # if mean > 0:
      # mean = mean ** 2
    # draw = np.random.normal(mean, std, len(mean))
    # draw = np.random.multivariate_normal(mean, cov)
    draw = np.array([mean[0] + error, mean[1] - error])
    xs.append(draw)
  return np.array(xs)


def generate(out_dir):
  num = 10000

  zs = generate_z(num)
  plot_z(zs, out_dir + 'z_hist.pdf')  

  # xs = igp_true(zs)
  xs = igp_true_2dx(zs)

  np.savetxt(out_dir + 'Z.csv', zs, delimiter = ',')
  np.savetxt(out_dir + 'X.csv', xs, delimiter = ',')

  return

@util.time_dec
def main(inp_dir, out_dir, run = True):
  print NAME  
  util.ensure_dir_exists(out_dir)
  if not run:
    print '\tskipped'
    return out_dir

  # Function calls
  generate(out_dir)

  return out_dir


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1], sys.argv[2])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')