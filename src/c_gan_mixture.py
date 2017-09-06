# Implements a Generative Adversarial Network, from
# arxiv.org/abs/1406.2661
# but, it always collapses to generating a single image.
# Let me know if you can get it to work! - David Duvenaud

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import multigrad, grad
from autograd.util import flatten
import matplotlib.pyplot as plt
import subprocess, os
from mylib import util

### Define geneerator, discriminator, and objective ###

def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)
def logsigmoid(x): return x - np.logaddexp(0, x)
def leaky_relu(x): return np.maximum(0, x) + np.minimum(0, x) * 0.001

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
  """Build a list of (weights, biases) tuples,
     one for each layer in the net."""
  return [(scale * rs.randn(m, n),   # weight matrix
           scale * rs.randn(n))      # bias vector
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
  mbmean = np.mean(activations, axis=0, keepdims=True)
  return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict_gen(params, inputs):
  """Params is a list of (weights, bias) tuples.
     inputs is an (N x D) matrix."""
  inpW, inpb = params[0]
  inputs = leaky_relu(np.dot(inputs, inpW) + inpb)
  for W, b in params[1:-1]:
    outputs = np.dot(inputs, W) + b
    outputs = batch_normalize(np.dot(inputs, W) + b)
    inputs = leaky_relu(outputs)
  outW, outb = params[-1]
  outputs = np.dot(inputs, outW) + outb
  return outputs

def neural_net_predict_dsc(inputs, params):
  """Params is a list of (weights, bias) tuples.
     inputs is an (N x D) matrix."""
  inpW, inpb = params[0]
  inputs = relu(np.dot(inputs, inpW) + inpb)
  for W, b in params[1:-1]:
    outputs = np.dot(inputs, W) + b
    # outputs = batch_normalize(np.dot(inputs, W) + b)
    inputs = leaky_relu(outputs)
    # inputs = relu(outputs)
  outW, outb = params[-1]
  outputs = np.dot(inputs, outW) + outb
  return outputs

def generate_from_noise(gen_params, num_samples, noise_dimZ, rs):
  noise = rs.randn(num_samples, noise_dimZ)
  samples = neural_net_predict_gen(gen_params, noise)
  return sigmoid(samples)

def igp_hat(zs, noiseX):
  # noiseX is unit gaussian

  # true_means_0 = np.array([-1, 1, -1, 3, 4])
  # true_means_1 = np.array([1, -1, 2, 3, 0])

  true_means_0 = np.array([-50])
  true_means_1 = np.array([50])
  
  std = 1.00
  # differs in last column, is 2 instead of 0.
  alt_means_1 = np.array([1, -1, 2, 3, 2])
  # xs = true_means_0 + zs * alt_means_1
  xs = (1-zs) * true_means_0 + zs * true_means_1
  # xs = np.minimum(6, xs)
  # xs = np.maximum(-6, xs)
  xs = xs + std*noiseX
  # xs = xs + np.random.normal(0, std, xs.shape)
  return xs 

def wgan_objective(gen_params, dsc_params, real_data, num_samples, noise_dimZ, noise_dimX, wasserstein_lambda, rs):
  fake_z = generate_from_noise(gen_params, num_samples, noise_dimZ, rs)
  noiseX = rs.randn(num_samples, noise_dimX)
  fake_data = igp_hat(fake_z, noiseX)
  # fake_data = fake_z
  assert fake_data.shape == real_data.shape

  # Discrimination loss on real/fake data
  logprobs_fake = logsigmoid(neural_net_predict_dsc(fake_data, dsc_params))
  logprobs_real = logsigmoid(neural_net_predict_dsc(real_data, dsc_params))
  
  # Wasserstein loss
  gtest = grad(neural_net_predict_dsc)
  wasserstein_loss = 0
  for i in range(num_samples):
    x_hat = ( real_data[i] + fake_data[i] ) / 2
    gxh = gtest(x_hat, dsc_params)
    norm = np.sqrt(sum(gxh ** 2))
    wasserstein_loss += wasserstein_lambda * (norm - 1)**2
  wasserstein_loss /= num_samples

  # print(np.mean(logprobs_real) - np.mean(logprobs_fake))
  # print(wasserstein_loss)

  return np.mean(logprobs_real) - np.mean(logprobs_fake) + wasserstein_loss

def entropy_objective(gen_params, batch_size, noise_dimZ, rs, neighbors_function):
  # try to maximize entropy
  fake_z = generate_from_noise(gen_params, batch_size, noise_dimZ, rs)

  neighbors = neighbors_function(fake_z)

  entropy_loss = 0
  for i in range(len(fake_z)):
    sq_dist = 0
    if i == neighbors[i][0] or i == neighbors[i][1]:
      continue
    left_dist = fake_z[i] - fake_z[neighbors[i][0]]
    right_dist = fake_z[neighbors[i][1]] - fake_z[i]
    assert left_dist >= 0 and right_dist >= 0, 'not sorted'
    left_dist = max(1e-5, left_dist)
    right_dist = max(1e-5, right_dist)
    sq_dist += 0.5 * np.log(left_dist) + 0.5 * np.log(right_dist)
    entropy_loss += -sq_dist
  return entropy_loss / len(fake_z - 1)

def alphabetize(num):
  assert num < 26**3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
  hundreds = int(num / (26*26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def save_images(fake_z, fake_data, real_data, out_dir, nm, dsc_params, vmin=0, vmax=1):
  # plot Z
  assert not np.isnan(fake_z).any(), 'NaN in fake_z'
  rmax = int( max(1, max(fake_z)) )
  rmin = int( min(0, min(fake_z)))
  binsize = (int(rmax) - int(rmin)) / 100
  plt.hist(fake_z, bins = np.arange(rmin, rmax, binsize))
  plt.ylabel('Histogram counts', color = 'b')
  plt.xlim([rmin, rmax])
  plt.title('Generated Z')
  plt.savefig(out_dir + 'gan_samples_Z_' + nm + '.png')
  plt.close()


  rmax = max(max(real_data), max(fake_data))
  rmin = min(min(real_data), min(fake_data))
  binsize = (rmax - rmin) / 100
  # plot X
  query_pts = np.arange(rmin, rmax, binsize)
  query_pts = query_pts.reshape(len(query_pts), 1)
  curr_dsc = sigmoid(neural_net_predict_dsc(query_pts, dsc_params))
  fig, ax1 = plt.subplots()
  ax1.hist(fake_data, bins = np.arange(rmin, rmax, binsize),
           color = 'b',
           alpha = 0.5)
  ax1.hist(real_data, bins = np.arange(rmin, rmax, binsize), 
           color = 'g', 
           alpha = 0.5)
  ax1.set_ylabel('Histogram counts', color = 'b')
  ax1.tick_params('y', colors = 'b')
  ax1.set_xlim(rmin, rmax)
  ax2 = ax1.twinx()
  ax2.plot(np.arange(rmin, rmax, binsize), curr_dsc, 'r')
  ax2.set_ylabel('Discriminator label | 0=real ; 1=fake', 
                  color = 'r')
  ax2.set_ylim(0, 1)
  ax2.tick_params('y', colors = 'r')
  fig.tight_layout()
  plt.title('Generated X')
  plt.savefig(out_dir + 'gan_samples_X_' + nm + '.png')
  plt.close()
  return

### Define minimax version of adam optimizer ###

def adam_minimax(grad_both, init_params_max, init_params_min, neighbors_function, callback=None, num_iters=100,
         step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
  """Adam modified to do minimiax optimization, for instance to help with
  training generative adversarial networks."""

  def exponential_decay(step_size_min, step_size_max):
    if step_size_min > 0.0001:
      step_size_min *= 0.99
    if step_size_max > 0.01:
      step_size_max *= 0.99
    return step_size_min, step_size_max

  x_max, unflatten_max = flatten(init_params_max)
  x_min, unflatten_min = flatten(init_params_min)

  m_max = np.zeros(len(x_max))
  v_max = np.zeros(len(x_max))
  m_min = np.zeros(len(x_min))
  v_min = np.zeros(len(x_min))

  K = 5

  for i in range(num_iters):
    g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                   unflatten_min(x_min), i, neighbors_function)
    g_max, _ = flatten(g_max_uf)
    g_min, _ = flatten(g_min_uf)

    if callback: 
      callback(unflatten_max(x_max), 
              unflatten_min(x_min), 
              i, 
              unflatten_max(g_max), 
              unflatten_min(g_min))

    step_size_min, step_size_max = exponential_decay(step_size_min, step_size_max)

    # Update generator (maximizer)
    m_max = (1 - b1) * g_max      + b1 * m_max  # First  moment estimate.
    v_max = (1 - b2) * (g_max**2) + b2 * v_max  # Second moment estimate.
    mhat_max = m_max / (1 - b1**(i + 1))    # Bias correction.
    vhat_max = v_max / (1 - b2**(i + 1))
    x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)

    # Update discriminator (minimizer)
    m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
    v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
    mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
    vhat_min = v_min / (1 - b2**(i + 1))
    x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)

    for k in range(K-1):
      g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                 unflatten_min(x_min), i, neighbors_function)
      g_min, _ = flatten(g_min_uf)

      # Update discriminator (minimizer)
      m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
      v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
      mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
      vhat_min = v_min / (1 - b2**(i + 1))
      x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)

  return unflatten_max(x_max), unflatten_min(x_min)


def create_gif(out_dir):
  print('Creating GIF...')
  subprocess.call('convert -delay 15 -loop 0 ' + out_dir + '*_X_*.png ' + out_dir + '_anim_X.gif', shell = True)
  subprocess.call('convert -delay 15 -loop 0 ' + out_dir + '*_Z_*.png ' + out_dir + '_anim_Z.gif', shell = True)
  print('Done.')
  return

def count_num_folders(out_dir):
  for fold in os.listdir(out_dir):
    assert os.path.isdir(out_dir + fold), 'Not a folder!'
  return len(os.listdir(out_dir))
  
def copy_script(out_dir):
  src_dir = '/cluster/mshen/prj/gans/src/'
  script_nm = __file__
  subprocess.call('cp ' + src_dir + script_nm + ' ' + out_dir, shell = True)
  return

def save_gen_params(gen_params, gp_out_dir, nm):
  np.savetxt(gp_out_dir + nm + '_00.csv', gen_params[0][0], delimiter = ',')
  np.savetxt(gp_out_dir + nm + '_01.csv', gen_params[0][1], delimiter = ',')
  np.savetxt(gp_out_dir + nm + '_10.csv', gen_params[1][0], delimiter = ',')
  np.savetxt(gp_out_dir + nm + '_11.csv', gen_params[1][1], delimiter = ',')
  return

### Setup and run ###

if __name__ == '__main__':
  out_place = '/cluster/mshen/prj/gans/out/2017-06-19/c_gan/'
  num_folds = count_num_folders(out_place)
  out_dir = out_place + alphabetize(num_folds + 1) + '/'
  util.ensure_dir_exists(out_dir)
  gen_params_out_dir = out_dir + 'gen_params/'
  util.ensure_dir_exists(gen_params_out_dir)
  print('outdir: ' + alphabetize(num_folds + 1))

  copy_script(out_dir)
  counter = 0

  # Model hyper-parameters
  noise_dimZ = 1
  noise_dimX = 1
  gen_layer_sizes = [noise_dimZ, 16, 16, 1]
  dsc_layer_sizes = [1, 16, 16, 1]
  wasserstein_lambda = 10
  
  # Training parameters
  gen_param_scale = 0.1         # generate diverse samples
  dsc_param_scale = 0.1      # ensure 50/50 prior
  # batch_size = 10000
  batch_size = 200
  num_epochs = 500
  step_size_max = 0.10
  # step_size_max = 0.005
  step_size_min = 0.10


  print("Loading training data...")
  train_data = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/mixture/X.csv', delimiter = ',')
  train_data = train_data.reshape(len(train_data), 1)

  # train_data = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/X.csv', delimiter = ',')
  # train_data = train_data.reshape(len(train_data), 1)

  init_gen_params = init_random_params(gen_param_scale, gen_layer_sizes)
  init_dsc_params = init_random_params(dsc_param_scale, dsc_layer_sizes)

  num_batches = int(np.ceil(len(train_data) / batch_size))
  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

  def neighbors_function(data):
    data = list(data)
    sorteddata = sorted(data)   # fast: 0.1 seconds or less
    ns = []
    # print('Constructing neighbors...')
    # timer = util.Timer(total = len(data))
    for i, d in enumerate(data):
      if sorteddata.index(d) > 0:
        n1 = data.index( sorteddata[ sorteddata.index(d) - 1 ] )
      else:
        n1 = data.index( d )
      try:
        n2 = data.index( sorteddata[ sorteddata.index(d) + 1 ] )
      except IndexError:
        n2 = data.index(d)
      ns.append( np.array([n1, n2]) )
      # timer.update()
    # print('Done with neighbors')
    return np.array(ns)        

  # Define training objective
  seed = npr.RandomState(1)
  def objective(gen_params, dsc_params, iter, neighbors_function):
    idx = batch_indices(iter)
    c1, c2 = c1c2_schedule(iter)
    return c1 * wgan_objective(gen_params, dsc_params, 
                              train_data[idx], 
                              batch_size, 
                              noise_dimZ,
                              noise_dimX, 
                              wasserstein_lambda,
                              seed) - \
           c2 * entropy_objective(gen_params, 
                                  batch_size, 
                                  noise_dimZ, 
                                  seed, 
                                  neighbors_function)

  def c1c2_schedule(iter):
    if iter < 50:
      return 0, 1
    else:
      return 1, 0.2
    return c1, c2

  # Get gradients of objective using autograd.
  both_objective_grad = multigrad(objective, argnums=[0,1])

  print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
  def print_perf(gen_params, dsc_params, iter, gen_gradient, dsc_gradient):
    if iter % 10 == 0:
      ability = np.mean(objective(gen_params, dsc_params, iter, neighbors_function))
      
      fake_z = generate_from_noise(gen_params, 10000, noise_dimZ, seed)
      noiseX = seed.randn(10000, noise_dimX)
      fake_data = igp_hat(fake_z, noiseX)
      
      # fake_data = fake_z
      
      real_data = train_data
      probs_fake = np.mean(sigmoid(neural_net_predict_dsc(fake_data, dsc_params)))
      probs_real = np.mean(sigmoid(neural_net_predict_dsc(real_data, dsc_params)))
      print("{:15}|{:20}|{:20}|{:20}".format(iter//num_batches, ability, probs_fake, probs_real))
      save_images(fake_z, fake_data, real_data, out_dir, alphabetize(int(iter/10)), dsc_params, vmin=0, vmax=1)
      save_gen_params(gen_params, gen_params_out_dir, alphabetize(int(iter/10)) )
      return ability
    return None

  # The optimizers provided can optimize lists, tuples, or dicts of parameters.
  optimized_params = adam_minimax(both_objective_grad,
                                  init_gen_params, init_dsc_params, neighbors_function, 
                                  step_size_max=step_size_max, step_size_min=step_size_min,
                                  num_iters=num_epochs * num_batches, callback=print_perf)

  print('Done')
  create_gif(out_dir)
  # import code; code.interact(local=dict(globals(), **locals()))