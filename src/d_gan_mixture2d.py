# Implements a Generative Adversarial Network, from
# arxiv.org/abs/1406.2661
# but, it always collapses to generating a single image.
# Let me know if you can get it to work! - David Duvenaud

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import multigrad
from autograd.util import flatten
import matplotlib.pyplot as plt
import subprocess, os
from mylib import util
from sklearn.metrics import r2_score

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

def neural_net_predict_genZ(params, inputs):
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

def generate_Z_from_noise(genZ_params, num_samples, noise_dimZ, rs):
  noiseZ = rs.randn(num_samples, noise_dimZ)
  samples = neural_net_predict_genZ(genZ_params, noiseZ)
  return sigmoid(samples)

def neural_net_predict_genX(params, inputs):
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

def neural_net_predict_dsc(params, inputs):
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

def generate_X_from_noise_and_real_z(genX_params, num_samples, noise_dimX, real_z, rs):
  noiseX = rs.randn(num_samples, noise_dimX)
  noise_and_z = np.hstack([real_z, noiseX])
  assert noise_and_z.shape[0] == num_samples
  samples = neural_net_predict_genX(genX_params, noise_and_z)
  return samples

def generate_X_from_noise_and_genz(genX_params, genZ_params, num_samples, noise_dimX, noise_dimZ, rs):
  genz = generate_Z_from_noise(genZ_params, num_samples, noise_dimZ, rs)
  noiseX = rs.randn(num_samples, noise_dimX)
  noise_and_z = np.hstack([genz, noiseX])
  assert noise_and_z.shape[0] == num_samples
  samples = neural_net_predict_genX(genX_params, noise_and_z)
  return samples

def generate_X_from_given_noise_and_given_z(genX_params, num_samples, noiseX, fake_Z, rs):
  noise_and_z = np.hstack([fake_Z, noiseX])
  assert noise_and_z.shape[0] == num_samples
  samples = neural_net_predict_genX(genX_params, noise_and_z)
  return samples

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
  xs = xs + std*noiseX
  return xs 

def gan_objective(genX_params, genZ_params, dsc_params, real_data, real_z, num_samples, noise_dimX, noise_dimZ, rs):
  fake_data = generate_X_from_noise_and_genz(genX_params, genZ_params, num_samples, noise_dimX, noise_dimZ, rs)
  assert fake_data.shape == real_data.shape
  logprobs_fake = logsigmoid(neural_net_predict_dsc(dsc_params, fake_data))
  logprobs_real = logsigmoid(neural_net_predict_dsc(dsc_params, real_data))
  return np.mean(logprobs_real) - np.mean(logprobs_fake)

def igp_objective(genX_params, genZ_params, real_z, num_samples, noise_dimX, noise_dimZ, rs):
  fake_Z = generate_Z_from_noise(genZ_params, num_samples, noise_dimZ, rs)
  noiseX = rs.randn(num_samples, noise_dimX)

  # GAN2: deterministic given noise
  fake_data = generate_X_from_given_noise_and_given_z(genX_params, num_samples, noiseX, fake_Z, rs)
  
  # IGP-hat: deterministic given noise
  igp_data = igp_hat(fake_Z, noiseX)
  
  return np.sum((fake_data - igp_data)**2) / num_samples

def alphabetize(num):
  assert num < 26**3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
  hundreds = int(num / (26*26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def save_images(fake_data, igp_data, real_data, out_dir, nm, dsc_params):
  # plot igp objective
  binsize = (max(igp_data)-min(igp_data))/100
  plt.hist(fake_data, 
           bins = np.arange(min(igp_data), max(igp_data), binsize ), 
           color = 'b', alpha = 0.5)
  plt.hist(igp_data, 
           bins = np.arange(min(igp_data), max(igp_data), binsize ), 
           color = 'g', alpha = 0.5)
  plt.ylabel('Histogram counts')
  plt.title('Generated X (blue) vs. IGP (green)')
  plt.savefig(out_dir + 'gan_samples_X_IGP_' + nm + '.png')
  plt.close()

  # plot real_data/disc
  binsize = (max(real_data)-min(real_data))/100
  curr_dsc = sigmoid(neural_net_predict_dsc(dsc_params, np.arange(min(real_data), max(real_data), binsize).reshape(100, 1)))
  fig, ax1 = plt.subplots()
  ax1.hist(fake_data, bins = np.arange(min(real_data), max(real_data), binsize),
           color = 'b',
           alpha = 0.5)
  ax1.hist(real_data, bins = np.arange(min(real_data), max(real_data), binsize), 
           color = 'g', 
           alpha = 0.5)
  ax1.set_ylabel('Histogram counts', color = 'b')
  ax1.tick_params('y', colors = 'b')
  ax1.set_xlim(min(real_data), max(real_data))
  ax2 = ax1.twinx()
  ax2.plot(np.arange(min(real_data), max(real_data), binsize), curr_dsc, 'r')
  ax2.set_ylabel('Discriminator label | 0=real ; 1=fake', 
                  color = 'r')
  ax2.set_ylim(0, 1)
  ax2.tick_params('y', colors = 'r')
  fig.tight_layout()
  plt.title('Generated X vs. real data & discriminator')
  plt.savefig(out_dir + 'gan_samples_X_REAL_' + nm + '.png')
  plt.close()
  return

def measure_meaningful_z(fake_data, igp_data, out_dir, nm):
  out_fn = out_dir + nm + '.png'
  plt.plot(igp_data, fake_data, '.', alpha = 0.3, zorder = 0)
  plt.title('Meaning of Z-space: Comparing GAN to IGP-hat output across Z-space')
  r2s = r2_score(igp_data, fake_data)
  plt.xlabel('IGP-hat output | Rsq = ' + str(r2s))
  plt.ylabel('Generated data')

  _min = min(plt.ylim()[0], plt.xlim()[0])
  _max = max(plt.ylim()[1], plt.xlim()[1])
  plt.plot([_min, _max], [_min, _max], 'k-', zorder = 10)

  plt.savefig(out_fn)
  plt.close()
  return

### Define minimax version of adam optimizer ###

def adam_minimax(grad_both, init_params_max, init_params_min, callback=None, num_iters=100,
         step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
  """Adam modified to do minimiax optimization, for instance to help with
  training generative adversarial networks."""

  def exponential_decay(step_size_min, step_size_max):
    if step_size_min > 0.0001:
        step_size_min *= 0.99
    if step_size_max > 0.0001:
        step_size_max *= 0.99
    return step_size_min, step_size_max

  x_max, unflatten_max = flatten(init_params_max)
  x_min, unflatten_min = flatten(init_params_min)

  m_max = np.zeros(len(x_max))
  v_max = np.zeros(len(x_max))
  m_min = np.zeros(len(x_min))
  v_min = np.zeros(len(x_min))
  ability = 0
  HANDICAP = float('inf')
  for i in range(num_iters):
    g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                   unflatten_min(x_min), i)
    g_max, _ = flatten(g_max_uf)
    g_min, _ = flatten(g_min_uf)

    if callback: 
      callback(unflatten_max(x_max), 
              unflatten_min(x_min), 
              i, 
              unflatten_max(g_max), 
              unflatten_min(g_min))
      if i % 10 == 0:
        ability = objective(unflatten_max(x_max), 
                            unflatten_min(x_min),
                            i)
    
    step_size_min, step_size_max = exponential_decay(step_size_min, step_size_max)

    if ability < HANDICAP:
      m_max = (1 - b1) * g_max      + b1 * m_max  # First  moment estimate.
      v_max = (1 - b2) * (g_max**2) + b2 * v_max  # Second moment estimate.
      mhat_max = m_max / (1 - b1**(i + 1))    # Bias correction.
      vhat_max = v_max / (1 - b2**(i + 1))
      x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)
    else:
      print('Skipping generator update because objective is too high')

    if ability > - HANDICAP:
      m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
      v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
      mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
      vhat_min = v_min / (1 - b2**(i + 1))
      x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)
    else:
      print('Skipping discriminator update because objective is too low')
  return unflatten_max(x_max), unflatten_min(x_min)


def create_gif(out_dir, meaning_z_dir):
  print('Creating GIF...')
  subprocess.call('convert -delay 15 -loop 0 ' + out_dir + '*_X_IGP*.png ' + out_dir + '_anim_X_IGP.gif', shell = True)
  subprocess.call('convert -delay 15 -loop 0 ' + out_dir + '*_X_REAL*.png ' + out_dir + '_anim_X_REAL.gif', shell = True)
  subprocess.call('convert -delay 15 -loop 0 ' + meaning_z_dir + '*.png ' + meaning_z_dir + '_anim_meaning.gif', shell = True)
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
  
def import_ganZ_gen_params(gp_fold, iter_nm):
  n00 = np.loadtxt(gp_fold + iter_nm + '_00.csv', delimiter = ',')
  n00 = n00.reshape(1, n00.shape[0])
  n01 = np.loadtxt(gp_fold + iter_nm + '_01.csv', delimiter = ',')
  n10 = np.loadtxt(gp_fold + iter_nm + '_10.csv', delimiter = ',')
  n10 = n10.reshape(n10.shape[0], 1)
  n11 = np.loadtxt(gp_fold + iter_nm + '_11.csv', delimiter = ',')
  n11 = n11.reshape(1)
  genZ_params = [(n00, n01), (n10, n11)]
  return genZ_params

### Setup and run ###

if __name__ == '__main__':
  out_place = '/cluster/mshen/prj/gans/out/2017-06-19/d_gan/'
  num_folds = count_num_folders(out_place)
  out_dir = out_place + alphabetize(num_folds + 1) + '/'
  util.ensure_dir_exists(out_dir)
  meaning_z_dir = out_dir + '_meaningful_z/'
  util.ensure_dir_exists(meaning_z_dir)
  print('outdir: ' + alphabetize(num_folds + 1))

  copy_script(out_dir)
  counter = 0

  # Import GAN1 parameters to generate Z
  run_nm = 'afr'
  iter_nm = 'drj'
  gp_fold = '/cluster/mshen/prj/gans/out/2017-06-19/c_gan/' + run_nm + '/gen_params/'
  genZ_params = import_ganZ_gen_params(gp_fold, iter_nm)

  # Model hyper-parameters
  noise_dimZ = 1    # pull from c_gan.py = GAN1.
  noise_dimX = 1
  z_dim = 1
  genX_layer_sizes = [z_dim + noise_dimX, 16, 2]
  # dsc_layer_sizes = [1, 16, 16, 16, 1]
  dsc_layer_sizes = [2, 16, 16, 1]

  # Training parameters
  genX_param_scale = 0.1
  dsc_param_scale = 0.1
  batch_size = 200
  num_epochs = 100
  step_size_max = 0.10
  step_size_min = 0.001


  print("Loading training data...")
  true_data = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/mixture_2dx/X.csv', delimiter = ',')
  true_data = true_data.reshape(len(true_data), 1)

  true_z = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/mixture_2dx/Z.csv', delimiter = ',')
  true_z = true_z.reshape(len(true_z), 1)
  assert true_z.shape[1] == z_dim, 'z_dim does not match data dim'

  init_genX_params = init_random_params(genX_param_scale, genX_layer_sizes)
  init_dsc_params = init_random_params(dsc_param_scale, dsc_layer_sizes)

  num_batches = int(np.ceil(len(true_data) / batch_size))
  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

  # Define training objective
  seed = npr.RandomState(1)
  def objective(genX_params, dsc_params, iter):
    idx = batch_indices(iter)
    c1, c2 = c1c2_schedule(iter)
    return c1 * gan_objective(genX_params, genZ_params, dsc_params, 
                         true_data[idx], true_z[idx],
                         batch_size, noise_dimX, noise_dimZ, seed) \
           - c2 * igp_objective(genX_params, genZ_params,
                         true_z[idx], 
                         batch_size, noise_dimX, noise_dimZ, seed)

  def c1c2_schedule(iter):
    if iter < 100:
      return 1, 1
    else:
      return 1, max(1 - (iter-100)/500, 0.01)
    return c1, c2

  # Get gradients of objective using autograd.
  both_objective_grad = multigrad(objective, argnums=[0,1])

  print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
  def print_perf(genX_params, dsc_params, iter, gen_gradient, dsc_gradient):
    if iter % 10 == 0:
      ability = np.mean(objective(genX_params, dsc_params, iter))
   
      fake_Z = generate_Z_from_noise(genZ_params, 10000, noise_dimZ, seed)
      noiseX = seed.randn(10000, noise_dimX)

      # GAN2: deterministic given noise
      fake_data = generate_X_from_given_noise_and_given_z(genX_params, 10000, noiseX, fake_Z, seed)
      
      # IGP-hat: deterministic given noise
      igp_data = igp_hat(fake_Z, noiseX)
      # igp_data = igp_hat(true_z[batch_indices(iter)])

      real_data = true_data
      # real_data = true_data[batch_indices(iter)]
      
      probs_fake = np.mean(sigmoid(neural_net_predict_dsc(dsc_params, fake_data)))
      probs_real = np.mean(sigmoid(neural_net_predict_dsc(dsc_params, real_data)))
      print("{:15}|{:20}|{:20}|{:20}".format(iter//num_batches, ability, probs_fake, probs_real))
      measure_meaningful_z(fake_data, igp_data, meaning_z_dir, alphabetize(int(iter/10)))
      save_images(fake_data, igp_data, real_data, out_dir, alphabetize(int(iter/10)), dsc_params)
      return ability
    return None

  # The optimizers provided can optimize lists, tuples, or dicts of parameters.
  optimized_params = adam_minimax(both_objective_grad,
                                  init_genX_params, init_dsc_params,
                                  step_size_max=step_size_max, step_size_min=step_size_min,
                                  num_iters=num_epochs * num_batches, callback=print_perf)

  print('Done')
  create_gif(out_dir, meaning_z_dir)
  # import code; code.interact(local=dict(globals(), **locals()))