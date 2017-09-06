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
import matplotlib
import matplotlib.pyplot as plt
import subprocess, os, datetime
from mylib import util
import seaborn as sns
from matplotlib.colors import Normalize
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

  # true_means_0 = np.array([-50, -50])
  # true_means_0 = np.array([-50, -50])
  # true_means_1 = np.array([50, 50])
  true_means_0 = np.array([-60, -40])
  true_means_1 = np.array([40, 60])
 

  std = 1.00
  xs = (1-zs) * true_means_0 + zs * true_means_1
  # xs = np.minimum(6, xs)
  # xs = np.maximum(-6, xs)
  # xs[0] = xs[0] + std*noiseX
  # xs[1] = xs[1] - std*noiseX
  n = xs.T[0].shape[0]
  xs = np.hstack([xs.T[0].reshape(n, 1) + std*noiseX, xs.T[1].reshape(n, 1) - std*noiseX])
  # xs = xs + np.random.normal(0, std, xs.shape)
  return xs 

def wgan_objective(genX_params, genZ_params, dsc_params, real_data, real_z, num_samples, noise_dimX, noise_dimZ, rs):
  fake_data = generate_X_from_noise_and_genz(genX_params, genZ_params, num_samples, noise_dimX, noise_dimZ, rs)
  assert fake_data.shape == real_data.shape
  score_fake = neural_net_predict_dsc(fake_data, dsc_params)
  score_real = neural_net_predict_dsc(real_data, dsc_params)
  return np.mean(score_real) - np.mean(score_fake)

def igp_objective(genX_params, genZ_params, real_z, num_samples, noise_dimX, noise_dimZ, rs):
  fake_Z = generate_Z_from_noise(genZ_params, num_samples, noise_dimZ, rs)
  noiseX = rs.randn(num_samples, noise_dimX)

  # GAN2: deterministic given noise
  fake_data = generate_X_from_given_noise_and_given_z(genX_params, num_samples, noiseX, fake_Z, rs)
  
  # IGP-hat: deterministic given noise
  igp_data = igp_hat(fake_Z, noiseX)
  
  return np.sum((fake_data - igp_data)**2) / num_samples

def wasserstein_lipschitz_objective(genX_params, genZ_params, dsc_params, real_data, num_samples, noise_dimZ, noise_dimX, grad_dsc, wasserstein_lambda, rs):
  fake_z = generate_Z_from_noise(genZ_params, num_samples, noise_dimZ, rs)
  noiseX = rs.randn(num_samples, noise_dimX)
  fake_data = generate_X_from_given_noise_and_given_z(genX_params, num_samples, noiseX, fake_z, rs)
  
  # Wasserstein loss
  wasserstein_loss = 0
  K_LIPSCHITZ = 10
  for i in range(num_samples):
    eps = rs.uniform()
    x_hat = eps * real_data[i] + (1 - eps) * fake_data[i]
    gxh = grad_dsc(x_hat, dsc_params)
    norm = np.sqrt(sum(gxh ** 2))
    wasserstein_loss += (max(norm, K_LIPSCHITZ) - K_LIPSCHITZ)**2
    # wasserstein_loss += (norm - 1)**2
  wasserstein_loss /= num_samples
  wasserstein_loss *= wasserstein_lambda
  # print(wasserstein_loss)
  return wasserstein_loss

def alphabetize(num):
  assert num < 26**3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
  hundreds = int(num / (26*26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def save_images(fake_data, igp_data, real_data, out_dir, nm, dsc_params, iter):
  # plot igp objective, dimension 1
  binsize = (max(igp_data.T[0])-min(igp_data.T[0]))/100
  plt.hist(fake_data.T[0], 
           bins = np.arange(min(igp_data.T[0]), max(igp_data.T[0]), binsize ), 
           color = 'b', alpha = 0.5)
  plt.hist(igp_data.T[0], 
           bins = np.arange(min(igp_data.T[0]), max(igp_data.T[0]), binsize ), 
           color = 'g', alpha = 0.5)
  plt.ylabel('Histogram counts (Dimension 1)')
  plt.title('Generated X (blue) vs. IGP (green)')
  plt.savefig(out_dir + 'gan_samples_X_IGP_' + nm + '_d1.png')
  plt.close()

  # plot igp objective, dimension 2
  binsize = (max(igp_data.T[1])-min(igp_data.T[1]))/100
  plt.hist(fake_data.T[1], 
           bins = np.arange(min(igp_data.T[1]), max(igp_data.T[1]), binsize ), 
           color = 'b', alpha = 0.5)
  plt.hist(igp_data.T[1], 
           bins = np.arange(min(igp_data.T[1]), max(igp_data.T[1]), binsize ), 
           color = 'g', alpha = 0.5)
  plt.ylabel('Histogram counts (Dimension 2)')
  plt.title('Generated X (blue) vs. IGP (green)')
  plt.savefig(out_dir + 'gan_samples_X_IGP_' + nm + '_d2.png')
  plt.close()

  if iter % 10 == 0:
    g = sns.jointplot(x = fake_data.T[0], y = fake_data.T[1], stat_func = None, alpha = 0.1)
    plt.sca(g.ax_joint)
    sns.kdeplot(real_data.T[0], real_data.T[1], ax = g.ax_joint)

    rmin = np.array([-61, -41])
    rmax = np.array([41, 61])
    dsc_bins = np.zeros((103, 103))
    for _i in range(rmin[0], rmax[0]):
      for _j in range(rmin[1], rmax[1]):
        query_pt = np.array([_i, _j])
        val = neural_net_predict_dsc(query_pt, dsc_params)[0]
        dsc_bins[_i + 61][_j + 41] = val

    norm = Normalize(vmin = min(dsc_bins.flatten()), vmax = max(dsc_bins.flatten()))

    g.ax_joint.imshow(dsc_bins, interpolation = 'nearest', cmap = matplotlib.cm.autumn, origin = 'upper', norm = norm, extent = [-51, 51, -51, 51], alpha = 0.5)

    g.ax_marg_x.set_title('Critic: Red = Real, Yellow = Fake')
    plt.tight_layout()
    plt.savefig(out_dir + 'gan_samples_X_' + nm + '.png')
    plt.close()
  return

  return

def measure_meaningful_z(fake_data, igp_data, out_dir, nm):
  # 1st dimension
  plt.plot(igp_data.T[0], fake_data.T[0], '.', alpha = 0.3, zorder = 0)
  plt.title('Meaning of Z-space: Comparing GAN to IGP-hat output across Z-space')
  r2s = r2_score(igp_data.T[0], fake_data.T[0])
  plt.xlabel('IGP-hat output (1st dimension) | Rsq = ' + str(r2s))
  plt.ylabel('Generated data (1st dimension)')

  _min = min(plt.ylim()[0], plt.xlim()[0])
  _max = max(plt.ylim()[1], plt.xlim()[1])
  plt.plot([_min, _max], [_min, _max], 'k-', zorder = 10)

  plt.savefig(out_dir + nm + '_d1.png')
  plt.close()

  # 2nd dimension
  plt.plot(igp_data.T[1], fake_data.T[1], '.', alpha = 0.3, zorder = 0)
  plt.title('Meaning of Z-space: Comparing GAN to IGP-hat output across Z-space')
  r2s = r2_score(igp_data.T[1], fake_data.T[1])
  plt.xlabel('IGP-hat output (2nd dimension) | Rsq = ' + str(r2s))
  plt.ylabel('Generated data (2nd dimension)')

  _min = min(plt.ylim()[0], plt.xlim()[0])
  _max = max(plt.ylim()[1], plt.xlim()[1])
  plt.plot([_min, _max], [_min, _max], 'k-', zorder = 10)

  plt.savefig(out_dir + nm + '_d2.png')
  plt.close()

  return

### Define minimax version of adam optimizer ###

def adam_minimax(grad_both, init_params_max, init_params_min, callback=None, num_iters=100,
         step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
  """Adam modified to do minimiax optimization, for instance to help with
  training generative adversarial networks."""

  def exponential_decay(step_size_max):
    if step_size_max > 0.001:
        step_size_max *= 0.999
    return step_size_max

  x_max, unflatten_max = flatten(init_params_max)
  x_min, unflatten_min = flatten(init_params_min)

  m_max = np.zeros(len(x_max))
  v_max = np.zeros(len(x_max))
  m_min = np.zeros(len(x_min))
  v_min = np.zeros(len(x_min))
  for i in range(num_iters):
    print(i, datetime.datetime.now(), alphabetize(int(i/10)))
    K = 3

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
    
    step_size_max = exponential_decay(step_size_max)

    m_max = (1 - b1) * g_max      + b1 * m_max  # First  moment estimate.
    v_max = (1 - b2) * (g_max**2) + b2 * v_max  # Second moment estimate.
    mhat_max = m_max / (1 - b1**(i + 1))    # Bias correction.
    vhat_max = v_max / (1 - b2**(i + 1))
    x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)

    m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
    v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
    mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
    vhat_min = v_min / (1 - b2**(i + 1))
    x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)

    for k in range(K-1):
      if k <= 0:
        step_size_min_temp = step_size_min
      if k > 0:
        step_size_min_temp = step_size_min_temp * 0.50
      g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                 unflatten_min(x_min), i)
      g_min, _ = flatten(g_min_uf)

      # Update discriminator (minimizer)
      m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
      v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
      mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
      vhat_min = v_min / (1 - b2**(i + 1))
      x_min = x_min - step_size_min_temp * mhat_min / (np.sqrt(vhat_min) + eps)


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
  run_nm = 'all'
  iter_nm = 'adh'
  gp_fold = '/cluster/mshen/prj/gans/out/2017-06-19/c_gan/' + run_nm + '/gen_params/'
  genZ_params = import_ganZ_gen_params(gp_fold, iter_nm)

  # Model hyper-parameters
  noise_dimZ = 1    # pull from c_gan.py = GAN1.
  noise_dimX = 1
  z_dim = 1
  genX_layer_sizes = [z_dim + noise_dimX, 16, 16, 2]
  # dsc_layer_sizes = [1, 16, 16, 16, 1]
  dsc_layer_sizes = [2, 64, 64, 1]
  wasserstein_lambda = 100

  # Training parameters
  genX_param_scale = 0.1
  dsc_param_scale = 0.1
  batch_size = 500
  num_epochs = 50
  step_size_max = 0.10
  step_size_min = 0.05


  print("Loading training data...")
  true_data = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/mixture_2dx/X.csv', delimiter = ',')

  true_z = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/mixture_2dx/Z.csv', delimiter = ',')
  true_z = true_z.reshape(len(true_z), 1)
  assert true_z.shape[1] == z_dim, 'z_dim does not match data dim'

  init_genX_params = init_random_params(genX_param_scale, genX_layer_sizes)
  init_dsc_params = init_random_params(dsc_param_scale, dsc_layer_sizes)

  # Used for Wasserstein score
  grad_dsc = grad(neural_net_predict_dsc)

  num_batches = int(np.ceil(len(true_data) / batch_size))
  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

  # Define training objective
  seed = npr.RandomState(1)
  def objective(genX_params, dsc_params, iter):
    idx = batch_indices(iter)
    c1, c2 = c1c2_schedule(iter)
    return c1 * wgan_objective(genX_params, genZ_params, dsc_params, 
                         true_data[idx], true_z[idx],
                         batch_size, noise_dimX, noise_dimZ, seed) \
           - c2 * igp_objective(genX_params, genZ_params,
                         true_z[idx], 
                         batch_size, noise_dimX, noise_dimZ, seed) \
           + wasserstein_lipschitz_objective(genX_params,
                            genZ_params, 
                            dsc_params, 
                            true_data[idx], 
                            batch_size,
                            noise_dimZ, 
                            noise_dimX, 
                            grad_dsc, 
                            wasserstein_lambda, 
                            seed)

  def c1c2_schedule(iter):
    if iter < 100:
      return 0.1, 10
    else:
      return 1, max(10 - (300 - iter)/200, 0.01)
    return c1, c2

  # Get gradients of objective using autograd.
  both_objective_grad = multigrad(objective, argnums=[0,1])

  print("   Iter | Objective | Fake score | Real score | Copy score | Lipschitz Score")
  def print_perf(genX_params, dsc_params, iter, gen_gradient, dsc_gradient):
    if True:
    # if iter % 10 == 0:
      ability = np.mean(objective(genX_params, dsc_params, iter))
   
      fake_Z = generate_Z_from_noise(genZ_params, 10000, noise_dimZ, seed)
      noiseX = seed.randn(10000, noise_dimX)

      # GAN2: deterministic given noise
      fake_data = generate_X_from_given_noise_and_given_z(genX_params, 10000, noiseX, fake_Z, seed)
      
      c1, c2 = c1c2_schedule(iter)

      # IGP-hat: deterministic given noise
      igp_data = igp_hat(fake_Z, noiseX)
      # igp_data = igp_hat(true_z[batch_indices(iter)])

      real_data = true_data
      # real_data = true_data[batch_indices(iter)]
      
      score_fake = c1 * np.mean(neural_net_predict_dsc(fake_data, dsc_params))
      score_real = c1 * np.mean(neural_net_predict_dsc(real_data, dsc_params))
      copy_score = c2 * igp_objective(genX_params, genZ_params, fake_Z, batch_size, noise_dimX, noise_dimZ, seed)
      lipschitz_score = wasserstein_lipschitz_objective(genX_params,genZ_params, dsc_params, true_data, batch_size,noise_dimZ, noise_dimX, grad_dsc, wasserstein_lambda, seed)

      print("{:8}|{:11}|{:12}|{:12}|{:15}|{:12}".format(iter, ability, score_fake, score_real, copy_score, lipschitz_score))

      measure_meaningful_z(fake_data, igp_data, meaning_z_dir, alphabetize(int(iter/10)))
      save_images(fake_data, igp_data, real_data, out_dir, alphabetize(int(iter/10)), dsc_params, iter)

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