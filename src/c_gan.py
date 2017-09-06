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

### Define geneerator, discriminator, and objective ###

def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)
def logsigmoid(x): return x - np.logaddexp(0, x)
def leaky_relu(x): return np.maximum(0, x) + np.minimum(0, x) * 0.01

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    inpW, inpb = params[0]
    inputs = leaky_relu(np.dot(inputs, inpW) + inpb)
    for W, b in params[1:-1]:
        outputs = batch_normalize(np.dot(inputs, W) + b)
        inputs = leaky_relu(outputs)
        # inputs = relu(outputs)
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

def generate_from_noise(gen_params, num_samples, noise_dim, rs):
    noise = rs.randn(num_samples, noise_dim)
    samples = neural_net_predict(gen_params, noise)
    return sigmoid(samples)

def igp_hat(zs):
    try:
        assert (0 <= zs).all() and (zs <= 1).all(), 'Bad data'
    except AssertionError:
        import code; code.interact(local=dict(globals(), **locals()))
    # true_means_0 = np.array([-1, 1, -1, 3, 4])
    # true_means_1 = np.array([1, -1, 2, 3, 0])

    true_means_0 = np.array([-5])
    true_means_1 = np.array([5])
    
    std = 0.00
    # differs in last column, is 2 instead of 0.
    alt_means_1 = np.array([1, -1, 2, 3, 2])
    # xs = true_means_0 + zs * alt_means_1
    xs = (1-zs) * true_means_0 + zs * true_means_1
    xs = xs + np.random.normal(0, std, xs.shape)
    return xs 

def gan_objective(gen_params, dsc_params, real_data, num_samples, noise_dim, rs):
    fake_z = generate_from_noise(gen_params, num_samples, noise_dim, rs)
    fake_data = igp_hat(fake_z)
    # fake_data = fake_z
    logprobs_fake = logsigmoid(neural_net_predict(dsc_params, fake_data))
    logprobs_real = logsigmoid(neural_net_predict(dsc_params, real_data))
    return np.mean(logprobs_real) - np.mean(logprobs_fake)

def alphabetize(num):
    mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
    hundreds = int(num / 260)
    tens = int(num / 26)
    ones = num % 26
    return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def save_images(fake_data, out_fn, dsc_params, vmin=0, vmax=1):
    assert (0 <= fake_data).all() and (fake_data <= 1).all(), 'bad data' 

    curr_dsc = sigmoid(neural_net_predict(dsc_params, np.arange(0, 1, 0.01).reshape(100, 1)))
    fig, ax1 = plt.subplots()
    ax1.hist(fake_data, bins = np.arange(0, 1, 0.01))
    ax1.set_ylabel('Histogram counts', color = 'b')
    ax1.tick_params('y', colors = 'b')
    ax1.set_xlim(-0.1, 1.1)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, 1, 0.01), curr_dsc, 'r')
    ax2.set_ylabel('Discriminator label | 0=real ; 1=fake', 
                    color = 'r')
    ax2.set_ylim(0, 1)
    ax2.tick_params('y', colors = 'r')
    fig.tight_layout()
    plt.title('Generated Z')
    plt.savefig(out_fn)
    plt.close()
    return

### Define minimax version of adam optimizer ###

def adam_minimax(grad_both, init_params_max, init_params_min, callback=None, num_iters=100,
         step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam modified to do minimiax optimization, for instance to help with
    training generative adversarial networks."""

    x_max, unflatten_max = flatten(init_params_max)
    x_min, unflatten_min = flatten(init_params_min)

    m_max = np.zeros(len(x_max))
    v_max = np.zeros(len(x_max))
    m_min = np.zeros(len(x_min))
    v_min = np.zeros(len(x_min))
    ability = 0
    HANDICAP = 100
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


def create_gif(out_dir):
  print('Creating GIF...')
  subprocess.call('convert -delay 15 -loop 0 ' + out_dir + '*.png ' + out_dir + '_anim.gif', shell = True)
  print('Done.')
  return

def clear_pngs(out_dir):
  subprocess.call('rm -rf ' + out_dir + '*.png', shell = True)
  return

def count_num_folders(out_dir):
  for fold in os.listdir(out_dir):
    assert os.path.isdir(out_dir + fold), 'Not a folder!'
  return len(os.listdir(out_dir))

### Setup and run ###

if __name__ == '__main__':
    out_place = '/cluster/mshen/prj/gans/out/2017-06-19/c_gan/'
    num_folds = count_num_folders(out_place)
    out_dir = out_place + alphabetize(num_folds + 1) + '/'
    util.ensure_dir_exists(out_dir)
    print('outdir: ' + alphabetize(num_folds + 1))

    clear_pngs(out_dir)
    counter = 0

    # Model hyper-parameters
    noise_dim = 1
    gen_layer_sizes = [noise_dim, 8, 8, 1]
    # dsc_layer_sizes = [1, 16, 16, 16, 1]
    dsc_layer_sizes = [1, 16, 16, 16, 1]

    # Training parameters
    param_scale = 0.05
    batch_size = 10000
    num_epochs = 1000
    step_size_max = 0.001
    step_size_min = 0.001


    print("Loading training data...")
    train_data = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/trimodal/X.csv', delimiter = ',')
    train_data = train_data.reshape(len(train_data), 1)

    # train_data = np.loadtxt('/cluster/mshen/prj/gans/out/2017-06-19/a_generate/X.csv', delimiter = ',')
    # train_data = train_data.reshape(len(train_data), 1)

    init_gen_params = init_random_params(param_scale, gen_layer_sizes)
    init_dsc_params = init_random_params(param_scale, dsc_layer_sizes)

    num_batches = int(np.ceil(len(train_data) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(1)
    def objective(gen_params, dsc_params, iter):
        idx = batch_indices(iter)
        return gan_objective(gen_params, dsc_params, train_data[idx],
                             batch_size, noise_dim, seed)

    # Get gradients of objective using autograd.
    both_objective_grad = multigrad(objective, argnums=[0,1])

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(gen_params, dsc_params, iter, gen_gradient, dsc_gradient):
        if iter % 10 == 0:
            ability = np.mean(objective(gen_params, dsc_params, iter))
            fake_z = generate_from_noise(gen_params, batch_size, noise_dim, seed)
            fake_data = igp_hat(fake_z)
            # fake_data = fake_z
            real_data = train_data[batch_indices(iter)]
            probs_fake = np.mean(sigmoid(neural_net_predict(dsc_params, fake_data)))
            probs_real = np.mean(sigmoid(neural_net_predict(dsc_params, real_data)))
            print("{:15}|{:20}|{:20}|{:20}".format(iter//num_batches, ability, probs_fake, probs_real))
            save_images(fake_z, out_dir + 'gan_samples_' + alphabetize(int(iter/10)) + '.png', dsc_params, vmin=0, vmax=1)
            return ability
        return None

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam_minimax(both_objective_grad,
                                    init_gen_params, init_dsc_params,
                                    step_size_max=step_size_max, step_size_min=step_size_min,
                                    num_iters=num_epochs * num_batches, callback=print_perf)

    print('Done')
    create_gif(out_dir)
    import code; code.interact(local=dict(globals(), **locals()))