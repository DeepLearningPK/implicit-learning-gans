g_max_uf, g_min_uf = grad_both(unflatten_max(x_max), unflatten_min(x_min), i, neighbors_function)
g_max, _ = flatten(g_max_uf)
g_min, _ = flatten(g_min_uf)

callback(unflatten_max(x_max), unflatten_min(x_min), i, unflatten_max(g_max), unflatten_min(g_min))

for k in range(K-1):
  g_max_uf, g_min_uf = grad_both(unflatten_max(x_max), unflatten_min(x_min), i, neighbors_function)
  g_min, _ = flatten(g_min_uf)
  # Update discriminator (minimizer)
  m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
  v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
  mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
  vhat_min = v_min / (1 - b2**(i + 1))
  x_min = x_min - step_size_min_temp * mhat_min / (np.sqrt(vhat_min) + eps)
  callback(unflatten_max(x_max), unflatten_min(x_min), i, unflatten_max(g_max), unflatten_min(g_min))
  print(k)


# Single update

g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                         unflatten_min(x_min), i, neighbors_function)
g_min, _ = flatten(g_min_uf)
# Update discriminator (minimizer)
m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
vhat_min = v_min / (1 - b2**(i + 1))
x_min = x_min - step_size_min_temp * mhat_min / (np.sqrt(vhat_min) + eps)
callback(unflatten_max(x_max), unflatten_min(x_min), i, unflatten_max(g_max), unflatten_min(g_min))

gp_fold = '/cluster/mshen/prj/gans/out/2017-06-19/c_gan/ajc/gen_params/'
iter_nm = 'akb'
genZ_params = import_ganZ_gen_params(gp_fold, iter_nm)
x_max, unflatten_max = flatten(genZ_params)