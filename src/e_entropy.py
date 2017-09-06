import autograd.numpy as np
from autograd import grad, elementwise_grad

def make_neighbors_indexes(data):
  data = list(data)
  sorteddata = sorted(data)
  ns = []
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
  return np.array(ns)

def make_neighbors_values(data):
  sorteddata = sorted(data)
  ns = []
  for i, d in enumerate(data):
    if i > 0 and i < len(data) - 1:
      n1 = sorteddata[sorteddata.index(d) - 1]
      n2 = sorteddata[sorteddata.index(d) + 1]
      ns.append( np.array([n1, n2]) )
  return np.array(ns)

# def entropy_loss(data):
#   # assume sorted
#   avg_dist = 0
#   for i in range(1, len(data)):
#     avg_dist += (data[i] - data[i-1]) ** 2
#   avg_dist /= len(data - 1)

#   loss = 0
#   for i in range(1, len(data) - 1):
#     sq_dist = 0.5 * (data[i] - data[i-1])**2 + 0.5 * (data[i+1] - data[i])**2
#     loss += np.abs( np.log(sq_dist) - np.log(avg_dist) ) 
#   return loss

# def entropy_loss_values(data, neighbors):
# #   # assume not sorted, but neighbors values given externally of autodiff
#   avg_dist = 0
#   for i in range(1, len(data)):
#     avg_dist += (data[i] - data[i-1]) ** 2
#   avg_dist /= len(data - 1)
#   print 'average distance:', avg_dist

#   loss = 0
#   for i in range(1, len(data) - 1):
#     sq_dist = 0.5 * (data[i] - neighbors[i-1][0])**2 + 0.5 * (neighbors[i-1][1] - data[i])**2
#     loss += np.abs( np.log(sq_dist) - np.log(avg_dist) ) 
#   return loss

def entropy_loss_indices(data, neighbors, debug = False):
#   # assume not sorted, but neighbor indices given
#   # using indices allows autodiff to have all info about how loss is associated with the data.
  loss = 0
  for i in range(len(data)):
    sq_dist = 0
    if i == neighbors[i][0] or i == neighbors[i][1]:
      continue
    # sq_dist += 0.5 * (data[i] - data[neighbors[i][0]])**2
    sq_dist += 0.5 * np.log(np.abs(data[i] - data[neighbors[i][0]]))
    # sq_dist += 0.5 * (data[i] - data[neighbors[i][1]])**2
    sq_dist += 0.5 * np.log(np.abs(data[i] - data[neighbors[i][1]]))
    # loss += - np.log(sq_dist) 
    loss += -sq_dist
    if debug:
      # print data[i], data[neighbors[i][0]], data[neighbors[i][1]], sq_dist, np.log(sq_dist)
      print data[i], data[neighbors[i][0]], data[neighbors[i][1]], sq_dist
      print '\t', i, neighbors[i][0], neighbors[i][1]
    # loss += np.log(avg_dist) - np.log(sq_dist) 
    # loss += np.abs( np.log(sq_dist) - np.log(avg_dist) ) 
  return loss

def get_avg_dist(data, neighbors):
  avg_dist = 0
  for i in range(1, len(data)):
    avg_dist += (data[i] - data[neighbors[i][0]]) ** 2
  avg_dist /= len(data) - 1

  checked_avg_dist = 0
  sorted_data = sorted(data)
  for i in range(1, len(sorted_data)):
    checked_avg_dist += (sorted_data[i] - sorted_data[i-1]) ** 2
  checked_avg_dist /= len(sorted_data) - 1
  
  return avg_dist

import random
random.seed(0)

# clumped between 1.04 - 1.10
data = [-1.0, 0.0, 1.0, 1.04, 1.05, 1.055, 1.0575, 1.06, 1.065, 1.075, 1.10, 2.0, 3.0, 4.0, 5.0, 6.0]
random.shuffle(data)

# all items must be unique: taking log of 0 is bad.
assert len(set(data)) == len(data)
neighbors = make_neighbors_indexes(data)
print neighbors

data = np.array(data)


# gew = elementwise_grad(entropy_loss)
g = grad(entropy_loss_indices)
# g = elementwise_grad(entropy_loss_indices)


print '\tCurrent point:\n', '\n'.join([str(s) for s in sorted(data)])
print '\tCurrent avg. dist:', get_avg_dist(data, neighbors)
curr_loss = entropy_loss_indices(data, neighbors)
print '\tCurrent loss', curr_loss
print '\tGradient:\n', '\n'.join([str(s) for s in g(data, neighbors)])
print '\tUnsorted data:\n', '\n'.join([str(s) for s in data])

stepsize = 1 / max(abs( np.nan_to_num( g(data, neighbors) )))
prop_accepted = 0
for iter in range(1000):
  print iter
  proposal = data - stepsize * np.nan_to_num( g(data, neighbors) )
  print '\tProposal:\n', '\n'.join([str(s) for s in sorted(proposal)])
  proposed_neighbors = make_neighbors_indexes(proposal)
  print '\tProposed avg. dist:', get_avg_dist(proposal, proposed_neighbors)
  proposed_loss = entropy_loss_indices(proposal, proposed_neighbors)
  print '\tProposed loss:', proposed_loss

  # import code; code.interact(local=dict(globals(), **locals()))

  if stepsize < 0.0001:
    break

  if proposed_loss > curr_loss:
    stepsize /= 2
    print 'Rejected proposal, halving step size to', stepsize
    continue

  data = proposal
  neighbors = proposed_neighbors
  curr_loss = proposed_loss
  stepsize = 1 / max(abs( np.nan_to_num( g(data, neighbors) )))
  prop_accepted += 1

print '\n\nProposals accepted:', prop_accepted
print '\tFinal data:\n', '\n'.join([str(s) for s in sorted(data)])
print '\tCurrent avg. dist:', get_avg_dist(data, neighbors)
print '\tCurrent loss:', curr_loss
import code; code.interact(local=dict(globals(), **locals()))