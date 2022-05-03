"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf

This is a VAE with a conditional prior.
# based on code from jalexvig
"""
import random

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

def equals(x, y, eps=1e-8):
    return torch.abs(x-y) <= eps

# courtesy of Mehdi C 
# https://github.com/mehdidc/ae_gen        
def spatial_sparsity(x):
    maxes = x.amax(dim=(2,3), keepdims=True)
    return x * equals(x, maxes)


class ConvWTA(torch.nn.Module):
    def __init__(self, code_len, batch_size):
        super(ConvWTA, self).__init__()
        self.hidden_size = 128
        hidden_size = self.hidden_size

        self.c1 = nn.Conv2d(1, hidden_size, 5, 1, padding=0)
        self.c2 = nn.Conv2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.c3 = nn.Conv2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.ic3 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.ic2 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.ic1 = nn.ConvTranspose2d(hidden_size, 1, 5, 1, padding=0)
        #self.ic3 = nn.Conv2d(hidden_size, hidden_size, 5, 1, padding=4)
        #self.ic2 = nn.Conv2d(hidden_size, hidden_size, 5, 1, padding=4)
        #self.ic1 = nn.Conv2d(hidden_size, 1, 5, 1, padding=4)

    def spatial_sparsity(self, h):
        return spatial_sparsity(h)

    def lifetime_sparsity(self, h, rate=0.05):
        shp = h.shape
        n = shp[0]
        c = shp[1]
        h_reshape = h.reshape((n, -1))
        thr, ind = torch.topk(h_reshape, int(rate * n), dim=0)
        batch_mask = 0. * h_reshape
        batch_mask.scatter_(0, ind, 1)
        batch_mask = batch_mask.reshape(shp)
        return h * batch_mask

    def channel_sparsity(self, h, rate=0.05):
        shp = h.shape
        n = shp[0]
        c = shp[1]
        h_reshape = h.reshape((n, c, -1))
        thr, ind = torch.topk(h_reshape, int(rate * c), dim=1)
        batch_mask = 0. * h_reshape
        batch_mask.scatter_(1, ind, 1)
        batch_mask = batch_mask.reshape(shp)
        return h * batch_mask

    def encode(self, inputs: torch.Tensor, spatial=True, lifetime=True):
        h1 = F.relu(self.c1(inputs))
        h2 = F.relu(self.c2(h1))
        h3 = F.relu(self.c3(h2))
        if spatial:
            h3 = self.spatial_sparsity(h3)
        if lifetime:
            h3 = self.lifetime_sparsity(h3)
        return h3

    def decode(self, latent: torch.Tensor, spatial=True, lifetime=True):
        ih1 = F.relu(self.ic3(latent))
        ih2 = F.relu(self.ic2(ih1))
        ih3 = torch.sigmoid(self.ic1(ih2))
        #ih3 = self.ic1(ih2)
        return ih3

    def forward(self, inputs: torch.Tensor):
        latent = self.encode(inputs)
        output = self.decode(latent)
        return output, latent


class SlowPriorNetwork(torch.nn.Module):
    def __init__(self, code_len, codebook_size, k=5):
        """
        Args:
            k: Number of neighbors to choose from when picking code to condition prior.
        """

        super().__init__()
        self.fc1 = nn.Linear(code_len, 512)
        self.fc2_u = nn.Linear(512, code_len)
        self.fc2_s = nn.Linear(512, code_len)

        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=2 * k)
        codes = torch.randn((codebook_size, code_len)).numpy()
        self.fit_knn(codes)

    def pick_close_neighbor(self, code: torch.Tensor) -> torch.Tensor:
        """
        K-nearest neighbors to choose a close code. This emulates an ordering of the original data.

        Args:
            code: Latent activations of current training example.

        Returns: Numpy array of same dimension as code.
        """

        # TODO(jalex): This is slow - can I make it faster by changing search algo/leaf size?
        neighbor_idxs = self.knn.kneighbors([code.detach().numpy()], return_distance=False)[0]
        valid_idxs = [n for n in neighbor_idxs if n not in self.seen]
        if len(valid_idxs) < self.k:

            codes_new = [c for i, c in enumerate(self.codes) if i not in self.seen]
            self.fit_knn(codes_new)

            return self.pick_close_neighbor(code)

        neighbor_codes = [self.codes[idx] for idx in valid_idxs]

        if len(neighbor_codes) > self.k:
            code_np = code.detach().numpy()
            # this is same metric KNN uses
            neighbor_codes = sorted(neighbor_codes, key=lambda n: ((code_np - n) ** 2).sum())[:self.k]

        neighbor = random.choice(neighbor_codes)

        return neighbor

    def fit_knn(self, codes: np.ndarray):
        """
        Reset the KNN. This can be used when we get too many misses or want to update the codes.

        Args:
            codes: New codes to fit.
        """

        self.codes = codes
        self.seen = set()

        y = [0] * len(codes)

        self.knn.fit(codes, y)

    def forward(self, codes: torch.Tensor):
        """
        Calculate prior conditioned on codes.

        Args:
            codes: latent activations.

        Returns: Two parameters each of dimensionality codes. These can be used as mu, std for a Gaussian.
        """
        # Can use this to emulate uncoditional prior
        # return torch.zeros(codes.shape[0], 1), torch.ones(codes.shape[0], 1)
        previous_codes = [self.pick_close_neighbor(c) for c in codes]
        previous_codes = torch.tensor(np.array(previous_codes))
        return self.encode(previous_codes)

    def encode(self, prev_code: torch.Tensor):
        h1 = F.relu(self.fc1(prev_code))
        mu, logstd = self.fc2_u(h1), self.fc2_s(h1)
        return mu, logstd


class PriorNetwork(nn.Module):
    def __init__(self, code_length, size_training_set, n_hidden=512, k=5, random_seed=4543):
        super(PriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.input_layer = nn.Linear(code_length, n_hidden)
        self.skipin_to_2 = nn.Linear(n_hidden, n_hidden)
        self.skipin_to_3 = nn.Linear(n_hidden, n_hidden)
        self.skip1_to_out = nn.Linear(n_hidden, n_hidden)
        self.skip2_to_out = nn.Linear(n_hidden, n_hidden)
        self.h1 = nn.Linear(n_hidden, n_hidden)
        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.h3 = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, self.code_length)
        self.fc_s = nn.Linear(n_hidden, self.code_length)

        # needs to be a param so that we can load
        self.codes = nn.Parameter(torch.FloatTensor(self.rdn.standard_normal((self.size_training_set, self.code_length))), requires_grad=False)

        self.codes2_init = False
        self.codes2 = torch.FloatTensor((self.size_training_set, 1))

        # start off w/ default batch size - this will change automatically if
        # different input is given
        batch_size = 64
        n_neighbors = 5
        self.neighbors = torch.LongTensor((batch_size, n_neighbors))
        self.distances = torch.FloatTensor((batch_size, n_neighbors))
        self.batch_indexer = torch.LongTensor(torch.arange(batch_size))


    def to(self, device):
        new_self = super(PriorNetwork, self).to(device)
        new_self.neighbors = new_self.neighbors.to(device)
        new_self.distances = new_self.distances.to(device)
        new_self.batch_indexer = new_self.batch_indexer.to(device)
        new_self.codes2 = new_self.codes2.to(device)
        return new_self

    def update_codebook(self, indexes, values):
        assert indexes.min() >= 0
        assert indexes.max() < self.codes.shape[0]
        self.codes[indexes] = values
        self.codes2[indexes] = torch.square(self.codes[indexes]).sum(axis=1)[:, None]

    def kneighbors(self, test, n_neighbors):
        with torch.no_grad():
            device = test.device
            bs = test.shape[0]
            return_size = (bs, n_neighbors)
            # dont recreate unless necessary
            if self.neighbors.shape != return_size:
                print('updating prior sizes')
                self.neighbors = torch.LongTensor(torch.zeros(return_size, dtype=torch.int64)).to(device)
                self.distances = torch.zeros(return_size).to(device)
                self.batch_indexer = torch.LongTensor(torch.arange(bs)).to(device)

            if device != self.codes.device:
                print('transferring prior to %s'%device)
                self.neighbors = self.neighbors.to(device)
                self.distances = self.distances.to(device)
                self.codes = self.codes.to(device)

            if self.codes2_init == False:
                self.codes2_init = True
                self.codes2 = torch.square(self.codes).sum(axis=1)[:, None]
                # DONE: can do dist calc like Roland's hebbian K-means
                # http://www.cs.toronto.edu/~rfm/code/hebbian_kmeans.py
                #     X2 = (X**2).sum(1)[:, None]
                #     for epoch in range(numepochs):
                #         for i in range(0, X.shape[0], batchsize):
                #             D = -2*numpy.dot(W, X[i:i+batchsize,:].T) + (W**2).sum(1)[:, None] + X2[i:i+batchsize].T
                # Where X is self.codes, W is the incoming codes (or vice versa)
                # Would need a bit of tricks to only update the entries of X and X2 that changed
                # But would allow to cache more computations
                # see detailed writeup https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
                # this should save memory and be a decent speedup on large datasets, potentially

            dists = -2 * torch.matmul(test, self.codes.T) + torch.square(test).sum(axis=1)[:, None] + self.codes2.T
            # broadcast it
            #dists = torch.sum(torch.square(test[:, None] - self.codes[None]), dim=-1)
            bidxs = self.batch_indexer
            self.distances[bidxs], self.neighbors[bidxs] = dists.topk(n_neighbors, largest=False, dim=1)
            del dists
        return self.distances.detach(), self.neighbors.detach()

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training
        '''
        neighbor_distances, neighbor_indexes = self.kneighbors(codes, n_neighbors=self.k)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            chosen_neighbor_index = torch.LongTensor(self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize))
        else:
            chosen_neighbor_index = torch.LongTensor(torch.zeros(bsize, dtype=torch.int64))
        return self.codes[neighbor_indexes[self.batch_indexer, chosen_neighbor_index]]

    def forward(self, codes):
        previous_codes = self.batch_pick_close_neighbor(codes)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        """
        The prior network was an
        MLP with three hidden layers each containing 512 tanh
        units
        - and skip connections from the input to all hidden
        layers and
        - all hiddens to the output layer.
        """
        i = torch.tanh(self.input_layer(prev_code))
        # input goes through first hidden layer
        _h1 = torch.tanh(self.h1(i))

        # make a skip connection for h layers 2 and 3
        _s2 = torch.tanh(self.skipin_to_2(_h1))
        _s3 = torch.tanh(self.skipin_to_3(_h1))

        # h layer 2 takes in the output from the first hidden layer and the skip
        # connection
        _h2 = torch.tanh(self.h2(_h1+_s2))

        # take skip connection from h1 and h2 for output
        _o1 = torch.tanh(self.skip1_to_out(_h1))
        _o2 = torch.tanh(self.skip2_to_out(_h2))
        # h layer 3 takes skip connection from prev layer and skip connection
        # from nput
        _o3 = torch.tanh(self.h3(_h2+_s3))

        out = _o1+_o2+_o3
        mu = self.fc_mu(out)
        logstd = self.fc_s(out)
        return mu, logstd


def calc_loss(x, recon, u_q, s_q, u_p, s_p):
    """
    Loss derived from variational lower bound (ELBO) or information theory (see bits-back for details).

    The loss comprises two parts:

    1. Reconstruction loss (how good the VAE is at reproducing the output).
    2. The coding cost (KL divergence between the model posterior and conditional prior).

    Args:
        x: Inputs.
        recon: Reconstruction from a VAE.
        u_q: Mean of model posterior.
        s_q: Log std of model posterior.
        u_p: Mean of (conditional) prior.
        s_p: Log std of (conditional) prior.

    Returns: Loss.
    """

    # reconstruction
    xent = F.binary_cross_entropy(recon, x, reduction='none')

    # coding cost
    dkl = torch.sum(s_p - s_q - 0.5 +
                    ((2 * s_q).exp() + (u_q - u_p).pow(2)) /
                    (2 * (2 * s_p).exp()))

    return xent + dkl
