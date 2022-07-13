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

class ResBlock(torch.nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, inputs):
        x = self.bn1(self.c1(F.relu(inputs)))
        x = self.bn2(self.c2(F.relu(x)))
        return x + inputs

class ConvACNVQVAE(torch.nn.Module):
    def __init__(self, hidden_size, code_len, batch_size):
        super(ConvACNVQVAE, self).__init__()
        assert hidden_size % 4 == 0

        self.hidden_size = hidden_size
        hidden_size = self.hidden_size

        self.c1 = nn.Conv2d(1, hidden_size, 5, 1, padding=0)
        self.c2 = nn.Conv2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.c3 = nn.Conv2d(hidden_size, hidden_size, 5, 1, padding=0)

        self.res1 = ResBlock(hidden_size)
        self.res2 = ResBlock(hidden_size)
        self.c4 = nn.Conv2d(hidden_size, hidden_size, 3, 1, padding=1)

        self.c5 = nn.Conv2d(hidden_size, hidden_size, 3, 1, padding=1)


        self.c5_mu = nn.Conv2d(hidden_size, hidden_size // 4, 1, 1, padding=0)
        self.c5_log_std = nn.Conv2d(hidden_size, hidden_size // 4, 1, 1, padding=0)

        self.l5_mu = nn.Linear(hidden_size // 4 * 16 * 16, code_len)
        self.l5_log_std = nn.Linear(hidden_size // 4 * 16 * 16, code_len)

        self.il5 = nn.Linear(code_len, hidden_size // 4 * 16 * 16)

        self.c1_v = nn.Conv2d(hidden_size // 4, hidden_size, 1, 1, padding=0)
        self.c1_v_bn = nn.BatchNorm2d(hidden_size)
        self.res1_v = ResBlock(hidden_size)
        self.c1_v_last = nn.Conv2d(hidden_size, hidden_size, 1, 1, padding=0)


        self.ic4 = nn.ConvTranspose2d(hidden_size, hidden_size, 1, 1, padding=0)
        self.ires1 = ResBlock(hidden_size)
        self.ires2 = ResBlock(hidden_size)

        self.ic3 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.ic2 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.ic1 = nn.ConvTranspose2d(hidden_size, hidden_size, 5, 1, padding=0)
        self.ic1_o = nn.ConvTranspose2d(hidden_size, 1, 1, 1, padding=0)

        embedding_dim = hidden_size

        num_embeddings = 512
        commitment_cost = 0.25
        decay = 0.99
        epsilon = 1E-5

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.zeros(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = epsilon

    def vq_indices_to_codes(self, encoding_indices):
        # indices in batch, H, W form
        # returns batch, embedding_dim, H, W
        input_shape = encoding_indices.shape
        input_shape = (input_shape[0], input_shape[1], input_shape[2], self._embedding_dim)
        flat_encoding_indices = encoding_indices.reshape(-1, 1)
        encodings = torch.zeros(flat_encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, flat_encoding_indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = quantized.permute(0, 3, 1, 2)
        return quantized

    def vq(self, pre_vq_latents):
        # Courtesy of zalando
        # https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
        inputs = pre_vq_latents
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances - TODO: speed this up using embedding update tricks? like in fast acn
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        # don't need loss and perplexity and all that jazz
        #return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        spatial_encodings = encodings.reshape(-1, 16, 16, self._num_embeddings)
        return quantized.permute(0, 3, 1, 2).contiguous(), spatial_encodings.permute(0, 3, 1, 2)

    def encode(self, inputs):
        h1 = F.relu(self.c1(inputs))
        h2 = F.relu(self.c2(h1))
        h3 = F.relu(self.c3(h2))
        # now 16x16

        r1 = self.res1(h3)
        r2 = self.res2(r1)

        # project down to smaller dim before flattening to avoid humongous linear
        fh = self.c5(self.c4(r2))

        acn_mu_i = self.c5_mu(fh)
        acn_log_std_i = self.c5_log_std(fh)
        # split to mu, logstd
        shp = acn_mu_i.shape

        # flatten
        acn_mu_flat_pre = acn_mu_i.reshape((shp[0], -1))
        acn_log_std_flat_pre = acn_log_std_i.reshape((shp[0], -1))

        # project down to code size
        acn_mu_flat = self.l5_mu(acn_mu_flat_pre)
        acn_log_std_flat = self.l5_log_std(acn_log_std_flat_pre)

        # reparameterize
        acn_z_flat = self.acn_reparametrize(acn_mu_flat, acn_log_std_flat)

        # project back up to right size to reshape
        acn_z = self.il5(acn_z_flat).reshape(shp)
        # inner projection
        vq_e_z = self.c1_v_last(self.res1_v(F.relu(self.c1_v_bn(self.c1_v(acn_z)))))
        # vq
        vq_q_z, latents = self.vq(vq_e_z)
        return acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, latents

    def decode(self, vq_q_z):
        latent = vq_q_z
        ih1 = self.ic4(latent)
        ires1 = self.ires1(ih1)
        ires2 = self.ires2(ires1)

        ih2 = F.relu(self.ic3(ires2))
        ih3 = F.relu(self.ic2(ih2))
        ih4 = self.ic1_o(self.ic1(ih3))
        return ih4

    def forward(self, inputs):
        acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices = self.encode(inputs)
        shp = vq_e_z.shape
        # batch, rest
        output = self.decode(vq_q_z)
        return output, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices

    def acn_reparametrize(self, u, s):
        """
        #
        Draw from standard normal distribution (as input) and then parametrize it (so params can be backpropped).

        Args:
            u: Means.
            s: Log standard deviations.

        Returns: Draws from a distribution.
        """
        if self.training:
            std = s.exp()
            activations = u + std * torch.randn_like(u)
            return activations
        return u


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
    def __init__(self, n_hidden, code_length, size_training_set, k=5, random_seed=4543):
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
