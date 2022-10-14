from parallel_perceiversunmask_mnist_vq_models import PerceiverSUNMASK, clipping_grad_value_, RampOpt, top_k_top_p_filtering
from conv_acn_vq_models import ConvACNVQVAE, PriorNetwork
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import imageio
import time
import copy

import random

def seed_everything(seed=1234):
    random.seed(seed)
    tseed = random.randint(1, 1E6)
    tcseed = random.randint(1, 1E6)
    npseed = random.randint(1, 1E6)
    ospyseed = random.randint(1, 1E6)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(tseed)
    torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    # cannot set these inside colab :(
    #torch.use_deterministic_algorithms(True)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['PYTHONHASHSEED'] = str(ospyseed)

# from jalexvig
def parse_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quad', default='A', choices=['A', 'B', 'C', 'D'], help='Quadrant to target.')
    parser.add_argument('--data_quad', default=None, choices=['A', 'B', 'C', 'D'], help='Swap quadrant of loader.')
    parser.add_argument('--ground_truth', default=None, choices=['True'], help='Use groundtruth data')
    args = parser.parse_args()
    return args

class IndexedDataset(Dataset):
    def __init__(self, dataset_function, path, train=True, download=True, transform=transforms.ToTensor()):
        """ class to provide indexes into the data -- needed for ACN prior
        """
        self.indexed_dataset = dataset_function(path,
                             download=download,
                             train=train,
                             transform=transform)

    def __getitem__(self, index):
        data, target = self.indexed_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.indexed_dataset)

# as an alternative to IndexedDataset from the acn/ and even_faster_acn/, use our own data iterator
# base mnist from 
# https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz
# vq quantized in
# conv_acn_vq code
# this also lets us load other mnist like datasets easily
#d = np.load("mnist.npz")
#print(d.files)
#['test', 'test_labels', 'train', 'train_labels']
mnist_path = "mnist_vq.npz"
data_f = np.load(mnist_path)

comb_fixed_split_inds = np.random.RandomState(7777).randint(0, 60000, size=10000)
valid_fixed_split_inds = np.sort(comb_fixed_split_inds[:5000])
test_fixed_split_inds = np.sort(comb_fixed_split_inds[-5000:])
train_fixed_split_inds = np.array([el for el in np.arange(60000) if el not in comb_fixed_split_inds])
#d["train_vq_indices"]
#d["train_vq_match_indices"]
#d["train_vq_match_dists"]
#d["test_vq_match_indices"]
#d["test_vq_match_dists"]

#kuzu_train = "/content/drive/MyDrive/kuzushiji_MNIST/k49-train-imgs.npz"
#kuzu_test = "/content/drive/MyDrive/kuzushiji_MNIST/k49-test-imgs.npz"
def dataset_itr(batch_size, subset_type="train", seed=1234):
    """
    Coroutine of experience replay.
    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    if subset_type == "train":
        data_np = data_f["train_vq_indices"]
        data_np = data_np[train_fixed_split_inds]
    elif subset_type == "valid":
        data_np = data_f["train_vq_indices"]
        data_np = data_np[valid_fixed_split_inds]
    elif subset_type == "test":
        data_np = data_f["train_vq_indices"]
        data_np = data_np[test_fixed_split_inds]
    else:
        raise ValueError("Unknown subset_type {}".format(subset_type))

    max_sz = len(data_np)
    random_state = np.random.RandomState(seed)
    while True:
        inds = np.arange(max_sz)
        batch_inds = random_state.choice(inds, size=batch_size, replace=True)
        batch = data_np[batch_inds]
        batch = batch.reshape((batch.shape[0], 1, 16, 16))
        # return in a similar format as pytorch
        yield batch, batch_inds


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 12
# this is a multiple of 4 - so 16 * 4 -> 64 latent length is 1 measure
latent_length = int(8 * 8)
# 3 other quadrants for total of 4
sequence_length = int(4 * latent_length)
n_classes = 512

n_unrolled_steps = 2

hidden_size = 380
self_inner_dim = 900
input_dropout_keep_prob = 1.0
cross_attend_dropout_keep_prob = 1.0
autoregression_dropout_keep_prob = 1.0
inner_dropout_keep_prob = 1.0
final_dropout_keep_prob = 1.0
n_layers = 16

args = parse_flags()
model_save_path = "parallel_perceiversunmask_mnist_vq_models_quad_{}".format(args.quad)

clip_grad = 3
n_train_steps = 125000
learning_rate = 0.0003
min_learning_rate = 1E-5
ramp_til = 10000
decay_til = n_train_steps - 5000
valid_steps_per = 20
save_every = n_train_steps // 10
show_every = max(1, n_train_steps // 500)

train_itr = dataset_itr(batch_size, subset_type="train", seed=123)
# start coroutine
next(train_itr);

valid_itr = dataset_itr(batch_size, subset_type="valid", seed=1234)
# start coroutine
next(valid_itr);

test_itr = dataset_itr(batch_size, subset_type="test", seed=12345)
# start coroutine
next(test_itr);

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

FPATH_PERCEIVER = os.path.join(model_save_path, 'perceiversunmask_mnist_vq_{}.pth')
FPATH_LOSSES = os.path.join(model_save_path, 'perceiversunmask_mnist_vq_losses_{}.npz')

model = PerceiverSUNMASK(n_classes=n_classes,
                         z_index_dim=latent_length,
                         n_processor_layers=n_layers,
                         input_embed_dim=hidden_size,
                         num_z_channels=hidden_size,
                         inner_expansion_dim=self_inner_dim,
                         input_dropout_keep_prob=input_dropout_keep_prob,
                         cross_attend_dropout_keep_prob=cross_attend_dropout_keep_prob,
                         autoregression_dropout_keep_prob=autoregression_dropout_keep_prob,
                         inner_dropout_keep_prob=inner_dropout_keep_prob,
                         final_dropout_keep_prob=final_dropout_keep_prob)
model = model.to(device)
#logits, masks = self.perceiver(inputs, input_idxs, input_mask)
print("Using device {}".format(device))
import glob
saved_model_paths = glob.glob(FPATH_PERCEIVER.format("*"))
saved_model_paths = sorted(saved_model_paths, key=lambda x: int(x.split("_")[-1].split(".pth")[0]))
model.load_state_dict(torch.load(saved_model_paths[-1], map_location=device))
print("Loaded {}".format(saved_model_paths[-1]))

'''
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1E9):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    """
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(logits, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).sum(-1, keepdim=True)

        #shift and sort
        shifted_scores = torch.abs((-ent) - normalized)
        _, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
      
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = sorted_indices_to_remove.long()

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0

        sorted_indices = torch.tensor(sorted_indices.cpu().data.numpy())
        shp = logits.shape
        logits_red = logits.reshape((-1, shp[-1]))
        sorted_indices_red = sorted_indices.reshape((-1, shp[-1]))
        sorted_indices_to_remove_red = sorted_indices_to_remove.reshape((-1, shp[-1]))
        for i in range(shp[0]):
            logits_red[i][sorted_indices_red[i]] = logits_red[i][sorted_indices_red[i]] * (1. - sorted_indices_to_remove_red[i]) + sorted_indices_to_remove_red[i] * filter_value
        logits = logits_red.reshape(shp)
    return logits


def typical_top_k_filtering(logits, top_k=0, top_p=0.0, temperature=1.0, min_tokens_to_keep=1, filter_value=-1E12):
    """ Filter a distribution of logits using typicality, with optional top-k and/or nucleus (top-p) filtering
        Meister et. al. https://arxiv.org/abs/2202.00666
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep top k tokens with highest prob (top-k filtering).
            top_p >0.0: keep the top p tokens which compose cumulative probability mass top_p (nucleus filtering).
            min_tokens_to_keep >=1: always keep at least this many tokens through the top_p / nucleus sampling
    """
    # https://arxiv.org/abs/2202.00666
    # based on hugging face impl but added top k
    # https://github.com/cimeister/typical-sampling/commit/0f24c9409dc078ed23982197e8af1439093eedd3#diff-cde731a000ec723e7224c8aed4ffdedc9751f5599fe0a859c5c65d0c5d94891dR249
    # changed some of the scatter logic to looping + stacking due to spooky threaded cuda errors, need to CUDA_NONBLOCKING=1 to fix

    # typical decoding
    scores = logits
    mass = top_p if top_p > 0.0 else 1.0
    # calculate entropy
    log_p = torch.nn.functional.log_softmax(scores, dim=-1)
    p = torch.exp(log_p)
    ent = -(p * log_p).sum(-1, keepdim=True)
    # shift and sort
    # abs(I() - H())
    # I() is -log(p()) from eq 5
    # so overall we see -log(p()) - ent
    # orig code was ((-ent) - log_p) 
    shifted_scores = torch.abs(-log_p - ent)

    # possible to calculate the scores over k steps? ala classifier free guidance / CLIP guides?

    # most typical (0) to least typical (high abs value)
    _, sorted_indices = torch.sort(shifted_scores, descending=False, stable=True)
    top_k = min(top_k, scores.size(-1) - 1)  # safety check that top k is not too large
    # this semi-butchers some of the core arguments of the paper, but top k can be good
    # think of this as doing typical decoding / reordering based on the top k by prob
    # top k by typicality seems to be kinda weird for music?
    #if top_k > 0:
    #    topkval = torch.topk(scores, top_k)[0][..., -1, None]
    #    indices_to_remove = scores < topkval
    #    scores[indices_to_remove] = filter_value
    if top_k > 0:
        topkval = torch.topk(torch.max(shifted_scores) - shifted_scores, top_k)[0][..., -1, None]
        indices_to_remove = (torch.max(shifted_scores) - shifted_scores) < topkval
        scores[indices_to_remove] = filter_value
   
    sorted_scores = scores.gather(-1, sorted_indices)
    cumulative_probs = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens once cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > mass
    sorted_indices_to_remove = sorted_indices_to_remove.long()
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_indices = torch.tensor(sorted_indices.cpu().data.numpy())
    shp = scores.shape
    # not great cuda errors on gather calls here, rewrote to a "slow" version
    scores_red = scores.reshape((-1, shp[-1]))
    sorted_indices_red = sorted_indices.reshape((-1, shp[-1]))
    sorted_indices_to_remove_red = sorted_indices_to_remove.reshape((-1, shp[-1]))
    for i in range(shp[0]):
        scores_red[i][sorted_indices_red[i]] = scores_red[i][sorted_indices_red[i]] * (1. - sorted_indices_to_remove_red[i]) + sorted_indices_to_remove_red[i] * filter_value
    scores = scores_red.reshape(shp)
    return scores
'''


def make_batch(batch, random_state, target_quadrant="A"):
    shp = batch.shape
    # target quadrant A = 0:14, 0:14
    # target quadrant B = 0:14, 14:28
    # target quadrant C = 14:28, 0:14
    # target quadrant D = 14:28, 14:28

    # flatten from B 1 28 28 -> B 1 14, 14 -> B 14*14

    # these are the main entries, and the targets
    bsize = shp[0]
    batch_A = batch[:, :, 0:8, 0:8].reshape((bsize, -1))
    batch_B = batch[:, :, 0:8, 8:16].reshape((bsize, -1))
    batch_C = batch[:, :, 8:16, 0:8].reshape((bsize, -1))
    batch_D = batch[:, :, 8:16, 8:16].reshape((bsize, -1))
    if target_quadrant == "A":
        batch_complement = np.concatenate((batch_B, batch_C, batch_D), axis=-1)
        batch = np.concatenate((batch_complement, batch_A), axis=-1)
    elif target_quadrant == "B":
        batch_complement = np.concatenate((batch_A, batch_C, batch_D), axis=-1)
        batch = np.concatenate((batch_complement, batch_B), axis=-1)
    elif target_quadrant == "C":
        batch_complement = np.concatenate((batch_A, batch_B, batch_D), axis=-1)
        batch = np.concatenate((batch_complement, batch_C), axis=-1)
    elif target_quadrant == "D":
        batch_complement = np.concatenate((batch_A, batch_B, batch_C), axis=-1)
        batch = np.concatenate((batch_complement, batch_D), axis=-1)
    else:
        raise ValueError("Unknown target_quadrant specified for 'make_batch'")

    # batch is now B T format, with 3 context at front and target entry at the back

    # generate corresponding idx, we assume all entries "fill" measure, no 0 padding
    assert sequence_length == batch.shape[1]
    batch_idx = 0. * batch + np.arange(sequence_length)[None]
    # batch now has correct shape overall
    # swap to T, B format
    batch = batch.transpose(1, 0)
    batch_idx = batch_idx.transpose(1, 0).astype("int32")
    # idx has trailing 1
    batch_idx = batch_idx[..., None]
    # was 0 min, now 1 min (0 for padding in future datasets)
    batch = batch + 1
    # rewrite the batch for SUNMASK
    targets = copy.deepcopy(batch[-latent_length:])
    # sub 1 so targets are 0 : n_classes again
    targets = targets - 1
    return batch, batch_idx, targets

"""
tmp_random = np.random.RandomState(2122)
(data, data_matches, batch_idx) = next(train_itr)
batch, batch_idx, targets = make_batch(data, tmp_random, target_quadrant="A")
"""

gumbel_sampling_random_state = np.random.RandomState(3434)
corruption_sampling_random_state = np.random.RandomState(1122)

# speed this up with torch generator?
def gumbel_sample(logits, temperature=1.):
    noise = gumbel_sampling_random_state.uniform(1E-5, 1. - 1E-5, logits.shape)
    torch_noise = torch.tensor(noise).contiguous().to(device)

    #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))
    # max indices
    # no keepdim here
    maxes = torch.argmax(logits / float(temperature) - torch.log(-torch.log(torch_noise)), axis=-1)
    return maxes

# same here - torch generator speed it up?
def get_random_pitches(shape, vocab_size, low=0):
    # add 1 due to batch offset reserving 0 for length masking
    r = corruption_sampling_random_state.randint(low=0, high=vocab_size, size=shape)
    random_pitch = torch.tensor(copy.deepcopy(r)).type(torch.LongTensor).to(device)
    return random_pitch

def corrupt_pitch_mask(batch, mask, vocab_size):
    random_pitches = get_random_pitches(batch.shape, vocab_size)
    #corrupted = (1 - mask[..., None]) * random_pitches + (1 * mask[..., None]) * batch
    corrupted = (1 - mask) * random_pitches + (1 * mask) * batch
    return corrupted

# SUNDAE https://arxiv.org/pdf/2112.06749.pdf
def build_logits_fn(vocab_size, n_unrolled_steps, enable_sampling):
    def logits_fn(input_batch, input_batch_idx, input_mask, input_mem_mask):
        def fn(batch, batch_idx, mask, mem_mask):
            logits = model(batch, batch_idx, mask, mem_mask)
            return logits

        def unroll_fn(batch, batch_idx, mask, mem_mask):
            # only corrupt query ones - this will break for uneven seq lengths!
            # +1 offset of batch to preserve 0 as a special token
            samples = corrupt_pitch_mask(batch[-latent_length:], mask, vocab_size)
            samples = torch.concat([batch[:-latent_length], samples], axis=0)

            mem_samples = corrupt_pitch_mask(batch[:-latent_length], mem_mask[:-latent_length], vocab_size)
            samples = torch.concat([mem_samples, samples[-latent_length:]], axis=0)
            all_logits = []
            for _ in range(n_unrolled_steps):
                #ee = samples.cpu().data.numpy()
                #print(ee.min())
                #print(ee.max())
                logits = fn(samples, batch_idx, mask, mem_mask)
                samples = gumbel_sample(logits).detach()
                # sanity check to avoid issues with stacked outputs
                assert samples.shape[1] == batch.shape[1]
                # for the SUNDAE piece
                samples = samples[:, :batch.shape[1]]
                # add 1 to account for batch offset
                samples = torch.concat([mem_samples, samples + 1], axis=0)
                all_logits += [logits[None]]
            final_logits = torch.cat(all_logits, dim=0)
            return final_logits

        if enable_sampling:
            return fn(input_batch, input_batch_idx, input_mask, input_mem_mask)
        else:
            return unroll_fn(input_batch, input_batch_idx, input_mask, input_mem_mask)
    return logits_fn

def build_loss_fn(vocab_size, n_unrolled_steps=4):
    logits_fn = build_logits_fn(vocab_size, n_unrolled_steps, enable_sampling=False)

    def local_loss_fn(batch, batch_idx, mask, mem_mask, targets):
        # repeated targets are now n_unrolled_steps
        # batch is T B F
        batch_shp = batch.shape
        repeated_targets = torch.cat([targets[..., None]] * n_unrolled_steps, dim=1)
        # T N 1 -> N T 1
        repeated_targets = repeated_targets.permute(1, 0, 2)
        assert repeated_targets.shape[-1] == 1
        # N T 1 -> N T P

        repeated_targets = F.one_hot(repeated_targets[..., 0].long(), num_classes=vocab_size)
        #t = torch.argmax(repeated_targets, axis=-1)
        #for i in range(t.shape[0]):
        #    print([ind_to_vocab[int(e)] for e in t[i].cpu().data.numpy()])
        #print(mask)

        logits = logits_fn(batch, batch_idx, mask, mem_mask)
        # S, T, N, P -> S, N, T, P
        logits = logits.permute(0, 2, 1, 3)
        out = logits.reshape(n_unrolled_steps * logits.shape[1], logits.shape[2], logits.shape[3])
        logits = out
        # N, T, P
        #? trouble
        raw_loss = -1. * (nn.functional.log_softmax(logits, dim=-1) * repeated_targets)
        # mask is currently T, N
        # change to N, T, 1, then stack for masking
        # only keep loss over positions which were dropped, no freebies here
        raw_masked_loss = raw_loss * torch.cat([(1. - mask.permute(1, 0)[..., None])] * n_unrolled_steps, dim=0)
        raw_unmasked_loss = raw_loss * torch.cat([(mask.permute(1, 0)[..., None])] * n_unrolled_steps, dim=0)
        reduced_mask_active = torch.cat([1. / ((1. - mask).sum(dim=0) + 1)] * n_unrolled_steps, dim=0)[..., None, None]

        #inactive = (1 - mask).sum(dim=0) + 1
        #active = mask.sum(dim=0) + 1
        #raw_comb_loss = (inactive / (inactive + active))[None] * raw_masked_loss + (active / (inactive + active))[None] * (raw_masked_loss + raw_unmasked_loss)
        raw_comb_loss = raw_masked_loss

        # Active mask sums up the amount that were inactive in time
        # downweighting more if more were not dropped out
        reduced_loss = (reduced_mask_active * raw_comb_loss.view(n_unrolled_steps * batch_shp[1], latent_length, vocab_size)).sum(dim=-1)
        loss = torch.mean(reduced_loss, dim=1).mean()
        # upweight by average actives in T since the overall 
        # average penalty for mask weight reduction goes up the longer the sequence is?
        loss = np.sqrt(latent_length) * loss
        return loss
    return local_loss_fn

# harmonize a melody
def torch_diffuse_perceiversunmask(batch, batch_idx, C, model,
                                   vocabulary_size,
                                   internal_batch_size=2,
                                   keep_mask=None,
                                   n_steps=sequence_length,
                                   n_reps_per_mask=1,
                                   n_reps_final_mask_dwell=0,
                                   sundae_keep_prob=0.33,
                                   decay_schedule="cosine",
                                   initial_corrupt=True,
                                   intermediate_corrupt=False,
                                   frozen_mask=False,
                                   force_mem_mask=False,
                                   use_evener=False,
                                   top_k=0, top_p=0.0,
                                   swap_at_eta=False,
                                   use_typical_sampling=False,
                                   temperature=1.0, o_nade_eta=3./4, seed=12,
                                   return_intermediates=False,
                                   verbose=True):
    #lcl_seed_track = copy.deepcopy(batch)
    #if len(lcl_seed_track.shape) == 2:
    #    lcl_seed_track = np.concatenate([lcl_seed_track[None] for _ in range(batch_size)])
    #    C = np.concatenate([C[None] for _ in range(batch_size)])
    try:
        C = C.astype(np.int32)
    except:
        C = C.long().cpu().data.numpy().astype(np.int32)
    # latent_length, B
    assert len(C.shape) == 2
    assert C.shape[1] == batch.shape[1]

    batch = batch[:, :internal_batch_size]
    batch_idx = batch_idx[:, :internal_batch_size]
    C = C[:, :internal_batch_size]
    # WARNING:sketch city
    # start as ignore, or start as keep?
    C_mem = 0 * np.copy(batch)# + 1
    if force_mem_mask == True:
        C_mem += 1

    x = torch.tensor(batch).type(torch.FloatTensor).to(device)
    x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
    C = torch.tensor(C).long().to(device)
    C_mem = torch.tensor(C_mem).long().to(device)

    model.eval()
    rs = np.random.RandomState(seed)
    trsg = torch.Generator(device=device)
    trsg.manual_seed(seed)
    def lcl_gumbel_sample(logits):
        #noise = rs.uniform(1E-5, 1. - 1E-5, logits.shape)
        #torch_noise = torch.tensor(noise).contiguous().to(device)
        torch_noise = torch.rand(logits.shape, generator=trsg, device=device) * ((1 - 1E-5) - 1E-5) + 1E-5

        #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))

        # max indices
        #maxes = torch.argmax(logits / float(temperature) - torch.log(-torch.log(torch_noise)), axis=-1, keepdim=True)
        maxes = torch.argmax(logits - torch.log(-torch.log(torch_noise)), axis=-1)
        return maxes
        #one_hot = 0. * logits
        #one_hot.scatter_(-1, maxes, 1)
        #return one_hot

    def lcl_get_random_pitches(shape, vocab_size):
        random_pitch = torch.randint(low=0, high=vocab_size, size=shape, device=device, generator=trsg)
        return random_pitch

    with torch.no_grad():
        if keep_mask is not None:
            keep_C = torch.tensor(keep_mask).long().to(device)

        C2 = torch.clone(C)#.copy()
        C2_mem = torch.clone(C_mem)
        #num_steps = int(2*I*T)
        alpha_max = .999
        alpha_min = .001
        eta = o_nade_eta

        """
        for i in range(num_steps):
            p = np.maximum(alpha_min, alpha_max - i*(alpha_max-alpha_min)/(eta*num_steps))
            sampled_binaries = rs.choice(2, size = C.shape, p=[p, 1-p])
            C2 += sampled_binaries
            C2[C==1] = 1
            x_cache = x
            x = model.pred(x, C2, seed=rs.randint(100000))
            x[C2==1] = x_cache[C2==1]
            C2 = C.copy()
        """

        x_cache = torch.clone(x)

        if initial_corrupt:
            x_sub = lcl_get_random_pitches(x[-latent_length:].shape, vocabulary_size).float()
            # add 1 since 0 is protected value for masking
            x[-latent_length:] = x_sub + 1
            x[-latent_length:][C2 == 1] = x_cache[-latent_length:][C2 == 1]
            if keep_mask is not None:
                x[-latent_length:][keep_C == 1] = x_cache[keep_C==1]
        # x is now corrupted in the portion corresponding to the query
        n_steps = max(1, int(n_steps))
        if sundae_keep_prob == "triangular":
            sundae_keep_tokens_per_step = [2 * C.shape[0] * min((t + 1) / float(n_steps), 1 - (t + 1) / float(n_steps))
                                           for t in range(int(n_steps))] + [1.0 * C.shape[0] for t in range(int(n_reps_final_mask_dwell))]
        else:
            sundae_keep_tokens_per_step = [sundae_keep_prob * C.shape[0]
                                          for t in range(int(n_steps))] + [1.0 * C.shape[0] for t in range(int(n_reps_final_mask_dwell))]

        has_been_kept = 1. + 0. * C
        has_been_kept_torch = torch.tensor(has_been_kept).to(device)

        # might need to renormalize the kept matrix at some point...
        sampled_binaries = None
        all_x = []
        all_C = []
        for n in range(int(n_steps + n_reps_final_mask_dwell)):
            try:
                step_top_k = top_k[0]
                min_k = min(top_k)
                max_k = max(top_k)
                step_frac_k = (max_k - min_k) / float(n_steps)
                if min_k == top_k[1]:
                    step_top_k = max(1, round(step_frac_k * (n_steps - n) + min_k))
                else:
                    step_top_k = max(1, round(step_frac_k * n + min_k))
            except:
                step_top_k = top_k

            try:
                step_top_p = top_p[0]
                min_p = min(top_p)
                max_p = max(top_p)
                step_frac_p = (max_p - min_p) / float(n_steps)
                if min_p == top_p[1]:
                    step_top_p = min(1.0, max(1E-6, step_frac_p * (n_steps - n) + min_p))
                else:
                    step_top_p = min(1.0, max(1E-6, step_frac_p * n + min_p))
            except:
                step_top_p = top_p
            # todo: unify these   
            try:
                step_temperature = temperature[0]
                min_temp = min(temperature)
                max_temp = max(temperature)
                step_frac_temp = (max_temp - min_temp) / float(n_steps)
                if min_temp == temperature[1]:
                    step_temperature = min(1.0, max(1E-8, step_frac_temp * (n_steps - n) + min_temp))
                else:
                    step_temperature = min(1.0, max(1E-8, step_frac_temp * n + min_temp))
            except:
                step_temperature = temperature
            k = int(sundae_keep_tokens_per_step[n])
            if k == 0:
                # skip zero keep scheduled steps to speed things up
                # do it this way because very long schedules need small k values
                # which necessarily causes 0 to be more frequent
                continue
            # do n_unroll_steps of resampling, randomly sampling masks during the procedure
            fwd_step = n
            if n_reps_per_mask > 1:
                # roll mask forward 
                fwd_step = int(fwd_step + n_reps_per_mask)
            # max value for p
            # based on keras cos anneal lr
            cos_p = 0.5 * 1.0 * (
            1 + np.cos(
                np.pi * min(fwd_step / eta * float(n_steps), 1.0))
            )
            cos_p = np.minimum(alpha_max, np.maximum(alpha_min, cos_p))
            p = np.maximum(alpha_min, alpha_max - fwd_step * (alpha_max-alpha_min)/(eta*int(n_steps)))
            if decay_schedule == "cosine":
                p = cos_p
            #if intermediate_corrupt:
            if not frozen_mask:
                if n % n_reps_per_mask == 0:
                    #sampled_binaries = rs.choice(2, size = C.shape, p=[p, 1-p])
                    sampled_binaries = torch.bernoulli(1. - (0 * C + p), generator=trsg).long()
                    C2 += sampled_binaries

                    # look at memory earlier than self?
                    sampled_mem_binaries = torch.bernoulli(1. - (0 * C_mem + p), generator=trsg).long()
                    C2_mem += sampled_mem_binaries

                if n > n_steps:
                    # set final mask to all ones
                    C2[:] = 1
            # modify mask
            # this way the model trusts different memory variables all the time
            # this should be close to the *best* sampling setup
            C2[C==1] = 1
            #C2_mem[:] = 0
            #C2_mem[:] = 1
            C2_mem[-latent_length:] = torch.clone(C2)[:]
            #C2_mem = C2_mem * 0 + 1
            #x_cache = x
            #if initial_corrupt:
            #    x = lcl_get_random_pitches(x.shape, P)
            #    x[C2==1] = x_cache[C2==1]
            #x = model.pred(x, C2)#, temperature=temperature)

            #x_e = torch.clone(x) # torch.tensor(x).float().to(device)
            #C2_e = torch.clone(C2) # torch.tensor(C2).float().to(device)
            # passing true will noise things
            logits_x = model(x, x_idx, C2, C2_mem)

            # dont predict just logits anymore
            # top k top p gumbel
            if swap_at_eta:
                swap_flag = n < (eta * int(n_steps))
            else:
                swap_flag = use_typical_sampling
            if use_typical_sampling and swap_flag:
                logits_x = logits_x / float(step_temperature)
                filtered_logits_x = typical_top_k_filtering(logits_x, top_k=step_top_k, top_p=step_top_p)
            else:
                logits_x = logits_x / float(step_temperature)
                filtered_logits_x = top_k_top_p_filtering(logits_x, top_k=step_top_k, top_p=step_top_p)
            x_new = lcl_gumbel_sample(filtered_logits_x).float()

            # the even-er
            p = has_been_kept_torch[:, :] / torch.sum(has_been_kept_torch[:, :], axis=0, keepdims=True)
            r_p = 1. - p
            r_p = r_p / torch.sum(r_p, axis=0, keepdims=True)

            if k > 0:
                shp = r_p.shape
                assert len(shp) == 2
                # turn it to B, T for torch.multinomial
                r_p = r_p.permute(1, 0)
                if use_evener:
                    keep_inds_torch = torch.multinomial(r_p, num_samples=k, replacement=False, generator=trsg)
                else:
                    keep_inds_torch = torch.multinomial(0. * r_p + 1. / float(shp[0]), num_samples=k, replacement=False, generator=trsg)

                # back to T, B
                keep_inds_torch = keep_inds_torch.permute(1, 0)

                # use scatter logic 
                for _ii in range(x.shape[1]):
                    # add 1 so that the sampled targets and batch match (keep 0 reserved in input)
                    x[-latent_length:, :][keep_inds_torch[:, _ii], _ii] = x_new[keep_inds_torch[:, _ii], _ii] + 1
                    has_been_kept_torch[keep_inds_torch[:, _ii], _ii] += 1
            else:
                pass

            x[-latent_length:][C==1] = x_cache[-latent_length:][C==1]
            if keep_mask is not None:
                x[-latent_length:][keep_C==1] = x_cache[-latent_length:][keep_C==1]

            if verbose:
                print("step {}".format(n))
            if return_intermediates:
                all_x.append(x.cpu().data.numpy())
                all_C.append(C2.cpu().data.numpy())
            C2 = torch.clone(C)
        if return_intermediates:
            return x, all_x, all_C
        return x

u_loss_fn = build_loss_fn(n_classes, n_unrolled_steps=n_unrolled_steps)
if True:
    _offset = 3
    _seed = 1122 + _offset
    seed_everything(_seed)
    data_random_state = np.random.RandomState(5544 + _offset)
    model.reset_generator()

    temperature_to_test = .1
    # temp .1 steps 10 n reps final mask to test 1
    # keep to test .2
    # topk 0 
    # topp 0
    # [0, 1, 3]
    # typical False gives "creative" digits
    steps_to_test = 10 #.01 * int(latent_length)
    n_reps_per_mask_to_test = 1
    n_reps_final_mask_dwell_to_test = 0
    #keep_to_test = .01 # .01 k = 1 makes weird symbols
    keep_to_test = 1.0 #1.0 #1.0 #- .005
    # top k 1 will cause it to make novel digits
    top_k_to_test = 0
    top_p_to_test = 0.0
    seed_offset_to_test = 7951
    typical_sampler_to_test = False
    evener_to_test = False

    n_steps = steps_to_test
    n_reps_per_mask = n_reps_per_mask_to_test
    n_reps_final_mask_dwell = n_reps_final_mask_dwell_to_test
    sundae_keep_prob = keep_to_test
    top_k = top_k_to_test
    top_p = top_p_to_test
    use_evener = evener_to_test
    use_typical_sampling = typical_sampler_to_test
    temperature = temperature_to_test
    this_seed = _seed
    return_intermediates = False
    decay_schedule = "linear"

    tmp_random = np.random.RandomState(2122)
    (data, batch_idx) = next(test_itr)
    if args.data_quad is not None:
        batch, batch_idx, targets = make_batch(data, tmp_random, target_quadrant=args.data_quad)
    else:
        batch, batch_idx, targets = make_batch(data, tmp_random, target_quadrant=args.quad)
    gt_batch = np.copy(batch)

    # delete the gt just in case
    rand_tgt_rng = np.random.RandomState(4142)
    batch[-latent_length:] = batch[-latent_length:] * 0 + rand_tgt_rng.randint(1, n_classes, size=(latent_length, batch.shape[1]))

    # mask is only latent_length long
    C_prob = data_random_state.rand(batch.shape[1])
    C_mask_base = data_random_state.rand(batch[-latent_length:].shape[0], batch.shape[1])
    C = 1 * (C_mask_base < C_prob[None, :])
    C = 0. * C # 0 means change in the sampling mask
    C = C.astype(np.int32)

    x = torch.tensor(batch).type(torch.FloatTensor).to(device)
    x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
    targets = torch.tensor(targets).type(torch.FloatTensor).to(device)
    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)

    ret = torch_diffuse_perceiversunmask(x, x_idx, C2,
                                         model,
                                         vocabulary_size=n_classes,
                                         n_steps=n_steps,
                                         n_reps_per_mask=n_reps_per_mask,
                                         n_reps_final_mask_dwell=n_reps_final_mask_dwell,
                                         sundae_keep_prob=sundae_keep_prob,
                                         top_k=top_k,
                                         top_p=top_p,
                                         use_evener=use_evener,
                                         use_typical_sampling=use_typical_sampling,
                                         decay_schedule=decay_schedule,
                                         temperature=temperature,
                                         seed=this_seed,
                                         return_intermediates=return_intermediates,
                                         verbose=False)
    if not return_intermediates:
        raw_pred = ret
    else:
        raw_pred, intermediate_x, intermediate_C = ret

    # need to undo the quadrants
    out_pred = raw_pred - 1
    quadrants = []

    for i in range(3):
        this_quad = out_pred[i * (8*8):(i + 1) * 8*8]
        quadrants.append(this_quad)
    pred_quad = out_pred[-(8*8):]
    ind = 1

    if args.ground_truth is not None:
        pred_quad = targets[:, :pred_quad.shape[1]]

    if args.quad == "A":
        reordered_pred_upper = np.concatenate([pred_quad[:, ind].reshape((8, 8)), quadrants[0][:, ind].reshape((8, 8))], axis=1)
        reordered_pred_lower = np.concatenate([quadrants[1][:, ind].reshape((8, 8)), quadrants[2][:, ind].reshape((8, 8))], axis=1)
        reordered_pred = np.concatenate([reordered_pred_upper, reordered_pred_lower], axis=0)
    elif args.quad == "B":
        reordered_pred_upper = np.concatenate([quadrants[0][:, ind].reshape((8, 8)), pred_quad[:, ind].reshape((8, 8))], axis=1)
        reordered_pred_lower = np.concatenate([quadrants[1][:, ind].reshape((8, 8)), quadrants[2][:, ind].reshape((8, 8))], axis=1)
        reordered_pred = np.concatenate([reordered_pred_upper, reordered_pred_lower], axis=0)
    elif args.quad == "C":
        reordered_pred_upper = np.concatenate([quadrants[0][:, ind].reshape((8, 8)), quadrants[1][:, ind].reshape((8, 8))], axis=1)
        reordered_pred_lower = np.concatenate([pred_quad[:, ind].reshape((8, 8)), quadrants[2][:, ind].reshape((8, 8))], axis=1)
        reordered_pred = np.concatenate([reordered_pred_upper, reordered_pred_lower], axis=0)
    elif args.quad == "D":
        reordered_pred_upper = np.concatenate([quadrants[0][:, ind].reshape((8, 8)), quadrants[1][:, ind].reshape((8, 8))], axis=1)
        reordered_pred_lower = np.concatenate([quadrants[2][:, ind].reshape((8, 8)), pred_quad[:, ind].reshape((8, 8))], axis=1)
        reordered_pred = np.concatenate([reordered_pred_upper, reordered_pred_lower], axis=0)
    else:
        raise ValueError("args.quad not found {}".format(args.quad))
    vq_model_save_path = "conv_acn_vq_models"
    FPATH_VAE = os.path.join(vq_model_save_path, 'conv_vae.pth')
    FPATH_PRIOR = os.path.join(vq_model_save_path, 'prior.pth')
    code_len = 48
    conv_batch_size = 128
    model_hidden_size = 256
    prior_hidden_size = 512
    n_neighbors = 5
    dataset_len = 60000
    testset_len = 10000
    conv_model = ConvACNVQVAE(model_hidden_size, code_len, conv_batch_size)
    prior = PriorNetwork(prior_hidden_size, code_len, dataset_len, k=n_neighbors, code_multiple=4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device {}".format(device))
    conv_model = conv_model.to(device)
    prior = prior.to(device)
    conv_model.load_state_dict(torch.load(FPATH_VAE, map_location=device))
    prior.load_state_dict(torch.load(FPATH_PRIOR, map_location=device))
    conv_model.eval()
    prior.eval()

    code_tmp = conv_model.vq_indices_to_codes(torch.tensor(reordered_pred[None]).long().to(device))
    out_code = conv_model.decode(code_tmp).detach().cpu().data.numpy()
    def sigmoid_np(x):
        return np.exp(-np.logaddexp(0, -x))
    out_im = sigmoid_np(out_code)
    plt.imshow(out_im[0, 0])
    plt.savefig("tmp.png")
    plt.close()
    from IPython import embed; embed(); raise ValueError()
    # encode full batch, save and store
    context0_pred = (out_pred).cpu().data.numpy().astype("int32")[0 * latent_length:1 * latent_length].transpose(1, 0).reshape(-1, 16, 16)
    context1_pred = (out_pred).cpu().data.numpy().astype("int32")[1 * latent_length:2 * latent_length].transpose(1, 0).reshape(-1, 16, 16)
    context2_pred = (out_pred).cpu().data.numpy().astype("int32")[2 * latent_length:3 * latent_length].transpose(1, 0).reshape(-1, 16, 16)
    final_pred = (out_pred).cpu().data.numpy().astype("int32")[-latent_length:].transpose(1, 0).reshape(-1, 16, 16)
    ground_truth_target = targets.cpu().data.numpy().astype("int32").transpose(1, 0).reshape(-1, 16, 16)
    np.savez("sunmask_vq_pred.npz", context0_pred=context0_pred, context1_pred=context1_pred, context2_pred=context2_pred,
                                    final_pred=final_pred, ground_truth_target=ground_truth_target)
