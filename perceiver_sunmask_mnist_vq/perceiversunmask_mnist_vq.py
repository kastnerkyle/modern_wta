from perceiversunmask_mnist_vq_models import PerceiverSUNMASK, clipping_grad_value_, RampOpt
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
    parser.add_argument('--task', default='train', choices=['train','sample'], help='Task to do.')
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

mnist_path = "../mnist.npz"
train_data_f = np.load(mnist_path)
train_data_np = train_data_f[train_data_f.files[2]].T
train_data_labels_np = train_data_f[train_data_f.files[3]].T

test_data_f = np.load(mnist_path)
test_data_np = test_data_f[test_data_f.files[0]].T
test_data_labels_np = test_data_f[test_data_f.files[1]].T

mnist_vq_path = "mnist_vq.npz"
d = np.load(mnist_vq_path)
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
        data_f = np.load(mnist_path)
        # subset to 50k only for the actual targets
        # this does allow some "valid" entries as conditional inputs
        data_np = data_f[data_f.files[2]].T
        label_np = data_f[data_f.files[3]].T
        data_vq_np = d["train_vq_indices"]
        data_match_indices_np = d["train_vq_match_indices"]
    elif subset_type == "valid":
        data_f = np.load(mnist_path)
        # need to reference full train set
        data_np = data_f[data_f.files[2]].T
        label_np = data_f[data_f.files[3]].T
        data_vq_np = d["train_vq_indices"]
        data_match_indices_np = d["train_vq_match_indices"]
    elif subset_type == "test":
        data_f = np.load(mnist_path)
        data_np = data_f[data_f.files[0]].T
        label_np = data_f[data_f.files[1]].T
        # no test vq indices used
        data_vq_np = 0 * d["train_vq_indices"]
        data_match_indices_np = d["test_vq_match_indices"]
    else:
        raise ValueError("Unknown subset_type {}".format(subset_type))

    if subset_type == "train":
        max_sz = 50000
    elif subset_type == "valid":
        max_sz = 10000
    elif subset_type == "test":
        max_sz = 10000

    random_state = np.random.RandomState(seed)
    while True:
        inds = np.arange(max_sz)
        if subset_type == "valid":
            inds = inds + 50000
        batch_inds = random_state.choice(inds, size=batch_size, replace=True)

        # B C H W (C is 1)
        batch = data_vq_np[batch_inds][:, None]
        # B N H W (N is neighbors)
        batch_matches = np.concatenate([data_vq_np[data_match_indices_np[el]][None] for el in batch_inds], axis=0)
        # return in a similar format as pytorch
        yield batch, batch_matches, batch_inds

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 12
# this is a multiple of 4 - so 16 * 4 -> 64 latent length is 1 measure
latent_length = int(16 * 16)
sequence_length = int(4 * latent_length)
n_classes = vq_classes = 512
n_neighbors = 5
n_neighbors_in_context = 3

n_unrolled_steps = 2

hidden_size = 380
self_inner_dim = 900
input_dropout_keep_prob = 1.0
cross_attend_dropout_keep_prob = 1.0
autoregression_dropout_keep_prob = 1.0
inner_dropout_keep_prob = 1.0
final_dropout_keep_prob = 1.0
n_layers = 16

model_save_path = "perceiversunmask_mnist_vq_models"
dataset_len = 60000
testset_len = 10000

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
args = parse_flags()

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


def make_batch(batch, batch_matches, random_state):
    shp = batch.shape
    # flatten from B 1 16 16 -> B 1 16*16

    # these are the main entries, and the targets
    bsize = shp[0]
    batch = batch.reshape(bsize, 1, -1)
    shp = batch_matches.shape
    # these are all possible neighbors - we will choose n_neighbors_in_context of them randomly, in random order
    batch_matches = batch_matches.reshape(bsize, shp[1], -1)

    n_inds = np.arange(n_neighbors)
    batch_inds = [random_state.choice(n_inds, size=n_neighbors_in_context, replace=False) for _ in range(bsize)]

    batch_matches_sub = np.concatenate([batch_matches[el, batch_inds[el]][None] for el in range(bsize)], axis=0)

    # now the batch matches are the right size, we can concat and flatten
    batch = np.concatenate((batch_matches_sub, batch), axis=1).reshape(bsize, -1)
    # batch is now B T format, with 3 context at front and target entry at the back

    # generate corresponding idx, we assume all entries "fill" measure, no 0 padding
    assert sequence_length == batch.shape[1]
    batch_idx = 0. * batch + np.arange(sequence_length)[None]
    # batch now has correct shape overall, and is interleaved
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
    def logits_fn(input_batch, input_batch_idx, input_mask):
        def fn(batch, batch_idx, mask):
            logits, mask = model(batch, batch_idx, mask)
            return logits

        def unroll_fn(batch, batch_idx, mask):
            # only corrupt query ones - this will break for uneven seq lengths!
            # +1 offset of batch to preserve 0 as a special token
            samples = corrupt_pitch_mask(batch[-latent_length:], mask, vocab_size)
            samples = torch.concat([batch[:-latent_length], samples + 1], axis=0)
            all_logits = []
            for _ in range(n_unrolled_steps):
                #ee = samples.cpu().data.numpy()
                #print(ee.min())
                #print(ee.max())
                logits = fn(samples, batch_idx, mask)
                samples = gumbel_sample(logits).detach()
                # sanity check to avoid issues with stacked outputs
                assert samples.shape[1] == batch.shape[1]
                # for the SUNDAE piece
                samples = samples[:, :batch.shape[1]]
                # add 1 to account for batch offset
                samples = torch.concat([batch[:-latent_length], samples + 1], axis=0)
                all_logits += [logits[None]]
            final_logits = torch.cat(all_logits, dim=0)
            return final_logits

        if enable_sampling:
            return fn(input_batch, input_batch_idx, input_mask)
        else:
            return unroll_fn(input_batch, input_batch_idx, input_mask)
    return logits_fn

def build_loss_fn(vocab_size, n_unrolled_steps=4):
    logits_fn = build_logits_fn(vocab_size, n_unrolled_steps, enable_sampling=False)

    def local_loss_fn(batch, batch_idx, mask, targets):
        # repeated targets are now n_unrolled_steps
        repeated_targets = torch.cat([targets[..., None]] * n_unrolled_steps, dim=1)
        # T N 1 -> N T 1
        repeated_targets = repeated_targets.permute(1, 0, 2)
        assert repeated_targets.shape[-1] == 1
        bsize = batch.shape[1]
        # N T 1 -> N T P

        repeated_targets = F.one_hot(repeated_targets[..., 0].long(), num_classes=vocab_size)
        #t = torch.argmax(repeated_targets, axis=-1)
        #for i in range(t.shape[0]):
        #    print([ind_to_vocab[int(e)] for e in t[i].cpu().data.numpy()])
        #print(mask)

        logits = logits_fn(batch, batch_idx, mask)
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
        reduced_loss = (reduced_mask_active * raw_comb_loss.view(n_unrolled_steps * bsize, latent_length, vocab_size)).sum(dim=-1)
        loss = torch.mean(reduced_loss, dim=1).mean()
        # upweight by average actives in T since the overall 
        # average penalty for mask weight reduction goes up the longer the sequence is?
        loss = np.sqrt(latent_length) * loss
        return loss
    return local_loss_fn

u_loss_fn = build_loss_fn(n_classes, n_unrolled_steps=n_unrolled_steps)
if args.task == 'sample':
    # harmonize a melody
    def torch_diffuse_perceiversunmask(batch,
                                       batch_idx,
                                       C,
                                       model,
                                       internal_batch_size=2,
                                       keep_mask=None,
                                       n_steps=latent_length,
                                       n_reps_per_mask=1,
                                       n_reps_final_mask_dwell=0,
                                       sundae_keep_prob=0.33,
                                       initial_corrupt=True,
                                       intermediate_corrupt=False,
                                       frozen_mask=False,
                                       use_evener=False,
                                       top_k=0, top_p=0.0,
                                       swap_at_eta=False,
                                       use_typical_sampling=False,
                                       temperature=1.0, o_nade_eta=3./4, seed=12, verbose=True):
        """
        return values will be in "batch space" which is +1 from targets (0 is reserved for masking in the perceiver impl)
        so for target results subtract 1 from returned values!
        """
        batch = batch[:, :internal_batch_size]
        batch_idx = batch_idx[:, :internal_batch_size]
        C = C[:, :internal_batch_size]

        x = torch.tensor(batch).type(torch.FloatTensor).to(device)
        x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
        # due to sampling interface we need to flatten the C mask to SATBSATBSATB order...
        C = torch.tensor(C).long().to(device)
        model_dir = str(os.sep).join(FPATH_PERCEIVER.split(os.sep)[:-1])
        model_path = FPATH_PERCEIVER.split(os.sep)[-1].replace("{}", "")
        model_ext = "." + model_path.split(".")[-1]
        model_name = "".join(model_path.split(".")[:-1])
        is_saved_model = [f for f in os.listdir(model_dir) if model_name in f and f.endswith(model_ext)]
        # model with highest step count will be first in list
        is_saved_model = sorted(is_saved_model, key=lambda x: int(x.split("_")[-1].split(".")[0]))[::-1]
        load_path = model_dir + os.sep + is_saved_model[0]
        model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))
        print("loaded model from {}".format(load_path))

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
                x_sub = lcl_get_random_pitches(x[-latent_length:].shape, n_classes).float()
                # add 1 since 0 is protected value for masking
                x[-latent_length:] = x_sub + 1
                x[-latent_length:][C2==1] = x_cache[-latent_length:][C2 == 1]
                if keep_mask is not None:
                    x[-latent_length:][keep_C==1] = x_cache[keep_C==1]
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
            for n in range(int(n_steps + n_reps_final_mask_dwell)):
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
                p = np.maximum(alpha_min, alpha_max - fwd_step*(alpha_max-alpha_min)/(eta*int(n_steps)))
                #if intermediate_corrupt:
                if not frozen_mask:
                    if n % n_reps_per_mask == 0:
                        #sampled_binaries = rs.choice(2, size = C.shape, p=[p, 1-p])
                        sampled_binaries = torch.bernoulli(1. - (0 * C + p), generator=trsg).long()
                        C2 += sampled_binaries
                    if n > n_steps:
                        # set final mask to all ones
                        C2[:] = 1

                # todo: always modify mask even if intermediate_corrupt is False
                # this way the model trusts different variables all the time
                # this should be close to the *best* sampling setup
                C2[C==1] = 1
                #x_cache = x
                #if initial_corrupt:
                #    x = lcl_get_random_pitches(x.shape, P)
                #    x[C2==1] = x_cache[C2==1]

                #x = model.pred(x, C2)#, temperature=temperature)

                #x_e = torch.clone(x) # torch.tensor(x).float().to(device)
                #C2_e = torch.clone(C2) # torch.tensor(C2).float().to(device)
                # passing true will noise things
                logits_x, masks = model(x, x_idx, C2)

                # dont predict just logits anymore
                # top k top p gumbel
                if swap_at_eta:
                    swap_flag = n < (eta * int(n_steps))
                else:
                    swap_flag = use_typical_sampling
                if use_typical_sampling and swap_flag:
                    logits_x = logits_x / float(temperature)
                    filtered_logits_x = typical_top_k_filtering(logits_x, top_k=top_k, top_p=top_p, temperature=float(temperature))
                else:
                    logits_x = logits_x / float(temperature)
                    filtered_logits_x = top_k_top_p_filtering(logits_x, top_k=top_k, top_p=top_p)
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
                    print("sundae_step {}".format(n))
                C2 = torch.clone(C)
            return x

    _seed = 1122
    seed_everything(_seed)
    data_random_state = np.random.RandomState(5544)
    model.reset_generator()

    temperature_to_test = .6
    steps_to_test = 10 * int(latent_length)
    n_reps_per_mask_to_test = 1
    n_reps_final_mask_dwell_to_test = 0
    keep_to_test = .33 #"triangular" #.33
    top_k_to_test = 3
    top_p_to_test = 0.0
    seed_offset_to_test = 7945
    typical_sampler_to_test = True
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

    # todo: add digit filter opts
    # ultimate test - rotate/flip?
    d = np.load("encoded_data.npz")

    """
    for i in range(50):
        (data, data_matches, batch_idx) = next(valid_itr)
    from IPython import embed; embed(); raise ValueError()
    """

    data = d["data_batches_quantized"]
    data_matches = d["neighbor_data_batches_quantized"]

    batch, batch_idx, targets = make_batch(data, data_matches, data_random_state)
    # delete the gt just in case
    batch[-latent_length:] = 1.

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

    this_seed = seed_offset_to_test
    raw_pred = torch_diffuse_perceiversunmask(x,
                                              x_idx,
                                              C2,
                                              model,
                                              internal_batch_size=data.shape[0],
                                              n_steps=n_steps,
                                              n_reps_per_mask=n_reps_per_mask,
                                              n_reps_final_mask_dwell=n_reps_final_mask_dwell,
                                              sundae_keep_prob=sundae_keep_prob,
                                              top_k=top_k,
                                              top_p=top_p,
                                              use_evener=use_evener,
                                              use_typical_sampling=use_typical_sampling,
                                              temperature=temperature,
                                              seed=this_seed,
                                              verbose=True)
    out_pred = raw_pred - 1
    # encode full batch, save and store
    context0_pred = (out_pred).cpu().data.numpy().astype("int32")[0 * latent_length:1 * latent_length].transpose(1, 0).reshape(-1, 16, 16)
    context1_pred = (out_pred).cpu().data.numpy().astype("int32")[1 * latent_length:2 * latent_length].transpose(1, 0).reshape(-1, 16, 16)
    context2_pred = (out_pred).cpu().data.numpy().astype("int32")[2 * latent_length:3 * latent_length].transpose(1, 0).reshape(-1, 16, 16)
    final_pred = (out_pred).cpu().data.numpy().astype("int32")[-latent_length:].transpose(1, 0).reshape(-1, 16, 16)
    ground_truth_target = targets.cpu().data.numpy().astype("int32").transpose(1, 0).reshape(-1, 16, 16)
    np.savez("sunmask_vq_pred.npz", context0_pred=context0_pred, context1_pred=context1_pred, context2_pred=context2_pred,
                                    final_pred=final_pred, ground_truth_target=ground_truth_target)

elif args.task == 'train':
    seed_everything(1234)
    params = list(model.parameters())

    def get_std_ramp_opt(model):
        return RampOpt(learning_rate, 1, ramp_til, decay_til,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.1, 0.999), eps=1E-6),
                       min_decay_learning_rate=min_learning_rate)

    optimizer = get_std_ramp_opt(model)

    model.train()
    data_random_state = np.random.RandomState(1177)

    losses = []
    valid_losses = []

    _last_time = time.time()
    _save_time = 0
    _last_save = 0
    _last_show = 0
    for _n_train_steps_taken in range(n_train_steps):
        (data, data_matches, batch_idx) = next(train_itr)
        batch, batch_idx, targets = make_batch(data, data_matches, data_random_state)
        # bring in the SUNMASK loss, fix for flat sequnces / no reshape

        # mask is only latent_length long
        C_prob = data_random_state.rand(batch.shape[1])
        C_mask_base = data_random_state.rand(batch[-latent_length:].shape[0], batch.shape[1])
        C = 1 * (C_mask_base < C_prob[None, :])
        C = (1. - C) # convert to 0 drop format
        C = C.astype(np.int32)

        x = torch.tensor(batch).type(torch.FloatTensor).to(device)
        x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
        targets = torch.tensor(targets).type(torch.FloatTensor).to(device)
        C2 = torch.tensor(C).type(torch.FloatTensor).to(device)

        loss = u_loss_fn(x, x_idx, C2, targets)
        losses.append(loss.item())
        loss.backward()
        clipping_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()

        if len(valid_losses) > 0:
            # carryover
            valid_losses.append(valid_losses[-1])

        if _n_train_steps_taken == 0 or (_n_train_steps_taken - _last_show) > show_every:
            model.eval()
            for _s in range(valid_steps_per):
                with torch.no_grad():
                    (data, data_matches, batch_idx) = next(valid_itr)
                    batch, batch_idx, targets = make_batch(data, data_matches, data_random_state)

                    # mask is only latent_length long
                    C_prob = data_random_state.rand(batch.shape[1])
                    C_mask_base = data_random_state.rand(batch[-latent_length:].shape[0], batch.shape[1])
                    C = 1 * (C_mask_base < C_prob[None, :])
                    C = (1. - C) # convert to 0 drop format
                    C = C.astype(np.int32)

                    x = torch.tensor(batch).type(torch.FloatTensor).to(device)
                    x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
                    targets = torch.tensor(targets).type(torch.FloatTensor).to(device)
                    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)

                    loss = u_loss_fn(x, x_idx, C2, targets)

                    valid_losses.append(loss.item())
                    # carryover
                    losses.append(losses[-1])
            optimizer.zero_grad()
            _new_time = time.time()
            _last_show = _n_train_steps_taken
            print('step {}'.format(_n_train_steps_taken))
            print('train loss avg (past 5k): ', np.mean(losses[-5000:]))
            print('valid loss avg (past 5k): ', np.mean(valid_losses[-5000:]))
            print('approx time (sec) per step (ignoring save time): ', ((_new_time - _last_time) - _save_time) / float(show_every))
            print('approx time (sec) per step (including save time): ', ((_new_time - _last_time) / float(show_every)))
            plt.plot(losses)
            plt.plot(valid_losses)
            plt.savefig("perceiversunmask_train_losses.png")
            plt.close('all')
            plt.plot(losses[-5000:])
            plt.plot(valid_losses[-5000:])
            plt.savefig("perceiversunmask_train_losses_recent.png")
            plt.close('all')
            _last_time = time.time()
            _save_time = 0
            model.train()

        if _last_save == 0 or (_n_train_steps_taken - _last_save) > save_every or _n_train_steps_taken == (n_train_steps - 1):
            _last_save = _n_train_steps_taken
            _save_start = time.time()
            torch.save(model.state_dict(), FPATH_PERCEIVER.format(_n_train_steps_taken + 1))
            np.savez(FPATH_LOSSES.format(_n_train_steps_taken + 1), losses=losses, valid_losses=valid_losses)
            _save_end = time.time()
            _save_time += (_save_end - _save_start)
            print('save time (sec): ', (_save_end - _save_start))
            model.train()
