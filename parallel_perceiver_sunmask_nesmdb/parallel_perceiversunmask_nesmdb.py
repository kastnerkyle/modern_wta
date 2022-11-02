# Find MIDIs
import glob
# Pick a random one
import random
import numpy as np
import pretty_midi
import os
import shelve
import time
from operator import itemgetter
from itertools import groupby
import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import os
import imageio
import copy
import psutil
import random
from collections import OrderedDict
import pretty_midi
import multiprocessing
from parallel_perceiversunmask_nesmdb_models import PerceiverSUNMASK, clipping_grad_value_, RampOpt

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
    parser.add_argument('--task', default='notes_target', choices=['notes_target','durs_target'], help='Task to do.')
    args = parser.parse_args()
    return args

# analyze_tokenized.py
pitch_n_classes = 267
dur_n_classes = 402

args = parse_flags()
if args.task not in ["notes_target", "durs_target"]:
    print(args)
    raise ValueError("Invalid task passes to launch args")

class IndexedBlob:
    # https://stackoverflow.com/questions/51181556/whats-the-fastest-way-to-save-load-a-large-collection-list-set-of-strings-in
    def __init__(self, filename):
        index_filename = filename + '.index'
        blob = np.memmap(filename, mode='r')

        try:
            # if there is an existing index
            indices = np.memmap(index_filename, dtype='>i8', mode='r')
        except FileNotFoundError:
            # else, create it
            indices, = np.where(blob == ord('\0'))
            # force dtype to predictable file
            indices = np.array(indices, dtype='>i8')
            with open(index_filename, 'wb') as f:
                # add a virtual newline
                np.array(-1, dtype='>i8').tofile(f)
                indices.tofile(f)
            # then reopen it as a file to reduce memory
            # (and also pick up that -1 we added)
            indices = np.memmap(index_filename, dtype='>i8', mode='r')

        self.blob = blob
        self.indices = indices

    def __getitem__(self, line):
        assert line >= 0

        lo = self.indices[line] + 1
        hi = self.indices[line + 1]

        return self.blob[lo:hi].tobytes().decode()

filesdir = "tokenized_nesmdb_song_str"
#all_files.extend(sorted([filesdir + os.sep + f for f in os.listdir(filesdir)]))
start_time = time.time()
print("running initial file count with dircnt to cache for efficiency")
print("if this is slow, kill the script, and run this command manually")
print("~/dircnt {}".format(filesdir))
os.system("~/dircnt {}".format(filesdir))
# do a system call to dircnt 
all_files = [filesdir + os.sep + f.name for f in os.scandir(filesdir)]
stop_time = time.time()
all_files = sorted(all_files)
print("time to read files", stop_time - start_time)

"""
# hack this to get a few to debug load
sz = 100
train_fpaths = [el for el in all_files if "_train_" in el][:sz]
valid_fpaths = [el for el in all_files if "_valid_" in el][:sz]
test_fpaths = [el for el in all_files if "_test_" in el][:sz]
all_files = train_fpaths + valid_fpaths + test_fpaths
"""

def _read_task(fpath):
    print("processing", fpath, flush=True)
    with open(fpath, "r") as file_handle:
        l = file_handle.read()
        tokenized_events = l.split("\n")
    return (fpath, tokenized_events)

def split_process_and_assert(tokenized_seq):
    no_notes = [el for el in tokenized_seq if "NOTE" not in el]
    no_notes_durs = [el for el in tokenized_seq if "DUR" not in el]
    # check em
    assert len(no_notes) == len(no_notes_durs)

    # turn to integer while we check them
    int_no_notes = []
    int_no_notes_durs = []
    for el1, el2 in zip(no_notes, no_notes_durs):
        # be sure all the wait times are identical
        if "WT" in el1:
            assert "WT" in el2
            assert el1 == el2
        if "WT" in el2:
            assert "WT" in el1
            assert el1 == el2
        int_no_notes_durs.append(pitch_class_map[el2.split("_")[-1]] if "WT" not in el1 else pitch_class_map["WT"])
        # no special for WT
        int_no_notes.append(dur_class_map[el1.split("_")[-1]])

    # now that we have 2 parallel sequences, convert all to integer repr...
    # then data loader will load from within these
    # pitch, durs
    return (int_no_notes_durs, int_no_notes)

pitch_class_map = {str(el): el for el in range(3, pitch_n_classes + 1)}
pitch_class_map["<s>"] = 0
pitch_class_map["</s>"] = 1
pitch_class_map["WT"] = 2

dur_class_map = {str(el): el for el in range(2, dur_n_classes + 1)}
dur_class_map["<s>"] = 0
dur_class_map["</s>"] = 1

memmap_paths = ["train_tok_pitch.memmap", "train_tok_dur.memmap",
                "valid_tok_pitch.memmap", "valid_tok_dur.memmap",
                "test_tok_pitch.memmap", "test_tok_dur.memmap"]
if not all([os.path.exists(memp) for memp in memmap_paths]):
    def _write_memmap(filepaths):
        for _n, this_file in enumerate(filepaths):
            if (_n > 1000) and (_n % 1000 == 0):
                pct = psutil.virtual_memory().percent
                if pct > 85:
                    print("Slowing way down to try and let the system catch up on cached writes... memory growing aggressively")
                    print("Sleeping process for 30s...")
                    time.sleep(30)
            # can manually append to each file...
            # extra \0 at the very end
            # TODO: support restart / continuation?
            if this_file == filepaths[-1]:
                extra_tag = "\0"
            else:
                extra_tag = ""
            all_tokenized_tups = [el for el in map(_read_task, [this_file])]
            #r = [el for el in map(_read_task, all_files)]
            #all_tokenized_tups = r
            #print("time to process files", stop_time - start_time)

            # shouldn't have conflices here
            train_tokenized_seqs = [el for name, el in all_tokenized_tups if "_train_" in name]
            valid_tokenized_seqs = [el for name, el in all_tokenized_tups if "_valid_" in name]
            test_tokenized_seqs = [el for name, el in all_tokenized_tups if "_test_" in name]
            del all_tokenized_tups
            if len(train_tokenized_seqs) != 0:
                train_split_seqs = [split_process_and_assert(t_i) for t_i in train_tokenized_seqs]

                train_split_seqs_pitch = [train_split_seqs[_ii][0] for _ii in range(len(train_tokenized_seqs))]
                train_split_seqs_dur = [train_split_seqs[_ii][1] for _ii in range(len(train_tokenized_seqs))]

                train_pitch_str = "\0".join(["\n".join([str(el_i) for el_i in el]) for el in train_split_seqs_pitch]) + extra_tag
                train_dur_str = "\0".join(["\n".join([str(el_i) for el_i in el]) for el in train_split_seqs_dur]) + extra_tag
                with open("train_tok_pitch.memmap", "a") as file_handle:
                    file_handle.write(train_pitch_str)
                with open("train_tok_dur.memmap", "a") as file_handle:
                    file_handle.write(train_dur_str)
            elif len(valid_tokenized_seqs) != 0:
                valid_split_seqs = [split_process_and_assert(t_i) for t_i in valid_tokenized_seqs]

                valid_split_seqs_pitch = [valid_split_seqs[_ii][0] for _ii in range(len(valid_tokenized_seqs))]
                valid_split_seqs_dur = [valid_split_seqs[_ii][1] for _ii in range(len(valid_tokenized_seqs))]

                valid_pitch_str = "\0".join(["\n".join([str(el_i) for el_i in el]) for el in valid_split_seqs_pitch]) + extra_tag
                valid_dur_str = "\0".join(["\n".join([str(el_i) for el_i in el]) for el in valid_split_seqs_dur]) + extra_tag
                with open("valid_tok_pitch.memmap", "a") as file_handle:
                    file_handle.write(valid_pitch_str)
                with open("valid_tok_dur.memmap", "a") as file_handle:
                    file_handle.write(valid_dur_str)
            else:
                assert len(train_tokenized_seqs) == 0
                assert len(valid_tokenized_seqs) == 0
                test_split_seqs = [split_process_and_assert(t_i) for t_i in test_tokenized_seqs]

                test_split_seqs_pitch = [test_split_seqs[_ii][0] for _ii in range(len(test_tokenized_seqs))]
                test_split_seqs_dur = [test_split_seqs[_ii][1] for _ii in range(len(test_tokenized_seqs))]

                test_pitch_str = "\0".join(["\n".join([str(el_i) for el_i in el]) for el in test_split_seqs_pitch]) + extra_tag
                test_dur_str = "\0".join(["\n".join([str(el_i) for el_i in el]) for el in test_split_seqs_dur]) + extra_tag
                with open("test_tok_pitch.memmap", "a") as file_handle:
                    file_handle.write(test_pitch_str)
                with open("test_tok_dur.memmap", "a") as file_handle:
                    file_handle.write(test_dur_str)
    # process for each train val test
    print("Starting multiprocessing jobs")
    start_time = time.time()
    # parallelize train valid and test... 3x speedup
    train_files = [el for el in all_files if "_train_" in el]
    valid_files = [el for el in all_files if "_valid_" in el]
    test_files = [el for el in all_files if "_test_" in el]
    """
    with multiprocessing.Pool(n_processes) as p:
        r = [el for el in p.imap_unordered(_read_task, all_files)]
    """
    n_processes = 3
    with multiprocessing.Pool(n_processes) as p:
        # let each process inspect the pool so we can wait a bit on processing if the cache gets big...
        p.imap_unordered(_write_memmap, [train_files, valid_files, test_files])
        p.close()
        p.join()
    stop_time = time.time()
    print("total time to parse and create memmap blobs,", stop_time - start_time)

print("memmap from saved cache")
# reading back into memory... probably could rewrite to just use the memmap but it should fit in mem
train_pitch_blob = IndexedBlob("train_tok_pitch.memmap")
train_pitch_blob_indices = list(range(0, len(train_pitch_blob.indices) - 1))
train_dur_blob = IndexedBlob("train_tok_dur.memmap")
train_dur_blob_indices = list(range(0, len(train_dur_blob.indices) - 1))
train_int_tokenized_seqs = [([int(el) for el in train_pitch_blob[train_pitch_blob_indices[_ii]].split("\n") if el != ""],
                             [int(el) for el in train_dur_blob[train_dur_blob_indices[_ii]].split("\n") if el != ""]) for _ii in train_pitch_blob_indices]
#from IPython import embed; embed(); raise ValueError()
#train_tokenized_seqs = [train_blob[train_blob_indices[_ii]].split("\n") for _ii in train_blob_indices]
#train_tokenized_seqs = [[el_i for el_i in el if el_i != ""] for el in train_tokenized_seqs]
valid_pitch_blob = IndexedBlob("valid_tok_pitch.memmap")
valid_pitch_blob_indices = list(range(0, len(valid_pitch_blob.indices) - 1))
valid_dur_blob = IndexedBlob("valid_tok_dur.memmap")
valid_dur_blob_indices = list(range(0, len(valid_dur_blob.indices) - 1))
valid_int_tokenized_seqs = [([int(el) for el in valid_pitch_blob[valid_pitch_blob_indices[_ii]].split("\n") if el != ""],
                             [int(el) for el in valid_dur_blob[valid_dur_blob_indices[_ii]].split("\n") if el != ""]) for _ii in valid_pitch_blob_indices]

test_pitch_blob = IndexedBlob("test_tok_pitch.memmap")
test_pitch_blob_indices = list(range(0, len(test_pitch_blob.indices) - 1))
test_dur_blob = IndexedBlob("test_tok_dur.memmap")
test_dur_blob_indices = list(range(0, len(test_dur_blob.indices) - 1))
test_int_tokenized_seqs = [([int(el) for el in test_pitch_blob[test_pitch_blob_indices[_ii]].split("\n") if el != ""],
                            [int(el) for el in test_dur_blob[test_dur_blob_indices[_ii]].split("\n") if el != ""]) for _ii in test_pitch_blob_indices]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_save_path = "parallel_perceiversunmask_nesmdb_{}_models".format(args.task)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
n_unrolled_steps = 2

batch_size = 20

# use 512 of dur, 512 of pitch for 1024 total
sequence_length = 1024
latent_length = 512

hidden_size = 380
self_inner_dim = 900
input_dropout_keep_prob = 1.0
cross_attend_dropout_keep_prob = 1.0
autoregression_dropout_keep_prob = 1.0
inner_dropout_keep_prob = 1.0
final_dropout_keep_prob = 1.0
n_layers = 16

clip_grad = 3
n_train_steps = 500000
learning_rate = 0.0003
min_learning_rate = 1E-5
ramp_til = 10000
decay_til = n_train_steps - 5000
valid_steps_per = 20
save_every = n_train_steps // 10
show_every = max(1, n_train_steps // 500)

# preprocess and throw out any sequences < latent_length

train_split_seqs = [el for el in train_split_seqs if len(el[0]) > latent_length]
train_indices = list(range(len(train_split_seqs)))
valid_split_seqs = [el for el in valid_split_seqs if len(el[0]) > latent_length]
valid_indices = list(range(len(valid_split_seqs)))
test_split_seqs = [el for el in test_split_seqs if len(el[0]) > latent_length]
test_indices = list(range(len(test_split_seqs)))

itr_random_state = np.random.RandomState(5122)
def dataset_itr(batch_size, subset_type="train", target="notes", seed=1234):
    """
    Coroutine of experience replay.
    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    if subset_type == "train":
        use_seqs = train_split_seqs
        use_indices = train_indices
    elif subset_type == "valid":
        use_seqs = valid_split_seqs
        use_indices = valid_indices
    else:
        raise ValueError("Unknown subset_type {}".format(subset_type))

    # start middle end 1/3rd each
    def slice_(ind):
        this_el = use_seqs[ind]
        el_len = len(this_el[0])
        # pitch, dur tuple
        type_split = itr_random_state.randint(3)
        if type_split == 0:
            slice_start = 0
            # start
            el_p = this_el[0][slice_start:slice_start + latent_length]
            el_p[1:] = el_p[:-1]
            # add in <s>
            el_p[0] = 0
            el_d = this_el[1][slice_start:slice_start + latent_length]
            el_d[1:] = el_d[:-1]
            el_d[0] = 0
        elif type_split == 1:
            slice_start = itr_random_state.choice(max(1, el_len - latent_length))
            # middle
            el_p = this_el[0][slice_start:slice_start + latent_length]
            el_d = this_el[1][slice_start:slice_start + latent_length]
        else:
            # end
            slice_start = el_len - latent_length
            el_p = this_el[0][slice_start:slice_start + latent_length]
            el_p[:-1] = el_p[1:]
            # add in </s>
            el_p[-1] = 1
            el_d = this_el[1][slice_start:slice_start + latent_length]
            el_d[:-1] = el_d[1:]
            el_d[-1] = 1
        if target == "notes":
            return (np.array(el_d)[None,None], np.array(el_p)[None,None])
        elif target == "durs":
            return (np.array(el_p)[None,None], np.array(el_d)[None,None])
        else:
            raise ValueErro("Unknown target {}".format(target))

    while True:
        inds = itr_random_state.choice(use_indices, size=batch_size, replace=False)
        batch = np.concatenate([np.concatenate(slice_(_ii), axis=-1) for _ii in inds], axis=0)
        # B I T
        yield batch


def make_batch(batch, random_state):
    shp = batch.shape
    N = shp[0]
    I = shp[1]
    T = shp[2]
    # reshaping B, I, T -> B, I * T will order it SSSSAAAAATTTTTBBBB
    # want interleaved
    batch = batch.transpose(0, 2, 1).reshape(N, T * I)
    # generate corresponding idx, we assume all entries "fill" measure, no 0 padding
    batch_idx = 0. * batch + np.arange(T * I)[None]
    # batch now has correct shape overall, and is interleaved
    # swap to T, B format
    batch = batch.transpose(1, 0)
    batch_idx = batch_idx.transpose(1, 0).astype("int32")
    # idx has trailing 1
    batch_idx = batch_idx[..., None]
    # was 0 min, now 1 min (0 for padding in future datasets)
    batch = batch + 1
    # rewrite the batch for autoregression.
    targets = copy.deepcopy(batch[-latent_length:])
    # sub 1 so targets are 0 : P again
    targets = targets - 1
    return batch, batch_idx, targets

if args.task == "notes_target":
    target = "notes"
elif args.task == "durs_target":
    target = "durs"
else:
    raise ValueError("Unknown task {}".format(args.task))
train_itr = dataset_itr(batch_size, subset_type="train", target=target, seed=123)
# start coroutine
next(train_itr);

valid_itr = dataset_itr(batch_size, subset_type="valid", target=target, seed=1234)
# start coroutine
next(valid_itr);

if args.task == "notes_target":
    n_classes = pitch_n_classes + 1
    query_n_classes = dur_n_classes + 1
elif args.task == "durs_target":
    n_classes = dur_n_classes + 1
    query_n_classes = pitch_n_classes + 1
else:
    raise ValueError("Unknown task {}".format(args.task))

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

FPATH_PERCEIVER = os.path.join(model_save_path, 'parallel_perceiver_sunmask_nesdb_' + args.task + '_{}.pth')
FPATH_LOSSES = os.path.join(model_save_path, 'parallel_perceiver_sunmask_nesdb_losses_' + args.task + '_{}.npz')

model = PerceiverSUNMASK(n_classes=n_classes,
                         query_n_classes=query_n_classes,
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

def build_loss_fn(vocab_size, n_unrolled_steps):
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

        repeated_targets = nn.functional.one_hot(repeated_targets[..., 0].long(), num_classes=vocab_size)
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

        overall_loss = (reduced_mask_active * raw_comb_loss.view(n_unrolled_steps * batch_shp[1], latent_length, vocab_size))
        reduced_loss = (overall_loss).sum(dim=-1)
        loss = torch.mean(reduced_loss, dim=1).mean()
        # upweight by average actives in T since the overall 
        # average penalty for mask weight reduction goes up the longer the sequence is?
        loss = np.sqrt(latent_length) * loss
        return loss
    return local_loss_fn

u_loss_fn = build_loss_fn(n_classes, n_unrolled_steps=n_unrolled_steps)

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
    data = next(train_itr)
    batch, batch_idx, targets = make_batch(data, data_random_state)
    # bring in the SUNMASK loss, fix for flat sequnces / no reshape

    # mask is only latent_length long
    C_prob = data_random_state.rand(batch.shape[1])
    C_mask_base = data_random_state.rand(batch[-latent_length:].shape[0], batch.shape[1])
    C = 1 * (C_mask_base < C_prob[None, :])
    C = (1. - C) # convert to 0 drop format
    C = C.astype(np.int32)

    # mask for the memory portion
    C_mem_prob = data_random_state.rand(batch.shape[1])
    C_mem_mask_base = data_random_state.rand(batch.shape[0], batch.shape[1])
    C_mem = 1 * (C_mem_mask_base < C_mem_prob[None, :])
    C_mem = (1. - C_mem) # convert to 0 drop format
    # set memory mask for target portion to match
    C_mem[-latent_length:] = np.copy(C[-latent_length:])
    # can set this mask to all keep for testing...
    C_mem = C_mem.astype(np.int32)
    #C_mem = C_mem * 0 + 1
    C_mem[-latent_length:] = np.copy(C[-latent_length:])

    x = torch.tensor(batch).type(torch.FloatTensor).to(device)
    x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
    targets = torch.tensor(targets).type(torch.FloatTensor).to(device)
    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
    C2_mem = torch.tensor(C_mem).type(torch.FloatTensor).to(device)

    loss = u_loss_fn(x, x_idx, C2, C2_mem, targets)
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
                data = next(valid_itr)
                batch, batch_idx, targets = make_batch(data, data_random_state)

                # mask is only latent_length long
                C_prob = data_random_state.rand(batch.shape[1])
                C_mask_base = data_random_state.rand(batch[-latent_length:].shape[0], batch.shape[1])
                C = 1 * (C_mask_base < C_prob[None, :])
                C = (1. - C) # convert to 0 drop format
                C = C.astype(np.int32)

                # mask for the memory portion
                C_mem_prob = data_random_state.rand(batch.shape[1])
                C_mem_mask_base = data_random_state.rand(batch.shape[0], batch.shape[1])
                C_mem = 1 * (C_mem_mask_base < C_mem_prob[None, :])
                C_mem = (1. - C_mem) # convert to 0 drop format
                # set memory mask for target portion to match
                C_mem[-latent_length:] = np.copy(C[-latent_length:])
                # can set this mask to all keep for testing...
                C_mem = C_mem.astype(np.int32)
                #C_mem = C_mem * 0 + 1
                C_mem[-latent_length:] = np.copy(C[-latent_length:])

                x = torch.tensor(batch).type(torch.FloatTensor).to(device)
                x_idx = torch.tensor(batch_idx).type(torch.FloatTensor).to(device)
                targets = torch.tensor(targets).type(torch.FloatTensor).to(device)
                C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
                C2_mem = torch.tensor(C_mem).type(torch.FloatTensor).to(device)

                loss = u_loss_fn(x, x_idx, C2, C2_mem, targets)

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
        plt.savefig(model_save_path + os.sep + "perceiver_sunmask_train_losses_{}.png".format(_n_train_steps_taken))
        plt.close('all')
        plt.plot(losses[-5000:])
        plt.plot(valid_losses[-5000:])
        plt.savefig(model_save_path + os.sep + "perceiver_sunmask_train_losses_recent_{}.png".format(_n_train_steps_taken))
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