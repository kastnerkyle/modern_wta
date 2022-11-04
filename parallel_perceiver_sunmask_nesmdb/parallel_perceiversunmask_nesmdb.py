import glob
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
from collections import defaultdict
import multiprocessing
from fractions import Fraction
from collections import defaultdict
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

# do all preproc in here, comment it later
def instrument_is_monophonic(ins):
    # Ensure sorted
    notes = ins.notes
    last_note_start = -1
    for n in notes:
        assert n.start >= last_note_start
        last_note_start = n.start

    monophonic = True
    for i in range(len(notes) - 1):
       n0 = notes[i]
       n1 = notes[i + 1]
       if n0.end > n1.start:
           monophonic = False
           break
    return monophonic


def midi_to_string(midi_fp):
    # use preprocessing from LakhNES
    # https://github.com/chrisdonahue/LakhNES/blob/master/data/adapt_lakh_to_nes.py
    min_num_instruments = 3
    filter_mid_len_below_seconds = 5.
    filter_mid_len_above_seconds = 600.
    filter_ins_max_below = 21
    filter_ins_min_above = 108
    filter_mid_bad_times = True
    filter_ins_duplicate = True
    # Ignore unusually large MIDI files (only ~25 of these in the dataset)
    if os.path.getsize(midi_fp) > (512 * 1024): #512K
        return

    try:
        midi = pretty_midi.PrettyMIDI(midi_fp)
    except:
        return

      # Filter MIDIs with extreme length
    midi_len = midi.get_end_time()
    if midi_len < filter_mid_len_below_seconds or midi_len > filter_mid_len_above_seconds:
        return

    # Filter out negative times and quantize to audio samples
    for ins in midi.instruments:
        for n in ins.notes:
            if filter_mid_bad_times:
                if n.start < 0 or n.end < 0 or n.end < n.start:
                      return
            n.start = round(n.start * 44100.) / 44100.
            n.end = round(n.end * 44100.) / 44100.

    instruments = midi.instruments

    # Filter out drum instruments
    drums = [i for i in instruments if i.is_drum]
    instruments = [i for i in instruments if not i.is_drum]

    # Filter out instruments with bizarre ranges
    instruments_normal_range = []
    for ins in instruments:
        pitches = [n.pitch for n in ins.notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        if max_pitch >= filter_ins_max_below and min_pitch <= filter_ins_min_above:
           instruments_normal_range.append(ins)
    instruments = instruments_normal_range
    if len(instruments) < min_num_instruments:
        return

    # Sort notes for polyphonic filtering and proper saving
    for ins in instruments:
        ins.notes = sorted(ins.notes, key=lambda x: x.start)

    # Filter out polyphonic instruments
    instruments = [i for i in instruments if instrument_is_monophonic(i)]
    if len(instruments) < min_num_instruments:
        return

    # Filter out duplicate instruments
    if filter_ins_duplicate:
        uniques = set()
        instruments_unique = []
        for ins in instruments:
            pitches = ','.join(['{}:{:.1f}'.format(str(n.pitch), n.start) for n in ins.notes])
            if pitches not in uniques:
                instruments_unique.append(ins)
                uniques.add(pitches)
        instruments = instruments_unique
        if len(instruments) < min_num_instruments:
            return

    # forget the music21 games
    pretty_midi.pretty_midi.MAX_TICK = 1e16

    ins_names = ['p1', 'p2', 'tr', 'no']
    instruments = sorted(midi.instruments, key=lambda x: ins_names.index(x.name))
    samp_to_events = defaultdict(list)
    for ins in instruments:
        instag = ins.name.upper()
        if instag == "NO":
            continue

        last_start = -1
        last_end = -1
        last_pitch = -1
        for note in ins.notes:
            start = (note.start * 44100) + 1e-6
            end = (note.end * 44100) + 1e-6

            assert start - int(start) < 1e-3
            assert end - int(end) < 1e-3

            start = int(start)
            end = int(end)

            assert start > last_start
            assert start >= last_end

            pitch = note.pitch

            # reformat as DUR

            if last_end >= 0 and last_end != start:
                samp_to_events[last_end].append('{}_NOTEOFF'.format(instag))
                samp_to_events[last_start].append('{}_{}_DUR_{}'.format(instag, last_start, last_end - last_start))
            elif last_end == start:
                samp_to_events[last_start].append('{}_{}_DUR_{}'.format(instag, last_start, last_end - last_start))

            samp_to_events[start].append('{}_{}_NOTEON_{}'.format(instag, start, pitch))

            last_start = start
            last_end = end
            last_pitch = pitch

        if last_pitch != -1:
            samp_to_events[last_start].append('{}_{}_DUR_{}'.format(instag, last_start, last_end - last_start))
            samp_to_events[last_end].append('{}_NOTEOFF'.format(instag))

    # keep the note off logic to make the WT calculations OK
    tx1 = []
    last_samp = 0
    for samp, events in sorted(samp_to_events.items(), key=lambda x: x[0]):
        wt = samp - last_samp
        assert last_samp == 0 or wt > 0
        if wt > 0:
            tx1.append('WT_DUR_{}'.format(wt))
        tx1.extend(events)
        last_samp = samp

    # now remove all noteoff placeholders, leaving pitch+dur pairs
    tx1 = [el for el in tx1 if "NOTEOFF" not in el]
    nsamps = int((midi.time_signature_changes[-1].time * 44100) + 1e-6)
    if nsamps > last_samp:
        tx1.append('WT_DUR_{}'.format(nsamps - last_samp))

    tx1 = '\n'.join(tx1)
    return (midi_fp, tx1)

# dur ranges
# 100 each?
# 1:100
# 101:1000
# 1001:10000
# 10001:
# 1200 across 4 voices... counting WT as its own voice since it is global
# 2 special tokens for dur SOS EOS
n_bins = 100
dur_offset = 2
# space so there's more buckets near 100?
#duration_range_low_bins = np.sort(100 + 1 + 1 - np.geomspace(1, 100 + 1, num=n_bins))
duration_range_low_bins = np.linspace(1, 100 + 1, n_bins)
duration_range_low_mid_bins = np.linspace(101, 1000 + 1, n_bins)
duration_range_high_mid_bins = np.linspace(1001, 10000 + 1, n_bins)
# more buckets near 10k than out at 1M+
duration_range_high_bins = np.logspace(3, 6, n_bins) + 1
#duration_range_high_bins = np.linspace(10001, 100000 + 1)
# shorthand quantize fn
def q_(x):
    if x > 10E3:
        if x > 10E6:
            # limit, digitize should handle but still
            x = 10E6
        return np.digitize([x], duration_range_high_bins)[0] + 3 * n_bins
    elif x > 1000:
        return np.digitize([x], duration_range_high_mid_bins)[0] + 2 * n_bins
    elif x > 100:
        return np.digitize([x], duration_range_low_mid_bins)[0] + n_bins
    elif x >= 1:
        return np.digitize([x], duration_range_low_bins)[0]
    else:
        raise ValueError("Unknown quantization input value {}".format(x))

def iq_(x):
    o_x = int(x) - dur_offset
    # return middle of the bin?
    if o_x >= (3 * n_bins):
        bin_spacing_per = (duration_range_high_bins[1:] - duration_range_high_bins[:-1]) / 2.
        return duration_range_high_bins[o_x - 3 * n_bins] + bin_spacing_per[o_x - 3 * n_bins]
    elif o_x >= (2 * n_bins):
        # uniform spaced here
        bin_spacing = np.mean(duration_range_high_mid_bins[1:] - duration_range_high_mid_bins[:-1]) / 2.
        return duration_range_high_mid_bins[o_x - 2 * n_bins] + bin_spacing
    elif o_x >= (1 * n_bins):
        bin_spacing = np.mean(duration_range_low_mid_bins[1:] - duration_range_low_mid_bins[:-1]) / 2.
        return duration_range_low_mid_bins[o_x - 1 * n_bins] + bin_spacing
    elif o_x >= 0:
        bin_spacing = np.mean(duration_range_low_bins[1:] - duration_range_low_bins[:-1]) / 2.
        return duration_range_low_bins[o_x] + bin_spacing
    else:
       assert x < dur_offset
       assert dur_offset == 2
       if int(x) == 0:
           return "<s>"
       else:
           return "</s>"

# collapse the durs to same buckets
# forces ambiguity in dur <-> notes mapping 
dur_p1_offset = dur_offset
dur_p2_offset = dur_offset #dur_p1_offset + n_bins
dur_tr_offset = dur_offset #dur_p2_offset + n_bins
dur_wt_offset = dur_offset #dur_tr_offset + n_bins
def q_p1(x):
    return q_(x) + dur_p1_offset

def q_p2(x):
    return q_(x) + dur_p2_offset

def q_tr(x):
    return q_(x) + dur_tr_offset

def q_wt(x):
    return q_(x) + dur_wt_offset


# pitch ranges
# 21 : 108
# 3 voices
pitch_min = 21
pitch_max = 108
pitch_range = 108 + 1 - 21
# 1 special WT symbol in notes side
# SOS and EOS globally (0, 1)
# so 
# SOS, EOS, WT
pitch_offset = 3
pitch_tokens_p1 = {k: v for k, v in zip(range(pitch_min, pitch_max+1), range(0 + pitch_offset, pitch_offset + pitch_range))}
rev_pitch_tokens_p1 = {v: k for k, v in pitch_tokens_p1.items()}
pitch_tokens_p2 = {k: v for k, v in zip(range(pitch_min, pitch_max+1), range(0 + pitch_offset + 1 * pitch_range, pitch_offset + 2 * pitch_range))}
rev_pitch_tokens_p2 = {v: k for k, v in pitch_tokens_p2.items()}
pitch_tokens_tr = {k: v for k, v in zip(range(pitch_min, pitch_max+1), range(0 + pitch_offset + 2 * pitch_range, pitch_offset + 3 * pitch_range))}
rev_pitch_tokens_tr = {v: k for k, v in pitch_tokens_tr.items()}
assert all([v not in pitch_tokens_p1.values() for v in pitch_tokens_p2.values()])
assert all([v not in pitch_tokens_p2.values() for v in pitch_tokens_tr.values()])

# useful for the model 
pitch_n_classes = pitch_range * 3 + pitch_offset
dur_n_classes = n_bins * 4 + dur_offset

# we will want to also get some information about the normal length to setup models?
aug_random_state = np.random.RandomState(4142)
def stringmidi_to_tokenized(midi_tup, do_aug=False):
    # logic is slightly weird since the original function operated over full lists
    if midi_tup is None:
        return
    miditxt_fp, miditxt = midi_tup
    # returns tokenized version with all augmentations?
    events = miditxt.split("\n")

    # pre pitch augment
    # start by getting max and min pitch shifts allowed
    if do_aug:
        pitch_augment_ok = []
        min_pitch = 21
        max_pitch = 108
        for p_o in range(-5, 6 + 1):
            this_aug_ok = True
            for event in events:
                if "NOTEON" in event:
                    # voice name, event offset (unused), tag for NOTEON, actual pitch
                    v_n, _, __, p_n = event.split("_")
                    base_p = int(p_n)
                    if p_o + base_p > max_pitch:
                        this_aug_ok = False
                    elif p_o + base_p < min_pitch:
                        this_aug_ok = False
                if not this_aug_ok:
                    break
            if this_aug_ok:
                pitch_augment_ok.append(p_o)
    else:
        pitch_augment_ok = [0]

    # do +- 5%? or no
    # i say no
    # lets spawn 50 clones of each song? with +- 5% timing variation
    # do fixed from -5 to +5?
    # LakhNES does this a fancy way but we are already quantizing to buckets... meh
    if do_aug:
        n_augments_to_try = 100
        z_1 = np.random.uniform(size=n_augments_to_try + 1)
        # now .95 to 1.05
        timing_augments = 1 + (.1 * z_1 - .05)
        # guarantee 1 timing is always the original
        timing_augments[-1] = 1.

        # we always start with 1
        timing_augments_final = [1.]
        # we do this to avoid nasty edge case of multiple processes writing to the same file... ouch
        timing_str_fnames = {"{:0.4f}".format(1.): None}

        # now we shrink the candidate pool down to unique string values...
        # if this fails as an edge case, we still won't get name clashes for file write
        # just less versions of that specific file
        desired_size = 11
        for t_a in timing_augments:
            str_t_a = "{:0.4f}".format(t_a)
            if str_t_a not in timing_str_fnames:
                timing_augments_final.append(t_a)
                timing_str_fnames[str_t_a] = None
            if len(timing_augments_final) >= desired_size:
                break
    else:
        timing_augments_final = [1.0]
        str_t_a = "{:0.4f}".format(1.0)
        timing_str_fnames[str_t_a] = None

    if do_aug:
        return_orig = aug_random_state.choice(2)
        if return_orig == 0:
            pitch_augment_ok = [0]
            timing_augments_final = [1.0]
            str_t_a = "{:0.4f}".format(1.0)
            timing_str_fnames[str_t_a] = None
        else:
            # low chance to get 0 pitch aug, fine
            aug_random_state.shuffle(pitch_augment_ok)
            pitch_augment_ok = pitch_augment_ok[:1]

            # will always get *some* timing aug
            aug_random_state.shuffle(timing_augments_final)
            timing_augments_final = timing_augments_final[:1]

    token_augment_seqs = []
    for t_a in timing_augments_final:
        for p_a in pitch_augment_ok:
            token_seq = []
            # go orderly
            for event in events:
                if "WT" in event:
                    # tag for WT, tag for DUR, then actual tick dur
                    _, __, r_d = event.split("_")
                    int_tick = max(1, int(t_a * int(r_d)))
                    q_bin = q_wt(int_tick)
                    token_seq.append("WT_{}".format(q_bin))
                elif "NOTEON" in event:
                    # voice name, event offset (unused), tag for NOTEON, actual pitch
                    v_n, _, __, p_n = event.split("_")
                    int_pitch = int(p_n)
                    int_pitch = int_pitch + p_a
                    if v_n == "P1":
                        q_pitch = pitch_tokens_p1[int_pitch]
                    elif v_n == "P2":
                        q_pitch = pitch_tokens_p2[int_pitch]
                    elif v_n == "TR":
                        q_pitch = pitch_tokens_tr[int_pitch]
                    else:
                        raise ValueError("Unknown NOTEON voice value {}".format(v_n))
                    token_seq.append("NOTE_{}".format(q_pitch))
                elif "DUR" in event:
                    # voice name, event offset (unused), tag for DUR, then actual tick dur
                    v_d, _, __, d_d = event.split("_")
                    int_tick = max(1, int(t_a * int(d_d)))
                    if v_d == "P1":
                        q_bin = q_p1(int_tick)
                    elif v_d == "P2":
                        q_bin = q_p2(int_tick)
                    elif v_d == "TR":
                        q_bin = q_tr(int_tick)
                    else:
                        raise ValueError("Unknown DUR voice value {}".format(v_d))
                    token_seq.append("DUR_{}".format(q_bin))
                else:
                    print("Unknown event")
                    from IPython import embed; embed(); raise ValueError()
            token_augment_seqs.append((miditxt_fp, p_a, t_a, "\n".join(token_seq)))
    return token_augment_seqs

all_files = []
base_filesdir = "nesmdb_midi"
for fdir in ["train", "valid", "test"]:
    filesdir = base_filesdir + os.sep + fdir
    all_files.extend(sorted([filesdir + os.sep + f for f in os.listdir(filesdir)]))

def _read_task(x):
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    print("processing {}".format(x), flush=True)
    return midi_to_string(x)

n_processes = multiprocessing.cpu_count() - 1
with multiprocessing.Pool(n_processes) as p:
    r = [el for el in p.imap_unordered(_read_task, all_files)]
all_string_tups = [el for el in r if el != None]

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

# this line will process a single file, what we want for data loader 
# split_process_and_assert(stringmidi_to_tokenized(train_string_tups[0], do_aug=True)[0][-1].split("\n"))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_save_path = "parallel_perceiversunmask_nesmdb_{}_models".format(args.task)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
n_unrolled_steps = 2

batch_size = 6

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

# only keep at least max len
all_string_tups = [el for el in all_string_tups if len(el[1].split("\n")) >= sequence_length]
train_string_tups = [(name, el) for name, el in all_string_tups if os.sep + "train" + os.sep in name]
valid_string_tups = [(name, el) for name, el in all_string_tups if os.sep + "valid" + os.sep in name]
test_string_tups = [(name, el) for name, el in all_string_tups if os.sep + "test" + os.sep in name]


# preprocess and throw out any sequences < latent_length
itr_random_state = np.random.RandomState(5122)
def dataset_itr(batch_size, subset_type="train", target="notes", seed=1234):
    """
    Coroutine of experience replay.
    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    if subset_type == "train":
        use_tups = train_string_tups
    elif subset_type == "valid":
        use_tups = valid_string_tups
    elif subset_type == "test":
        use_tups = test_string_tups
    else:
        raise ValueError("Unknown subset_type {}".format(subset_type))
    use_indices = list(range(len(use_tups)))

    # start middle end 1/3rd each
    def slice_(ind):
        if subset_type in ["train", "valid"]:
            aug_flag = True
        else:
            aug_flag = False
        this_el = split_process_and_assert(stringmidi_to_tokenized(use_tups[ind], do_aug=aug_flag)[0][-1].split("\n"))
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
