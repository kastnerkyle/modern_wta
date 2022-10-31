import pretty_midi
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict
import time

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


# we will want to also get some information about the normal length to setup models?
def stringmidi_to_tokenized(miditxt_fp):
    # returns tokenized version with all augmentations
    with open(miditxt_fp, "r") as file_handle:
        l = file_handle.read()
    events = l.split("\n")

    # pre pitch augment
    # start by getting max and min pitch shifts allowed
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

    # do +- 5%? or no
    # i say no
    # lets spawn 50 clones of each song? with +- 5% timing variation
    # do fixed from -5 to +5?
    # LakhNES does this a fancy way but we are already quantizing to buckets... meh
    n_augments = 10
    z_1 = np.random.uniform(size=n_augments + 1)
    # now .95 to 1.05
    timing_augments = 1 + (.1 * z_1 - .05)
    # guarantee 1 timing is always the original
    timing_augments[-1] = 1.

    token_augment_seqs = []
    for t_a in timing_augments:
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

if __name__ == "__main__":
    import os
    all_files = []
    filesdir = "out_nesmdb_song_str"
    all_files.extend(sorted([filesdir + os.sep + f for f in os.listdir(filesdir)]))

    out_dir = "tokenized_nesmdb_song_str"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    """
    # parallelize...
    for _ii, f in enumerate(all_files):
        print("processing {} of {}".format(_ii + 1, len(all_files)), f)
        s = midi_to_string(f)
        f_lcl = out_dir + os.sep + "str_" + f.split("/")[-1] + ".txt"
        with open(f_lcl, "w") as file_handle:
            file_handle.write(s)
        #out_stream = string_to_midi(s)
        #f_lcl = "reco" + f.split("/")[-1]
        #out_stream.write("midi", f_lcl)
    """

    def _read_task(x):
        # set seed uniquely for each task since we are doing randomized pitch augmentations
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        print("processing {}".format(x), flush=True)
        return stringmidi_to_tokenized(x)

    fpath = "out_nesmdb_song_str/valid_293_Shinobi_08_09BossBGM2.mid.txt"
    all_files = [fpath]
    """
    # check multiprocessing for debug
    results = []
    for _ii, f in enumerate(all_files):
        if _ii > 10:
            break
        el = _read_task(f)
        results.extend(el)
    """

    n_processes = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(n_processes) as p:
        # flatten list of results with its augments...
        r = [el_i for el in p.imap_unordered(_read_task, all_files) for el_i in el]

    def _write_task(el):
        if el is None:
            return
        f = el[0]
        p_a = el[1]
        t_a = el[2]
        s = el[-1]
        # already has .txt on it
        p_m = "p" if int(p_a) >= 0 else "m"
        p_a_str = p_m + str(abs(int(p_a)))
        t_a_str = "{:0.2f}".format(t_a)
        t_a_str = t_a_str.replace(".", "-")
        f_lcl = out_dir + os.sep + "tokenized_" + p_a_str + "_" + t_a_str + "_" + f.split("/")[-1]
        print("writing", f_lcl)
        with open(f_lcl, "w") as file_handle:
            file_handle.write(s)

    """
    # check multiprocessing for debug
    for _ii, r_i in enumerate(r):
        if _ii > 120:
            break
        el = _write_task(r_i)
    """

    # write it all with another process pool?
    with multiprocessing.Pool(n_processes) as p:
        list(p.imap_unordered(_write_task, r))
        # example of writing them out
        #out_stream = string_to_midi(s)
        #f_lcl_rec = "reco" + f.split("/")[-1]
        #out_stream.write("midi", f_lcl)

    # what now...
