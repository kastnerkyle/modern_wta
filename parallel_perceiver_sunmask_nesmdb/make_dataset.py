import pretty_midi
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict

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

import os
all_files = []
base_filesdir = "nesmdb_midi"
for fdir in ["train", "valid", "test"]:
    filesdir = base_filesdir + os.sep + fdir
    all_files.extend(sorted([filesdir + os.sep + f for f in os.listdir(filesdir)]))

out_dir = "out_nesmdb_song_str"
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
    print("processing {}".format(x), flush=True)
    return midi_to_string(x)

"""
# check multiprocessing for debug
for _ii, f in enumerate(all_files):
    if _ii > 10:
        break
    el = _read_task(f)
    print(el)
"""

n_processes = multiprocessing.cpu_count() - 1
with multiprocessing.Pool(n_processes) as p:
    r = [el for el in p.imap_unordered(_read_task, all_files)]

def _write_task(el):
    if el is None:
        return
    f = el[0]
    s = el[1]
    if os.sep + "train" + os.sep in f:
        split_type = "train"
    elif os.sep + "valid" + os.sep in f:
        split_type = "valid"
    elif os.sep + "test" + os.sep in f:
        split_type = "test"
    else:
        raise ValueError("parse failed")
    f_lcl = out_dir + os.sep + split_type + "_" + f.split("/")[-1] + ".txt"
    print("writing", f_lcl)
    with open(f_lcl, "w") as file_handle:
        file_handle.write(s)

# write it all with another process pool?
with multiprocessing.Pool(n_processes) as p:
    list(p.imap_unordered(_write_task, r))
    # example of writing them out
    #out_stream = string_to_midi(s)
    #f_lcl_rec = "reco" + f.split("/")[-1]
    #out_stream.write("midi", f_lcl)

# make it into song strings after imap is done...
all_song_strings = {el[0]: el[1] for el in r if el is not None}

all_notes = []
all_notes_voices = []
all_notes_durs = []

all_notes_and_rests = []

all_rests_durs = []

# aggregate stats
for s_k in sorted(all_song_strings.keys()):
    s = all_song_strings[s_k]
    s_split = [el for el in s.split("\n") if el != ""]
    for el in s_split:
        if "WT" in el:
            # WT IS A DUR but handle here...
            # all durs are in ticks
            # WT across ALL voices

            # name (WT), tag (DUR) not useful here
            _, __, wt_dur = el.split("_")
            all_rests_durs.append(wt_dur)
            all_notes_and_rests.append("WT")
        elif "DUR" in el:
            # all durs are in ticks...
            # voice, tick / global song offset, tag (DUR), dur length
            v_d, tick_d, _, d_d = el.split("_")
            all_notes_durs.append(d_d)
        else:
            # all notes 
            assert ("NOTEON" in el)
            # voice, tick / global song offset, tag (NOTEON), pitch
            v_n, tick_n, _, p_n = el.split("_")
            all_notes.append(p_n)
            all_notes_voices.append(v_n)
            all_notes_and_rests.append("N")

def plot_loghist(x, bins):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins, log=True)
    plt.xscale('log')

labels, counts = np.unique(all_notes,return_counts=True)
ticks = range(len(counts))
plt.bar(ticks, counts, align='center')
plt.xticks(ticks, labels)
plt.savefig("string_notes_hist.png")
plt.close()

labels, counts = np.unique(all_notes_voices,return_counts=True)
for l, c in zip(labels, counts):
    print("voice {}".format(l), c)
ticks = range(len(counts))
plt.bar(ticks, counts, align='center')
plt.xticks(ticks, labels)
plt.savefig("string_voices_hist.png")
plt.close()

labels, counts = np.unique(all_notes_and_rests,return_counts=True)
for l, c in zip(labels, counts):
    print("voice {}".format(l), c)
ticks = range(len(counts))
plt.bar(ticks, counts, align='center')
plt.xticks(ticks, labels)
plt.savefig("string_notes_and_rests_hist.png")
plt.close()

all_notes_durs = [int(el) for el in all_notes_durs]
all_rests_durs = [int(el) for el in all_rests_durs]
plot_loghist(all_notes_durs, bins=300)
plt.savefig("float_note_time_dur_hist.png")
plt.close()

plot_loghist(all_rests_durs, bins=300)
plt.savefig("float_rest_time_dur_hist.png")
plt.close()

plot_loghist(all_notes_durs + all_rests_durs, bins=300)
plt.savefig("float_note_and_rest_time_dur_hist.png")
plt.close()
from IPython import embed; embed(); raise ValueError()
