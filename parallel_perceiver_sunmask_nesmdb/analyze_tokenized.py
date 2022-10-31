import pretty_midi
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict
# this will import the tokenization settings
import os
import tempfile
import pretty_midi
from make_tokenized import *
from IPython import embed; embed(); raise ValueError()

def detokenize(events):
    event_list = []
    for event in events:
        if "NOTE" in event:
            _, n_n = event.split("_")
            int_tok = int(n_n)
            if int_tok in rev_pitch_tokens_p1:
                assert int_tok not in rev_pitch_tokens_p2
                assert int_tok not in rev_pitch_tokens_tr
                int_note = rev_pitch_tokens_p1[int_tok]
                s_i = "P1_OFFSET_NOTEON_{}".format(int_note)
                event_list.append(s_i)
            elif int_tok in rev_pitch_tokens_p2:
                assert int_tok not in rev_pitch_tokens_p1
                assert int_tok not in rev_pitch_tokens_tr
                int_note = rev_pitch_tokens_p2[int_tok]
                s_i = "P2_OFFSET_NOTEON_{}".format(int_note)
                event_list.append(s_i)
            elif int_tok in rev_pitch_tokens_tr:
                assert int_tok not in rev_pitch_tokens_p1
                assert int_tok not in rev_pitch_tokens_p2
                int_note = rev_pitch_tokens_tr[int_tok]
                s_i = "TR_OFFSET_NOTEON_{}".format(int_note)
                event_list.append(s_i)
            else:
                #TR_5328_NOTEON_41
                print("Unknown note id {}".format(int_tok))
                from IPython import embed; embed(); raise ValueError()
        elif "WT" in event:
            # tag, token
            _, wt_d = event.split("_")
            int_dur = int(wt_d)
            tick_dur = iq_(int_dur)
            int_tick_dur = int(np.round(tick_dur))
            event_list.append("WT_DUR_{}".format(int_tick_dur))
        elif "DUR" in event:
            assert "NOTEON" in event_list[-1]
            # assign to the voice before this one...
            p_v, _, __, ___ = event_list[-1].split("_")
            _, d_d = event.split("_")
            int_dur = int(d_d)
            tick_dur = iq_(int_dur)
            int_tick_dur = int(np.round(tick_dur))
            event_list.append(p_v + "_OFFSET_DUR_{}".format(int_tick_dur))
    return event_list


def real_events_to_midi(real_events):
    nsamps = sum([int(x.split('_')[-1]) for x in real_events if "DUR" in x])

    # Create MIDI instruments
    p1_prog = pretty_midi.instrument_name_to_program('Lead 1 (square)')
    p2_prog = pretty_midi.instrument_name_to_program('Lead 2 (sawtooth)')
    tr_prog = pretty_midi.instrument_name_to_program('Synth Bass 1')
    no_prog = pretty_midi.instrument_name_to_program('Breath Noise')
    p1 = pretty_midi.Instrument(program=p1_prog, name='p1', is_drum=False)
    p2 = pretty_midi.Instrument(program=p2_prog, name='p2', is_drum=False)
    tr = pretty_midi.Instrument(program=tr_prog, name='tr', is_drum=False)
    no = pretty_midi.Instrument(program=no_prog, name='no', is_drum=True)

    name_to_ins = {'P1': p1, 'P2': p2, 'TR': tr, 'NO': no}
    name_to_pitch = {'P1': None, 'P2': None, 'TR': None, 'NO': None}
    name_to_start = {'P1': None, 'P2': None, 'TR': None, 'NO': None}
    name_to_max_velocity = {'P1': 100, 'P2': 100, 'TR': 30, 'NO': 100}

    # this should unite notes and durs
    no_notes = [el for el in real_events if "NOTEON" not in el]
    no_notes_durs = [el for el in real_events if ("WT" in el) or ("NOTEON" in el)]

    assert len(no_notes) == len(no_notes_durs)
    for el1, el2 in zip(no_notes, no_notes_durs):
        if "WT" in el1:
            assert "WT" in el2
            assert el1 == el2
        if "WT" in el2:
            assert "WT" in el1
            assert el1 == el2
    # if we get to here the unification worked
    zip_events = list([(el1, el2) for el1, el2 in zip(no_notes_durs, no_notes)])

    samp = 0
    for z_e in zip_events:
        note_half, dur_half = z_e[0], z_e[1]
        if "WT" in note_half:
            samp += int(note_half.split("_")[-1])
        else:
            #tokens = event.split('_')
            #name = tokens[0]
            assert "DUR" in dur_half
            assert "NOTEON" in note_half
            p_n, _, __, p_p = note_half.split("_")
            _, __, ___, p_d = dur_half.split("_")
            ins = name_to_ins[p_n]
            ins.notes.append(pretty_midi.Note(
                velocity=name_to_max_velocity[p_n],
                pitch=int(p_p),
                start=samp / 44100.,
                end=(samp + int(p_d)) / 44100.))

    # Deactivating this for generated files
    #for name, pitch in name_to_pitch.items():
    #  assert pitch is None

    # Create MIDI and add instruments
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=22050)
    midi.instruments.extend([p1, p2, tr])#, no])

    # Create indicator for end of song
    eos = pretty_midi.TimeSignature(1, 1, nsamps / 44100.)
    midi.time_signature_changes.append(eos)
    return midi

# listen to example after inverting tokenization
fpath = "tokenized_nesmdb_song_str/tokenized_p0_1-00_valid_293_Shinobi_08_09BossBGM2.mid.txt"
with open(fpath, "r") as file_handle:
    l = file_handle.read()
tokenized_events = l.split("\n")

"""
# simulate randomly cutting in for batch loader?
cut_length = 1024
cut_points = [n for n, el in enumerate(tokenized_events) if "WT" in el and (n + cut_length < len(tokenized_events))]
# example of how to randomly slice
rand_cut = np.random.RandomState(4142)
slice_ind = rand_cut.choice(cut_points)
cut_events = tokenized_events[slice_ind:slice_ind + cut_length]
real_events = detokenize(cut_events)
midi_res = real_events_to_midi(real_events)
midi_res.write("tmp.mid")
"""

all_files = []
filesdir = "tokenized_nesmdb_song_str"
all_files.extend(sorted([filesdir + os.sep + f for f in os.listdir(filesdir)]))
def _read_task(fpath):
    print("processing", fpath)
    with open(fpath, "r") as file_handle:
        l = file_handle.read()
        tokenized_events = l.split("\n")
    return len(tokenized_events)

n_processes = multiprocessing.cpu_count() - 1
with multiprocessing.Pool(n_processes) as p:
    r = [el for el in p.imap_unordered(_read_task, all_files)]
all_files_lens = r

n_bins = 100
#hist, bins = np.histogram(all_files_lens, bins=n_bins)
#logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]),len(bins))
#plt.hist(all_files_lens, bins=logbins, log=True)
plt.hist(all_files_lens, bins=n_bins, log=True)
#plt.xscale('log')
plt.savefig("float_length_hist.png")
plt.close()
from IPython import embed; embed(); raise ValueError()


