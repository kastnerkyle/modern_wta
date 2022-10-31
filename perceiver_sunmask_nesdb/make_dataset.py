# Find MIDIs
import glob
# Pick a random one
import random
import numpy as np
import pretty_midi
import os
mid_fps = sorted(glob.glob('nesmdb_midi/train/*'))
print(len(mid_fps))
# borrowed wholesale from Douglas Duhaime https://douglasduhaime.com/posts/making-chiptunes-with-markov-models.html
# any bugs mine (KK)
from music21.note import Note
import music21
import time
import shelve

# use filtering criteria from LakhNES
# https://github.com/chrisdonahue/LakhNES/blob/edd19a972742b449ab23c0f03c3fdff3b12b1bad/data/adapt_lakh_to_nes.py
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

def midi_to_ngod(midi_fp):
  try:
      midi = pretty_midi.PrettyMIDI(midi_fp)
      # parse the musical information stored in the midi file
      # store as note global offset duration tuple
      # will try and quantize + resolve issues later
      # we will quantize these manually to try and handle global timing offsets
  except:
      return None

  # use filtering criteria from LakhNES
  # https://github.com/chrisdonahue/LakhNES/blob/edd19a972742b449ab23c0f03c3fdff3b12b1bad/data/adapt_lakh_to_nes.py
  min_num_instruments = 1
  filter_mid_len_below_seconds = 5
  filter_mid_len_above_seconds = 600
  filter_mid_bad_times = True
  filter_ins_max_below = 21
  filter_ins_min_above = 108
  filter_ins_duplicate = True
  filter_mid_bad_times = True

  # Ignore unusually large MIDI files (only ~25 of these in the dataset)
  if os.path.getsize(midi_fp) > (512 * 1024): #512K
    return

  # Filter MIDIs with extreme length
  midi_len = midi.get_end_time()
  if midi_len < filter_mid_len_below_seconds or midi_len > filter_mid_len_above_seconds:
    return

  # Filter out negative times and very very short notes, and quantize to audio samples
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

  tempos = midi.get_tempo_changes()
  nztempos = [t[0] for t in tempos if t[0] != 0]
  tmin = min(nztempos)
  tmax = max(nztempos)
  if tmin == tmax:
      tempo = tmin
  else:
      return
  # pitch, time rolls
  # will have to resmaple/interpolate these to match length, ouch
  # assumptions - 30 bars per min
  # roughly 100 bpm
  # so that means we want 
  samp_fs = 24
  piano_rolls = [(i.name, i.get_piano_roll(fs=samp_fs)) for i in instruments]
  drum_rolls = [(d.name, d.get_piano_roll(fs=samp_fs)) for d in drums]
  return piano_rolls, drum_rolls


tot = 0
err = 0
times = []
start_time = time.time()
# need to cache?
from collections import OrderedDict
results = OrderedDict()
accepted = 0
rejected = 0
for mid_fp in mid_fps:
    beg_time = time.time()
    tot += 1
    #midi_data = pretty_midi.PrettyMIDI(mid_fp)
    ret = midi_to_ngod(mid_fp)
    if ret is not None:
        piano_rolls, drum_rolls = ret
        # filter out all the ones with rolls longer than 8k or under 225
        lens = [p_r[1].shape[1] for p_r in piano_rolls]
        if max(lens) <= 8000 and min(lens) >= 200:
            results[mid_fp.split("/")[-1]] = ret
            accepted += 1
        else:
            rejected += 1
        print("Processed {}".format(mid_fp))
        last_time = time.time()
        times.append(last_time - beg_time)
        last_time = beg_time
    else:
        print("Error {}".format(mid_fp))
        err += 1
final_time = time.time()
print("total starting files", len(mid_fps))
print("number which passed initial filter", accepted + rejected)
print("rejected due to length limits", rejected)
print("final number of pieces", accepted)
print("average processing time per file", np.mean(times))
print("total time", final_time - start_time)

with shelve.open("cached_midi_data.shlv", writeback=True) as db:
    for k in results.keys():
        db[k] = results[k]

# get a count of how many have 3 voices, how many have 2, how many have 1
# looks like almost all have 3 voices (~2900)
import matplotlib.pyplot as plt
num_voices = [len(results[k][0]) for k in results.keys()]
plt.hist(num_voices, bins=[1,2,3,4])
plt.savefig("tmp_voices.png")

time_lengths = [[tup[1].shape[1] for tup in results[k][0]] for k in results.keys()]
flat_time_lengths = [t for tlist in time_lengths for t in tlist]
plt.hist(flat_time_lengths, bins=100)
plt.savefig("tmp_len.png")

