from music21 import *
from collections import Counter
import numpy as np
import pandas as pd
from TIVlib import TIV
from scipy import spatial
import matplotlib.pyplot as plt
import glob

# The full TIV library isn't importing correctly to the program, so here is a part of the TIV library.
class TIV:
    weights_symbolic = [2, 11, 17, 16, 19, 7]
    weights_audio = [3, 8, 11.5, 15, 14.5, 7.5]

    def __init__(self, energy, vector):
        self.energy = energy
        self.vector = vector

    def abs_vector(self):
        return np.abs(self.vector)

    def phases_vector(self):
        return np.angle(self.vector)

    def get_vector(self):
        return np.array(self.vector)

    def diss(self):  # Compute dissonance
        return 1 - (np.linalg.norm(self.vector) / np.sqrt(np.sum(np.dot(self.weights_symbolic, self.weights_symbolic))))

    def coeffs(self, coef):  # Compute coefficient
        return self.abs_vector()[coef] / self.weights_symbolic[coef]

    def chromaticity(self):  # Compute chromaticity
        return self.abs_vector()[0] / self.weights_symbolic[0]

    def dyads(self):  # Compute dyadicity
        return self.abs_vector()[1] / self.weights_symbolic[1]

    def triads(self):  # Compute triadicity (triads)
        return self.abs_vector()[2] / self.weights_symbolic[2]

    def d_q(self):  # Refers a possible diminished quality
        return self.abs_vector()[3] / self.weights_symbolic[3]

    def diatonal(self):  # Compute diatonicity
        return self.abs_vector()[4] / self.weights_symbolic[4]

    def tone(self):  # Define wholetoneness
        return self.abs_vector()[5] / self.weights_symbolic[5]

    @classmethod
    def from_pcp(cls, pcp, symbolic=True):
        #      Get TIVs from pcp, as the original method
        #     :param pcp: 12xN vector containing N pcps
        #     :return: TIVCollection object
        #     """
        if pcp.shape[0] == 12:
            fft = np.fft.rfft(pcp, n=12)
            energy = fft[0]
            vector = fft[1:7]
            if symbolic:
                vector = ((vector / energy) * cls.weights_symbolic)
            else:
                vector = ((vector / energy) * cls.weights_audio)
            return cls(energy, vector)
        else:
            return cls(complex(0), np.array([0, 0, 0, 0, 0, 0]).astype(complex))

    def plot_TIV(self):
        titles = ["m2/M7", "TT", "M3/m6", "m3/M6", "P4/P5", "M2/m7"]
        TIVector = self.vector / self.weights_symbolic
        i = 1
        for tiv in TIVector:
            circle = plt.Circle((0, 0), 1, fill=False)
            plt.subplot(2, 3, i)
            plt.subplots_adjust(hspace=0.4)
            plt.gca().add_patch(circle)
            plt.title(titles[i - 1])
            plt.scatter(tiv.real, tiv.imag)
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            plt.grid()
            i = i + 1
        plt.show()

    def hchange(self):
        tiv_array = self.vector
        results = []
        for i in range(len(tiv_array)):
            distance = TIV.euclidean(tiv_array[i + 1], tiv_array[i])
            results.append(distance)
        return results

#Just to order list of TIV_Redux
def sort_dist(dist):
    return dist[1]

def distance_calc(pitches_chord, dist):
    dist_list = []
    hist = Counter(pitches_chord)
    pitch_list = [0] * 12
    for pitch in hist:
        index = pitch % 12
        pitch_list[index] += hist[pitch]
    TIV_vector = TIV.from_pcp(np.array(pitch_list))

    for i in range(len(pitches_chord)):
        each_pitch = pitches_chord[i]
        aux_vector = [0] * 12
        aux_vector[each_pitch % 12] += 1
        aux_vector = TIV.from_pcp(np.array(aux_vector))
        distance = dist(TIV_vector.vector, aux_vector.vector)
        dist_list.append((i, distance))
    return dist_list

#The aim is to only show the notes that are going to be cut by the reduction step
#(Taking into account that the representation of each chord are only up to 3 notes maximum)
def TIV_Redux(pitches_chord, dist):
    dist_list = distance_calc(pitches_chord, dist)
    dist_list.sort(key=sort_dist)
    dist_reduced = dist_list[3:] #only show the notes that are going to be cut, reducing the chord
    return dist_reduced

b = corpus.parse('bwv66.6')
bChords = b.chordify()
bChords_PAINTED_NOTES = b.chordify()
bChords_REDUCED_SCORE = b.chordify()

midi_pitches = []
duration_pitches = []
offset_pitches = []

for chord in bChords.flatten().getElementsByClass('Chord'):
    #chord = chord.removeRedundantPitchNames()
    midi_pitches.append([n.pitch.midi for n in chord.notes])
    duration_pitches.append([n.quarterLength for n in chord.notes])
    offset_pitches.append([chord.offset for n in chord])

#print(midi_pitches)
#print(duration_pitches)
#print(offset_pitches)

reduced_notes_lst = []
reduced_chords_lst = []
for s in midi_pitches:
    ss = s
    red_notes = TIV_Redux(ss, spatial.distance.euclidean)
    e = [x[0] for x in red_notes]
    n_name = [note.Note(ss[i]).nameWithOctave for i in e]
    if n_name:
        reduced_notes_lst.append(n_name)

    n = [s[i] for i in e]
    for element in n:
        if element in s:
            s.remove(element)
    reduced_chords_lst.append(s)

bChords_REDUCED_SCORE.replace(ss,s)
#for i in range(len(offset_pitches)):
#    bChords_REDUCED_SCORE.insertIntoNoteOrChord(offset_pitches[i][0], note.Note(reduced_chords_lst[i][j]))
bChords_REDUCED_SCORE = bChords_REDUCED_SCORE.flat

times_offset = []
for value in offset_pitches:
    if len(value) >= 4:
        times_offset.append(value[0])
#times_offset.sort()
#print(times_offset)

final_reduced_list = zip(reduced_notes_lst, times_offset)
zipped_reduced_list = list(final_reduced_list)
zipped_reduced_list.sort(key=lambda x: x[1])
#print(zipped_reduced_list)

for i in range(len(zipped_reduced_list)):
    for j in range(len(zipped_reduced_list[i][0])):
        if len(zipped_reduced_list[i][0]) > 0:
            note_zipped = note.Note(zipped_reduced_list[i][0][j])
            note_zipped.style.color = 'yellow'
            bChords_PAINTED_NOTES.insert(zipped_reduced_list[i][1], note_zipped)

bChords_PAINTED_NOTES = bChords_PAINTED_NOTES.flat

#bChords.show()
#bChords_PAINTED_NOTES.show()
bChords_REDUCED_SCORE.show()
