from music21 import *
from collections import Counter
import numpy as np
import glob
from TIVlib import TIV
from scipy import spatial
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby

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

def pitches_redux(file):
    midi_pitches = []
    stream = converter.parse(file)
    for part in stream.parts:
        for note in part.recurse().getElementsByClass('Chord'):
            midi_pitches.append([p.midi for p in note.pitches])
    return midi_pitches

def time_info(file):
    time_notes_rest = []
    group = {}
    stream = converter.parse(file)
    for part in stream.parts:
        for thisNote_note in part.recurse().getElementsByClass(note.Note):
            time_notes_rest.append([thisNote_note.nameWithOctave, thisNote_note.quarterLength, thisNote_note.offset])
        #for thisChord in part.recurse().getElementsByClass('Chord'):
        #    for c in thisChord.notes:
        #        time_notes_rest.append([c.nameWithOctave, c.quarterLength, c.offset])
        for thisRest in part.recurse().getElementsByClass(note.Rest):
            time_notes_rest.append([thisRest.name, thisRest.quarterLength, thisRest.offset])
    time_notes_rest.sort(key=lambda i: i[2])
    for name, x, y in time_notes_rest:
        group.setdefault(y, []).append((name, x))
    return group

def new_stream(file):
    stream1 = converter.parse(file)
    stream2 = stream.Stream(file)
    m = pitches_redux(file)
    t_info = time_info(file)
    #print(t_info)
    #for t in t_info:
        #IN CASE OF CONSTRUCTING THE SCORE (HAVE SOME PROBLEMS THOUGH)
    #    if t[0] == 'rest':
    #        nr = note.Rest(t[0])
    #        #nr.style.color = 'red'
    #        stream1.insert(t[2], nr)
    #    else:
    #        nt = note.Note(t[0])
    #        #nt.style.color = 'red'
    #        stream1.insert(t[2], nt)

    notes_lst = []
    for s in m:
        red = TIV_Redux(s, spatial.distance.euclidean)
        e = [x[0] for x in red]
        n = [note.Note(s[i]) for i in e]
        n_name = [note.Note(s[i]).nameWithOctave for i in e]
        if n_name:
            notes_lst.append(n_name)
    print(notes_lst)

    times_offset = []
    for key, value in t_info.items():
        #print(key, value)
        if len(value) > 3:
            times_offset.append(key)
    print(times_offset)

    final_list = zip(notes_lst , times_offset)
    zipped_list = list(final_list)
    #print(zipped_list)

    for i in range(len(zipped_list)):
        for j in range(len(zipped_list[i][0])):
            if len(zipped_list[i][0]) > 0:
                note_zipped = note.Note(zipped_list[i][0][j])
                #p = pitch.Pitch(note_zipped)
                #n = note.Note(p)
                note_zipped.style.color = 'pink'
                stream1.insert(zipped_list[i][1], note_zipped)
                stream2.insert(zipped_list[i][1], note_zipped)
    #stream1.show()
    stream2.show() #Just to see the notes that are going to be cutted
    return stream1

#path_Bach = './Datasets/Bach_Preludes'
#for file in glob.glob(path_Bach + './*.mid'):
#    t = time_info(file)
#    print(t)
#    s = new_stream(file)
#    print(s)

s = new_stream('./Datasets/Bach_Preludes/BachPrelude_05.mid')
#s.show()