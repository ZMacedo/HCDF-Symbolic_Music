#!/usr/bin/env python
# coding: utf-8

# # THINGS TO BE CLARIFIED WITH PEDRO
# 
# - NOT UNDERSTANDING WHY THE RESULTS ARE LIKE THAT (it must be because there are functions that are not giving the output that was supposed to have, but i can't understand why, and i have already made various kind of approaches to each one of them)
# - Trying to understand what get_peaks_hcdf, harmonic_change function really must have as output

# # HEADING ATTENTION TO:
# 
# - Resolve the results problem; (in progress)
# - Seeing what music21 can do to present this "tree representation" scheme (taking the stronger note of a chord); (in progress)
# - Writing the dissertation.

# # Symbolic Harmonic Change Detection in the Tonal Interval Space

# The detection of chord boundaries (onset or beginning or chords in the musical surface) is typically addressed within Music Information Retrieval as harmonic change detection. Existing solutions are known to improve complex systems for automatic chord detection as a preprocessing segmentation stage. This project aims at improving the performance of harmonic change detection by adopting a tree-based representation for reducing the complex structure of symbolic music manifestations to an n-chord representation, targeting the basic underlying triadic structure of Western tonal harmony.

# This is the code implemented to achieve the main objectives referred in the dissertation "Symbolic Harmonic Change Detection in the Tonal Interval Space" by José Macedo, with Gilberto Bernardes as Supervisor and Pedro Ramoneda as Co-Supervisor. 

# In[1]:


#get_ipython().system('pip install unidecode')

from music21 import *
import mido
from mido import MidiFile
import mirdata
import igraph as ig
import os
import plotly.express as px
import sys
import py_midicsv as pm
import numpy as np
import glob
import csv
import json
import librosa
import pretty_midi
import libfmp.c1
import libfmp.c3
import libfmp.b
import pickle
import mir_eval
from TIVlib import TIV
import matplotlib.pyplot as plt
import pandas as pd
from unidecode import unidecode
from pymidi import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter
#np.seterr(all='raise')


# # TIS&TIV - Tonal Interval Space & Tonal Interval Vectors 
# 
# A truncated version of TIV library [1].
# 
# [1] - Ramires, A., Bernardes, G., Davies, M.E., & Serra, X. (2020). TIV.lib: an open-source library for the tonal description of musical audio. ArXiv, abs/2008.11529.

# In[2]:


#The full TIV library isn't importing correctly to the program, so here is a part of the TIV library.

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

    def diss(self): #Compute dissonance
        return 1 - (np.linalg.norm(self.vector) / np.sqrt(np.sum(np.dot(self.weights_symbolic, self.weights_symbolic))))

    def coeffs(self, coef): #Compute coefficient
        return self.abs_vector()[coef] / self.weights_symbolic[coef]

    def chromaticity(self): #Compute chromaticity
        return self.abs_vector()[0] / self.weights_symbolic[0]

    def dyads(self): #Compute dyadicity
        return self.abs_vector()[1] / self.weights_symbolic[1]

    def triads(self): #Compute triadicity (triads)
        return self.abs_vector()[2] / self.weights_symbolic[2]

    def d_q(self): #Refers a possible diminished quality
        return self.abs_vector()[3] / self.weights_symbolic[3]

    def diatonal(self): #Compute diatonicity
        return self.abs_vector()[4] / self.weights_symbolic[4]

    def tone(self): #Define wholetoneness
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

   # @classmethod
   # def euclidean(cls, tiv1, tiv2):
   #     return np.linalg.norm(tiv1.vector - tiv2.vector)

   # @classmethod
   # def cosine(cls, tiv1, tiv2):
   #     a = np.concatenate((tiv1.vector.real, tiv1.vector.imag), axis=0)
   #     b = np.concatenate((tiv2.vector.real, tiv2.vector.imag), axis=0)
   #     if all_zero(a) or all_zero(b):
   #         distance_computed = euclidean(a, b)
   #     else:
   #         distance_computed = cosine(a, b)
   #     return distance_computed


# In[3]:


#Taken from https://github.com/qsdfo/midi_to_numpy/blob/master/read_midi.py

#To read midi into piano rolls (the only part of this class that is adamant is read_file())
class Read_midi(object):
    def __init__(self, song_path, quantization):
        ## Metadata
        self.__song_path = song_path
        self.__quantization = quantization

        ## Pianoroll
        self.__T_pr = None

        ## Private misc
        self.__num_ticks = None
        self.__T_file = None

    @property
    def quantization(self):
        return self.__quantization

    @property
    def T_pr(self):
        return self.__T_pr

    @property
    def T_file(self):
        return self.__T_file

    def get_total_num_tick(self):
        # Midi length should be written in a meta message at the beginning of the file,
        # but in many cases, lazy motherfuckers didn't write it...

        # Read a midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)

        # Parse track by track
        num_ticks = 0
        for i, track in enumerate(mid.tracks):
            tick_counter = 0
            for message in track:
                # Note on
                time = float(message.time)
                tick_counter += time
            num_ticks = max(num_ticks, tick_counter)
        self.__num_ticks = num_ticks

    def get_pitch_range(self):
        mid = MidiFile(self.__song_path)
        min_pitch = 200
        max_pitch = 0
        for i, track in enumerate(mid.tracks):
            for message in track:
                if message.type in ['note_on', 'note_off']:
                    pitch = message.note
                    if pitch > max_pitch:
                        max_pitch = pitch
                    if pitch < min_pitch:
                        min_pitch = pitch
        return min_pitch, max_pitch

    def get_time_file(self):
        # Get the time dimension for a pianoroll given a certain quantization
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat
        # Total number of ticks
        self.get_total_num_tick()
        # Dimensions of the pianoroll for each track
        self.__T_file = int((self.__num_ticks / ticks_per_beat) * self.__quantization)
        return self.__T_file

    def read_file(self):
        # Read the midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat

        # Get total time
        self.get_time_file()
        T_pr = self.__T_file
        # Pitch dimension
        N_pr = 128
        pianoroll = {}

        def add_note_to_pr(note_off, notes_on, pr):
            pitch_off, _, time_off = note_off
            # Note off : search for the note in the list of note on,
            # get the start and end time
            # write it in th pr
            match_list = [(ind, item) for (ind, item) in enumerate(notes_on) if item[0] == pitch_off]
            if len(match_list) == 0:
                print("Try to note off a note that has never been turned on")
                # Do nothing
                return

            # Add note to the pr
            pitch, velocity, time_on = match_list[0][1]
            pr[time_on:time_off, pitch] = velocity
            # Remove the note from notes_on
            ind_match = match_list[0][0]
            del notes_on[ind_match]
            return

        # Parse track by track
        counter_unnamed_track = 0
        for i, track in enumerate(mid.tracks):
            # Instanciate the pianoroll
            pr = np.zeros([T_pr, N_pr])
            time_counter = 0
            notes_on = []
            for message in track:

                ##########################################
                ##########################################
                ##########################################
                # TODO : keep track of tempo information
                # import re
                # if re.search("tempo", message.type):
                #     import pdb; pdb.set_trace()
                ##########################################
                ##########################################
                ##########################################


                # print message
                # Time. Must be incremented, whether it is a note on/off or not
                time = float(message.time)
                time_counter += time / ticks_per_beat * self.__quantization
                # Time in pr (mapping)
                time_pr = int(round(time_counter))
                # Note on
                if message.type == 'note_on':
                    # Get pitch
                    pitch = message.note
                    # Get velocity
                    velocity = message.velocity
                    if velocity > 0:
                        notes_on.append((pitch, velocity, time_pr))
                    elif velocity == 0:
                        add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)
                # Note off
                elif message.type == 'note_off':
                    pitch = message.note
                    velocity = message.velocity
                    add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)

            # We deal with discrete values ranged between 0 and 127
            #     -> convert to int
            pr = pr.astype(np.int16)
            if np.sum(np.sum(pr)) > 0:
                name = unidecode(track.name)
                name = name.rstrip('\x00')
                if name == u'':
                    name = 'unnamed' + str(counter_unnamed_track)
                    counter_unnamed_track += 1
                if name in pianoroll.keys():
                    # Take max of the to pianorolls
                    pianoroll[name] = np.maximum(pr, pianoroll[name])
                else:
                    pianoroll[name] = pr
        return pianoroll


# # Auxiliary Functions 
# 
# By Pedro Ramoneda in "Harmonic Change Detection from Musical Audio"

# In[4]:


def gaussian_blur(centroid_vector, sigma):
    centroid_vector = gaussian_filter(centroid_vector, sigma=sigma)
    return centroid_vector

def get_peaks_hcdf(hcdf_function, rate_centroids_second, symbolic=True):
    changes = [0]
    hcdf_changes = []
    last = 0
    for i in range(2, hcdf_function.shape[0] - 1):
        if hcdf_function[i - 1] < hcdf_function[i] and hcdf_function[i + 1] < hcdf_function[i]:
            hcdf_changes.append(hcdf_function[i])
            if not symbolic:
                changes.append(i / rate_centroids_second)
            else:
                changes.append(i)
            last = i
    return np.array(changes), np.array(hcdf_changes)

#4 - Distance Calculation (Euclidean and Cosine)
def distance_calc(centroid_point, distance):
    dist = []
    if distance == 'Euclidean':
        for j in range(1, centroid_point.shape[1] - 1):
            aux = 0
            for i in range(0, centroid_point.shape[0]):
                aux += ((centroid_point[i][j + 1] - centroid_point[i][j - 1]) ** 2)
            aux = np.math.sqrt(aux)
            dist.append(aux)
    
    if distance == 'Cosine':
        for j in range(1, centroid_point.shape[1] - 1):
            cosine_distance = cosine(centroid_point[:, j - 1], centroid_point[:, j + 1])
            dist.append(cosine_distance)
    dist.append(0)

    return np.array(dist)


# In[5]:


#Now we will need to take information from TIV. So we will need some additional functions
def all_zero(vector):
    for element in vector:
        if element != 0:
            return False
    return True 

def real_imag(TIVector):
    aux = []
    for i in range(0, TIVector.shape[1]):
        real_vector = []
        imag_vector = []
        for j in range(0, TIVector.shape[0]):
            real_vector.append(TIVector[j][i].real)
            imag_vector.append(TIVector[j][i].imag)
        aux.append(real_vector)
        aux.append(imag_vector)
    return np.array(aux)

def tonalIntervalSpace(chroma, symbolic=True):
    centroid_vector = []
    for i in range(0, chroma.shape[1]):
        each_chroma = [chroma[j][i] for j in range(0, chroma.shape[0])]
        each_chroma = np.array(each_chroma)
        if all_zero(each_chroma):
            centroid = [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]
        else:
            tonal = TIV.from_pcp(each_chroma, symbolic)          #Calculate the TIV for each chroma
            #tonal.plot_TIV() #PLOT TIV for each chroma -> too expensive in terms of program's space
            centroid = tonal.get_vector()
        centroid_vector.append(centroid)
    return real_imag(np.array(centroid_vector))


# In[6]:


def harmonic_change(chroma: list, window_size: int=2048, symbolic: bool=True, sigma: int=23, dist: str = 'euclidean'):
    chroma = np.array(chroma).transpose()
    centroid_vector = tonalIntervalSpace(chroma, symbolic=True)

    # Blur
    centroid_vector_blurred = gaussian_blur(centroid_vector, sigma)

    # Harmonic Distance Calculation - Euclidean or Cosine
    harmonic_function = distance_calc(centroid_vector_blurred, dist)

    changes, hcdf_changes = get_peaks_hcdf(harmonic_function, window_size, symbolic=True)

    return changes, hcdf_changes, harmonic_function


# In[7]:


np.set_printoptions(threshold=sys.maxsize)


# In[8]:


path_score_BPS = './Midi_Files/BPS'
path_score_ABC = './Midi_Files/ABC'
file_list_BPS = glob.glob(path_score_BPS + '/*.mid')
file_list_ABC = glob.glob(path_score_ABC + '/*.mid')

m_f = 'M7c06cae3.0PTairQuality.220523.mid'


# # Piano Roll Representations
# 
# In order to represent musical scores for computational analysis, two-dimensional piano roll graphics (with columns being time steps and rows being pitches) are provided. It also gives a sense of pitch variation through all time steps.

# In[9]:


def midi_pianoRoll(file):
    midi_data = pretty_midi.PrettyMIDI(file)
    score = libfmp.c1.midi_to_list(midi_data)
    libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True);
    
####ANOTHER WAY TO PLOT PIANO ROLL GRAPHICS
#def midi_pianoRoll(file):
#    s = music21.converter.parse(file)
#    s.plot('pianoroll')


#for file in file_list_BPS:
#    midi_pianoRoll(file) ###IN CASE WE WANT TO PLOT PIANO ROLL GRAPHICS FOR BPS DATASET

#for file in file_list_ABC:
#    midi_pianoRoll(file) ###IN CASE WE WANT TO PLOT PIANO ROLL GRAPHICS FOR ABC DATASET


# # Chroma Vectors
# 
# Chroma Vectors play an important role in music processing. Although mainly used for audio, it refers the cardinality of each pitch class in symbolic music (where the conotation "Harmonic Pitch Class Profile" is more suitable).
# 
# Given that music21 can't produce this kind of graphic, this was produced by libfmp library [2].
# 
# [2] - Meinard Müller and Frank Zalkow. libfmp: A Python Package for Fundamentals of Music Processing. Journal of Open Source Software (JOSS), 6(63), 2021.

# In[10]:


def chromagram(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    score = libfmp.c1.midi_to_list(midi_data)
    df = pd.DataFrame(score, columns=['Start', 'Duration', 'Pitch', 'Velocity', 'Instrument'])
    array_time = np.array(df[['Start']]) #It's in seconds
    array_pitch = np.array(df[['Pitch']])
    df_array = np.column_stack((array_time, array_pitch))
    chromagram = libfmp.b.b_sonification.list_to_chromagram(score, df_array.shape[0], 1)
    chroma = libfmp.b.b_plot.plot_chromagram(chromagram, xlim = (0, array_time[-1]))
    print(df['Start'].max())

    plt.xlabel("Time (Seconds)")
    plt.ylabel("Pitch")
    plt.title("Chroma Vectors")
    plt.show()

    return chroma

#for file in file_list_BPS:
#    chroma_bps = chromagram(file) 
    
#for file in file_list_ABC:
#    chroma_abc = chromagram(file) 
#    c_a = libfmp.b.b_plot.plot_chromagram(chroma_abc) ###JUST IN CASE WE WANT TO PLOT CHROMAGRAMS ABC DATASET


# For the next step, it will be necessary to transform the file into a suitable 12-element vector, according to TIV dimensions.

# In[11]:


def midi2chroma(m_vector):
    vector = np.zeros((m_vector.shape[0], 12))
    for i, frame in enumerate(m_vector):
        for j, element in enumerate(frame):
            vector[i][j % 12] += element
    return vector

#for file in file_list_BPS:
#    midi_vector = Read_midi(file,28).read_file()
#    for times, key in midi_vector.items():
#        print(key.shape)
#    vec = (list(midi_vector.values()))[0]
#    chroma_vector = midi2chroma(vec)
#    print(chroma_vector.shape)
    
#for file in file_list_ABC:
#    midi_vector = Read_midi(file,23).read_file()
#    for times, key in midi_vector.items():
#        print(key.shape)                        #HAVE A LOT "Try to note off a note that has never been turned on"
#    vec = (list(midi_vector.values()))[0]
#    chroma_vector = midi2chroma(vec)
#    print(chroma_vector.shape)

#chroma = np.array(chroma_vector).transpose()
#print(chroma.shape[1])


# A continuous division of the score to make only the most "strong" note of the chord, in a form of making a tree representation.
# 
# For now, the main idea is the drop all notes apart from the "stronger" one, that occurs at the same time (in time_pitch function). Still thinking in a way to do just that (because we need to discover the "stronger" note on simultaneous notes).

# In[12]:


def time_pitch(file):
    midi_data = pretty_midi.PrettyMIDI(file)
    score = libfmp.c1.midi_to_list(midi_data)
    df = pd.DataFrame(score, columns=['Start', 'Duration', 'Pitch', 'Velocity', 'Instrument'])
    df = df.sort_values(['Start','Pitch'], ascending=[True, True]) # sort the dataframe properly
    lst = list(df.Start)
    return lst
    
#for file in file_list_BPS:
#    keys_list_bps = time_pitch(file)
#    break
    
#for file in file_list_ABC:
#    abc = time_pitch(file)
#    keys_list_abc = list(abc)
#    print(keys_list_abc)


# In[20]:

def HCDF(path, sigma=10, distance='Euclidean'):
    f_measure_results = []
    precision_results = []
    recall_results = []
    if(path == file_list_BPS):
        print("HCDF in BPS Dataset", sigma, distance)
    elif(path == file_list_ABC):
        print("HCDF in ABC Dataset", sigma, distance)
    else:
        print("Not used dataset! Try again")
    for file in path:
        midi_vector = Read_midi(file,28).read_file()
        vec = (list(midi_vector.values()))[0]
        chroma_vector = midi2chroma(vec)
        time_values = time_pitch(file)
        
#         for i in range(0, len(time_values)):
#            for j in range(i+1, len(time_values)):
#                if(time_values[i] > time_values[j]):
#                    temp = time_values[i]    
#                    time_values[i] = time_values[j]    
#                    time_values[j] = temp
         
        changes, hcdf_changes, harmonic_function = harmonic_change(chroma=chroma_vector, symbolic=True, sigma=sigma, dist=distance)
        #print(changes, hcdf_changes, harmonic_function)
        changes = changes/220
        
        changes_ground_truth = np.array(time_values)
        #print(changes_ground_truth)
        
        f_measure, precision, recall = mir_eval.onset.f_measure(changes_ground_truth, changes, window=31.218) #same window than Harte
        f_measure_results.append(f_measure)
        precision_results.append(precision)
        recall_results.append(recall)
        #import pdb; pdb.set_trace()
        #breakpoint()
    return np.mean(np.array(f_measure_results)), np.mean(np.array(precision_results)), np.mean(np.array(recall_results))


# In[14]:


def tune_sigma_plot(evaluation_result):
    sigma_list = []; type_metric = []; metrics = []
    for s, v in evaluation_result.items():
        f, p, r = v
        # F-Measure
        sigma_list.append(s)
        type_metric.append("F_score")
        metrics.append(f)
        # Precision
        sigma_list.append(s)
        type_metric.append("Precision")
        metrics.append(p)
        # Recall
        sigma_list.append(s)
        type_metric.append("Recall")
        metrics.append(r)
    df_dict = {
        "sigma": sigma_list,
        "metric": type_metric,
        "value": metrics
    }

    df = pd.DataFrame(df_dict)
    fig = px.line(df, x="sigma", y="value", color="metric", render_mode="svg")
    fig.show()


# HCDF in BPS Dataset

# In[22]:


HCDF(file_list_BPS, sigma = 30, distance = 'Euclidean')
breakpoint()

# In[ ]:


results_euclidean_BPS = {
    sigma: HCDF(file_list_BPS, sigma=sigma, distance='euclidean') 
    for sigma in range(1, 52, 10)
}

tune_sigma_plot(results_euclidean_BPS)

results_cosine_BPS = {
    sigma: HCDF(file_list_BPS, sigma=sigma, distance='cosine') 
    for sigma in range(1, 52, 10)
}
tune_sigma_plot(results_cosine_BPS)


# HCDF in ABC Dataset

# In[30]:


# HCDF(file_list_ABC, sigma = 6, distance = 'euclidean')


# # In[31]:


# results_euclidean_ABC = {
#     sigma: HCDF(file_list_ABC, sigma=sigma, distance='euclidean') 
#     for sigma in range(1, 52, 10)
# }

# tune_sigma_plot(results_euclidean_ABC)

# results_cosine_ABC = {
#     sigma: HCDF(file_list_ABC, sigma=sigma, distance='cosine') 
#     for sigma in range(1, 52, 10)
# }
# tune_sigma_plot(results_cosine_ABC)

