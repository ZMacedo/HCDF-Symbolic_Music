#!/usr/bin/env python
# coding: utf-8

# # Symbolic Harmonic Change Detection in the Tonal Interval Space

# The detection of chord boundaries (onset or beginning or chords in the musical surface) is typically addressed within Music Information Retrieval as harmonic change detection. Existing solutions are known to improve complex systems for automatic chord detection as a preprocessing segmentation stage. This project aims at improving the performance of harmonic change detection by adopting a tree-based representation for reducing the complex structure of symbolic music manifestations to an n-chord representation, targeting the basic underlying triadic structure of Western tonal harmony.

# This is the code implemented to achieve the main objectives referred in the dissertation "Symbolic Harmonic Change Detection in the Tonal Interval Space" by José Macedo, with Gilberto Bernardes as Supervisor and Pedro Ramoneda as Co-Supervisor. 

# In[1]:


from music21 import *
import mido
from mido import MidiFile
import mirdata
import os
import plotly.express as px
import sys
import numpy as np
import glob
import pretty_midi
import libfmp.b
import libfmp.c1
import libfmp.c3
import mir_eval
import py_midicsv as pm
from TIVlib import TIV
import matplotlib.pyplot as plt
import pandas as pd
import csv
from unidecode import unidecode
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter
#np.seterr(all='raise')


# # TIS&TIV - Tonal Interval Space & Tonal Interval Vectors 
# 
# A truncated version of TIV library [1].
# 
# [1] - Ramires, A., Bernardes, G., Davies, M.E., & Serra, X. (2020). TIV.lib: an open-source library for the tonal description of musical audio. ArXiv, abs/2008.11529.

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


# # Auxiliary Functions 
# 
# By Pedro Ramoneda in "Harmonic Change Detection from Musical Audio"

# In[3]:


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


# In[4]:


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


# In[5]:


def harmonic_change(chroma: list, window_size: int=2048, symbolic: bool=True, sigma: int = 5, dist: str = 'euclidean'):
#     pdb.set_trace()
    chroma = np.array(chroma).transpose()
    centroid_vector = tonalIntervalSpace(chroma, symbolic=True)

    # Blur
    centroid_vector_blurred = gaussian_blur(centroid_vector, sigma)

    # Harmonic Distance Calculation - Euclidean or Cosine
    harmonic_function = distance_calc(centroid_vector_blurred, dist)

    changes, hcdf_changes = get_peaks_hcdf(harmonic_function, window_size, symbolic=True)

    return changes, hcdf_changes, harmonic_function


# In[6]:


np.set_printoptions(threshold=sys.maxsize)


# # Piano Roll Representations
# 
# In order to represent musical scores for computational analysis, two-dimensional piano roll graphics (with columns being time steps and rows being pitches) are provided. It also gives a sense of pitch variation through all time steps.

# In[7]:


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

# In[8]:


def chromagram(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    score = libfmp.c1.midi_to_list(midi_data)
    df = pd.DataFrame(score, columns=['Start', 'Duration', 'Pitch', 'Velocity', 'Instrument'])
    array_time = np.array(df[['Start']]) #It's in seconds
    array_pitch = np.array(df[['Pitch']])
    df_array = np.column_stack((array_time, array_pitch))
    chromagram = libfmp.b.b_sonification.list_to_chromagram(score, df_array.shape[0], 1)
    chroma = libfmp.b.b_plot.plot_chromagram(chromagram, xlim = (0, array_time[-1]), figsize=(16, 6))
    print(df['Start'].max())
    
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Pitch")
    plt.title("Chroma Vectors")
    plt.show()

    return chroma

#for file in file_list_BPS:
#    chroma_bps = chromagram(file) 
    
#for file in file_list_ABC:
#    chroma_abc = chromagram(file) ###JUST IN CASE WE WANT TO PLOT CHROMAGRAMS ABC DATASET


# In[47]:


import pdb

def hcdf_changes_gt(csv_file):
    if csv_file.endswith(".xlsx"):
        df = pd.read_excel(csv_file, header=None)
    elif csv_file.endswith(".csv"):
        df = pd.read_csv(csv_file, header=None)
    else:
        print("Not a valid excel format")
    beat_chord_onset = df[0].to_numpy() * 60 / 120
    return beat_chord_onset

def HCDF(file, csv_file, sigma=int, distance='Euclidean', resolution = 28):
    f_measure_results, precision_results, recall_results = [], [], []
    # read midi
    midi_vector = pretty_midi.PrettyMIDI(file, resolution, initial_tempo=120)
    
    # compute chroma     
    chroma_vector = midi_vector.get_chroma(resolution).transpose()
    
    # predicted harmonic changes
    changes, hcdf_changes, harmonic_function = harmonic_change(chroma=chroma_vector, symbolic=True, sigma=sigma, dist = distance)
    changes = changes / resolution
    
    # ground truth harmonic changes
    changes_ground_truth = hcdf_changes_gt(csv_file)
    
    #Plot
    #plt.figure(figsize=(10, 7))
    #plt.plot(hcdf_changes)
    #plt.vlines(x=changes_ground_truth, ymin=0, ymax=max(hcdf_changes), colors='green')
    #plt.title('Changes_GT / Changes')
    
    #  evaluation
    f_measure, precision, recall = mir_eval.onset.f_measure(changes_ground_truth, changes, window=0.628)
    f_measure_results.append(f_measure)
    precision_results.append(precision)
    recall_results.append(recall)
    
    return np.mean(np.array(f_measure_results)), np.mean(np.array(precision_results)), np.mean(np.array(recall_results))

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

def compute_hcdf(lst1, lst2, sigma):
    f_sc_results = []
    prec_results = []
    rec_results = []
    for file, file2 in zip(lst1, lst2):
        # print(file)
        # print(file2)

        hcdf = HCDF(file, file2, sigma=sigma, distance='Euclidean', resolution=28)
        f_sc_results.append(hcdf[0])
        prec_results.append(hcdf[1])
        rec_results.append(hcdf[2])

    return np.mean(np.array(f_sc_results)), np.mean(np.array(prec_results)), np.mean(np.array(rec_results))

def results(lst1,lst2):
    for file, file2 in zip(lst1, lst2):
        results_euclidean = {
            sigma: HCDF(file, file2, sigma=sigma, distance='Euclidean') for sigma in range(0, 50, 10)}
        results_cosine = {
            sigma: HCDF(file, file2, sigma=sigma, distance='Cosine') for sigma in range(0, 50, 10)}
    return results_euclidean, results_cosine

# # HCDF in BPS Dataset
path_score_BPS = './Datasets/BPS'
file_list_BPS = glob.glob(path_score_BPS + '/*.mid')
file_csv_BPS = glob.glob(path_score_BPS + '/*.xlsx')

lst1_bps = list()
lst2_bps = list()
for file in file_list_BPS:
    lst1_bps.append(file)
for file in file_csv_BPS:
    lst2_bps.append(file)

f_sc_bps, p_bps, r_bps = compute_hcdf(lst1_bps,lst2_bps, 10)
print("BPS")
print(f_sc_bps, p_bps, r_bps)

results_euclidean_BPS, results_cosine_BPS = results(lst1_bps,lst2_bps)

tune_sigma_plot(results_euclidean_BPS)
tune_sigma_plot(results_cosine_BPS)


# # # HCDF in Tavern Dataset
# # TAVERN consists of three types of files for each musical phrase for each annotator (A and B)

path_Tavern = './Datasets/Tavern'
lst_midi_beethoven = list()
lst_midi_mozart = list()
for file in glob.glob(path_Tavern + './Beethoven/*.mid'):
    lst_midi_beethoven.append(file)
for file in glob.glob(path_Tavern + './Mozart/*.mid'):
    lst_midi_mozart.append(file)

lst_csv_beethovenA = list()
lst_csv_beethovenB = list()
lst_csv_mozartA = list()
lst_csv_mozartB = list()

for file in glob.glob(path_Tavern + './Beethoven/*A.csv'):
    lst_csv_beethovenA.append(file)
for file in glob.glob(path_Tavern + './Beethoven/*B.csv'):
    lst_csv_beethovenB.append(file)
for file in glob.glob(path_Tavern + './Mozart/*A.csv'):
    lst_csv_mozartA.append(file)
for file in glob.glob(path_Tavern + './Mozart/*B.csv'):
    lst_csv_mozartB.append(file)

# #Beethoven with Annotator A
f_sc_beethovenA, p_beethovenA, r_beethovenA = compute_hcdf(lst_midi_beethoven,lst_csv_beethovenA, 10)
print("Beethoven with Annotator A:")
print(f_sc_beethovenA, p_beethovenA, r_beethovenA)

results_euclidean_TAVERN_Beethoven_A, results_cosine_TAVERN_Beethoven_A= results(lst_midi_beethoven,lst_csv_beethovenA)

tune_sigma_plot(results_cosine_TAVERN_Beethoven_A)
tune_sigma_plot(results_cosine_TAVERN_Beethoven_A)

# #Beethoven with Annotator B
f_sc_beethovenB, p_beethovenB, r_beethovenB = compute_hcdf(lst_midi_beethoven,lst_csv_beethovenB, 10)
print("Beethoven with Annotator B:")
print(f_sc_beethovenB, p_beethovenB, r_beethovenB)

results_euclidean_TAVERN_Beethoven_B, results_cosine_TAVERN_Beethoven_B= results(lst_midi_beethoven,lst_csv_beethovenB)

tune_sigma_plot(results_cosine_TAVERN_Beethoven_A)
tune_sigma_plot(results_cosine_TAVERN_Beethoven_A)

# #Mozart with Annotator A
f_sc_mozartA, p_mozartA, r_mozartA = compute_hcdf(lst_midi_mozart,lst_csv_mozartA, 10)
print("Mozart with Annotator A:")
print(f_sc_mozartA, p_mozartA, r_mozartA)

results_euclidean_TAVERN_MozartA, results_cosine_TAVERN_MozartA= results(lst_midi_mozart,lst_csv_mozartA)

tune_sigma_plot(results_euclidean_TAVERN_MozartA)
tune_sigma_plot(results_cosine_TAVERN_MozartA)

# #Mozart with Annotator B
f_sc_mozartB, p_mozartB, r_mozartB = compute_hcdf(lst_midi_mozart,lst_csv_mozartB, 10)
print("Mozart with Annotator B:")
print(f_sc_mozartB, p_mozartB, r_mozartB)

results_euclidean_TAVERN_MozartB, results_cosine_TAVERN_MozartB= results(lst_midi_mozart,lst_csv_mozartB)

tune_sigma_plot(results_euclidean_TAVERN_MozartB)
tune_sigma_plot(results_cosine_TAVERN_MozartB)


# # # HCDF in Bach's Preludes (First Book of Well Tempered Clavier Preludes)
path_Bach_Preludes = './Datasets/Bach_Preludes'
midi_bach = list()
csv_bach = list()
for file in glob.glob(path_Bach_Preludes + './*.mid'):
    midi_bach.append(file)
for file in glob.glob(path_Bach_Preludes + './*.csv'):
    csv_bach.append(file)

f_sc_bach_preludes, p_bach_preludes, r_bach_preludes = compute_hcdf(midi_bach,csv_bach, 10)
print("Bach Preludes")
print(f_sc_bach_preludes, p_bach_preludes, r_bach_preludes)

results_euclidean_Bach_Prelude, results_cosine_Bach_Prelude= results(midi_bach,csv_bach)

tune_sigma_plot(results_euclidean_Bach_Prelude)
tune_sigma_plot(results_cosine_Bach_Prelude)


# # # HCDF with Beethoven Quartets (ABC Dataset)
path_ABC_Beethoven_Quartets = './Datasets/ABC(Beethoven_Quartets)'
midi_beeQ = list()
csv_beeQ = list()
for file in glob.glob(path_ABC_Beethoven_Quartets + './*.mid'):
    midi_beeQ.append(file)
for file in glob.glob(path_ABC_Beethoven_Quartets + './*.csv'):
    csv_beeQ.append(file)

f_sc_beeQ, p_beeQ, r_beeQ = compute_hcdf(midi_beeQ,csv_beeQ, 10)
print("Beethoven Quartets (ABC)")
print(f_sc_beeQ, p_beeQ, r_beeQ)

results_euclidean_Beethoven_Quartets, results_cosine_Beethoven_Quartets = results(midi_beeQ,csv_beeQ)

tune_sigma_plot(results_euclidean_Beethoven_Quartets)
tune_sigma_plot(results_cosine_Beethoven_Quartets)

# # # HCDF in Haydn20 Dataset
# path_kern = './Datasets/Haydn_Op20/op20'
# file_list_kern = glob.glob(path_kern + './***/**/*.krn')

# for file in file_list_kern:
#     if file.endswith('_tsroot.krn'):
#         continue
#     else:
#         print(file)
#         s = converter.parse(file)
#         print(s.duration)
#         #s.write("txt", str(file) + '_TXT.txt')
#         #s.write("midi", str(file) + '_MIDI.mid')
#         #csv = pm.midi_to_csv(str(file) + '_MIDI.mid')
#         #with open(str(file) + '_CSV.csv', "w") as f:
#         #    f.writelines(csv)
#         #with open(str(file) + '_time_onsets.csv', "w") as f:
#         #    f.writelines(csv)

# def csv_time_pitch(csv_file):
#     df = pd.read_csv(csv_file, header=None, sep='\n')
#     df = df[0].str.split(',', expand=True)
#     #df = df[0].apply(str).astype(str).str.replace(',', '')
#     df[1] = pd.to_numeric(df[1], errors='coerce').div(1024)
#     df1 = df[df[2].str.contains("Note_on") == True]
#     #df1 = df1.filter(items=[1, 4]) #If we want Time and Pitch
#     df_time = df1.filter(items=[1]) #Just onset times
#     return df_time

# midi_haydn20 = list()
# csv_haydn20 = list()
# for file in glob.glob(path_kern + './*.mid'):
#     midi_haydn20.append(file)
# for csv_file in glob.glob(path_kern + './***/**/*time_onsets.csv'):
#     csv = csv_time_pitch(csv_file)
#     csv_f = csv.to_csv(str(csv_file), index=False, header=False)
#     csv_haydn20.append(csv_f)

# f_sc_haydn20, p_haydn20, r_haydn20 = compute_hcdf(midi_haydn20,csv_haydn20)
# print("Haydn20")
# print(f_sc_haydn20, p_haydn20, r_haydn20)

#results_euclidean_Haydn20, results_cosine_Haydn20 = results(midi_haydn20,csv_haydn20)

#tune_sigma_plot(results_euclidean_Haydn20)
#tune_sigma_plot(results_cosine_Haydn20)