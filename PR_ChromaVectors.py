# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:57:55 2022

@author: José Macedo
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian1DKernel
np.seterr(all='raise')

import music21
import mido
from mido import MidiFile
import mirdata
import os
import sys
import py_midicsv as pm
import glob
import csv
import json
import librosa
import pretty_midi
import libfmp.c1
import libfmp.b
import pickle
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd
from unidecode import unidecode
from vampy import *
from pymidi import *

from Read_Symbolic_Notation import *

#2 - Create Pitch Class Profile Vector/Chroma Vector
path_score_BPS = './Midi_Files/BPS'
path_score_ABC = './Midi_Files/ABC'

def midi_pianoRoll(file):
    midi_data = pretty_midi.PrettyMIDI(file)
    score = libfmp.c1.midi_to_list(midi_data)
    libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True);
    
####ANOTHER WAY TO PLOT PIANO ROLL GRAPHICS
#def midi_pianoRoll(file):
#    s = music21.converter.parse(file)
#    s.plot('pianoroll')

file_list_BPS = glob.glob(path_score_BPS + '/*.mid')
file_list_ABC = glob.glob(path_score_ABC + '/*.mid')

def midi_pianoRoll(file):
    #file_list = glob.glob(path_score + '/*.mid', recursive=True)
    #for file in file_list:
        midi_data = pretty_midi.PrettyMIDI(file)
        score = libfmp.c1.midi_to_list(midi_data)
        libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True);

# def get_chroma(midi_vector):
#     chroma_vector = []
#     chroma_bins = []
#     file_list = glob.glob(path_score + '/*.mid', recursive=True)
#     for file in file_list:
#         midi_object = pretty_midi.PrettyMIDI(file)
#         chroma_vector = np.zeros((len(midi_object.instruments),12))
#         midi_object.time_signature_changes
#         midi_object.get_beats()[0:1000]
#         midi_object.instruments[0].notes[0:1000]
        
#         #midifile = midi_to_list(midi_object)
#         #chroma = libfmp.b.b_sonification.list_to_chromagram(file)
#         #chromagram = libfmp.b.b_plot.plot_chromagram(chroma)
        
#         chromagram = midi_object.get_chroma()
#         plt.imshow(chromagram)

#         for c in chroma_vector:
#             chroma_bins.append(c['Chroma values'].tolist())
    
#     #for i, midi_object in enumerate(midi_vector):
#     #    for j, element in enumerate(midi_object):
#     #        chroma_vector[i][j % 12] += element
                
#     return np.array(chroma_vector).transpose()

# chroma_list = get_chroma(midi_matrix)

def chromagram(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    score = libfmp.c1.midi_to_list(midi_data)
    chromagram = libfmp.b.b_sonification.list_to_chromagram(score, 6000, 120)
    chroma = libfmp.b.b_plot.plot_chromagram(chromagram)

    return chroma

for file in file_list_BPS:
    chroma_bps = chromagram(file) ###JUST IN CASE WE WANT TO PLOT CHROMAGRAMS BPS DATASET
    midi_pianoRoll(file)
    
#for file in file_list_ABC:
#    chroma_abc = chromagram(file) ###JUST IN CASE WE WANT TO PLOT CHROMAGRAMS ABC DATASET
#    midi_pianoRoll(file)

#For the next step, it will be necessary to transform the file into a suitable 12-element vector, according to TIV dimensions.
def midi2chroma(m_vector):
    vector = np.zeros((m_vector.shape[0], 12))
    for i, frame in enumerate(m_vector):
        for j, element in enumerate(frame):
            vector[i][j % 12] += element
    return vector