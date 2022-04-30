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
from libfmp import *
import pickle
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd
from unidecode import unidecode
from vampy import *
!pip install py-midi2csv()

from Read_Symbolic_Notation import *

#Dict = {1: 'beats.xlsx', 2: 'chords.xlsx', 3: 'dBeats.xlsx', 4: 'notes.xlsx', 5: 'phrases.xlsx'}

#2 - Create Pitch Class Profile Vector

path_score = 'C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/AugmentedNet/rawdata/corrections/BPS'

def music21_converterScoreMidi(path_info):
    for file in os.listdir(path_info):
        for file in glob.glob("*.mxl"):
        #if file.endswith(".mxl"):
            sc = stream.Stream(file)
            mf = midi.translate.streamToMidiFile(sc)
            #fp = sc.write('midi', fp=files)
            mf.show('midi')
            
            return mf
    
music21_converterScoreMidi(path_score)

files = os.listdir(path_score)
midi_matrix = []
for file in files:
     if file.endswith('mid'):
        midi_matrix.append(file)

midi_matrix

def get_chroma(midi_vector):
    chroma_vector = []
    chroma_bins = []
    file_list = glob.glob(path_score + '/*.mid', recursive=True)
    for file in file_list:
        midi_object = pretty_midi.PrettyMIDI(file)
        chroma_vector = np.zeros((len(midi_object.instruments),12))
        midi_object.time_signature_changes
        midi_object.get_beats()[0:1000]
        midi_object.instruments[0].notes[0:1000]
        
        #midifile = midi_to_list(midi_object)
        #chroma = libfmp.b.b_sonification.list_to_chromagram(file)
        #chromagram = libfmp.b.b_plot.plot_chromagram(chroma)
        
        chromagram = midi_object.get_chroma()
        plt.imshow(chromagram)

        for c in chroma_vector:
            chroma_bins.append(c['Chroma values'].tolist())
    
    #for i, midi_object in enumerate(midi_vector):
    #    for j, element in enumerate(midi_object):
    #        chroma_vector[i][j % 12] += element
                
    return np.array(chroma_vector).transpose()

chroma_list = get_chroma(midi_matrix)
        