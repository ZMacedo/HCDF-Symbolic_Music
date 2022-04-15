# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:57:55 2022

@author: José Macedo
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# from astropy.convolution import convolve, Gaussian1DKernel
np.seterr(all='raise')

import os
import librosa
import vampy
import py_midicsv as pm

#Dict = {1: 'beats.xlsx', 2: 'chords.xlsx', 3: 'dBeats.xlsx', 4: 'notes.xlsx', 5: 'phrases.xlsx'}

#2 - Create Pitch Class Profile Vector
#NNLS
#chroma_path = "C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Chroma_Datasets/NNLS_Features/Cross-Era_Datasets/cross-era_chroma-nnls"
def midi2chroma(MIDI_file):
    chroma = np.zeros((MIDI_file.shape[0], 12))
    for i, MIDI_frame in enumerate(MIDI_file):
        for j, frame in enumerate(MIDI_frame):
            chroma[i][j % 12] += frame
    return chroma

chroma_path = "C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Chroma_Datasets/JAAH_features"
def get_NNLS(chroma_path):
    chroma_vector = []
    global csv
    file_list = os.listdir(chroma_path)
    #print(file_list)
    for file in file_list:
        if file.endswith('.csv') in file_list:
            with open(file, 'r') as csv_file:
                #csv = csv.reader(csv_file)
                midi_object = map(int,pm.csv_to_midi(csv_file))
                data, rate = librosa.load(midi_object)
                chroma = vampy.collect(data, rate, "nnls-chroma:nnls-chroma")
                #chroma = list(vamp.process_audio(y, sr, plugin, output="chroma", block_size=fr, step_size=off))
                stepsize, chromadata = chroma["Chroma"]
                plt.imshow(chromadata)
                plt.show()
                chroma_vector.append(chroma)
    
    chroma_bins = []
    for c_bins in chroma_vector:
        chroma_bins.append(c_bins['Chroma Values from NNLS'].tolist())
        
        print(chroma_bins)
    return np.array(chroma_bins), np.array(chroma_vector)

get_NNLS(chroma_path)

def get_NNLS_midi2chroma(chroma_path):
    chroma_vector = []
    for files in os.listdir(chroma_path):
        for file in files:
            with open(file, 'r') as csv:
                midi_object = pm.csv_to_midi(csv)
                chroma = midi2chroma(midi_object)
                #print(chroma)
                chroma_vector.append(chroma)
    return np.array(chroma_vector)

def get_NNLS_STFT(chroma_path):
    chroma_vector = []
    for files in os.walk(chroma_path):
        #print(files)
        for file in files:
            with open(file, 'r') as csv:
                midi_object = pm.csv_to_midi(csv)
                y, sr = librosa.load(midi_object)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                print(chroma)
                chroma_vector.append(chroma)
    return np.array(chroma_vector)

get_NNLS(chroma_path)

# def get_hpcp(x, sr, n_bins=12, f_min=55, f_ref=440.0, min_magn=-100):
#     #Based on code from https://python.hotexamples.com/pt/examples/vamp/-/collect/python-collect-function-examples.html
#     """Compute HPCP features from raw audio using the HPCP Vamp plugin.
#     Vamp, vamp python module and plug-in must be installed.
    
#     Args:
#         x (1d-array): audio signal, mono
#         sr (int): sample rate
#         n_bins (int): number of chroma bins
#         f_min (float): minimum frequency
#         f_ref (float): A4 tuning frequency
#         min_magn (float): minimum magnitude for peak detection, in dB
        
#     Returns:
#         1d-array: time vector
#         2d-array: HPCP features
#     """

#     plugin = 'vamp-hpcp-mtg:MTG-HPCP'
#     params = {'LF': f_min, 'nbins': n_bins, 'reff0': f_ref,
#               'peakMagThreshold': min_magn}
    
#     data = vamp.collect(x, sr, plugin, parameters=params)
#     vamp_hop, hpcp = data['matrix']
    
#     t = float(vamp_hop) * (8 + np.arange(len(hpcp)))
    
#     return t, hpcp
