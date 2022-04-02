# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:57:55 2022

@author: José Macedo
"""
import numpy as np
#import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# from astropy.convolution import convolve, Gaussian1DKernel
np.seterr(all='raise')

import os
import vampy
import py_midicsv as pm
import matplotlib.pyplot as plt

#Dict = {1: 'beats.xlsx', 2: 'chords.xlsx', 3: 'dBeats.xlsx', 4: 'notes.xlsx', 5: 'phrases.xlsx'}

#2 - Create Pitch Class Profile Vector
#NNLS
def get_NNLS(MIDI_path):
    chroma = []
    for subdir, dirs, files in os.walk(MIDI_path):
        for file in files:
            midi_object, rate = pm.csv_to_midi('chroma-nnls_orchestra_romantic.csv')
            chroma = vampy.collect(midi_object, rate, "nnls-chroma:nnls-chroma")
            #chroma = list(vamp.process_audio(y, sr, plugin, output="chroma", block_size=fr, step_size=off))
            stepsize, chromadata = chroma["matrix"]
            plt.imshow(chromadata)
            plt.show()
    
    chroma_bins = []
    for c_bins in chroma:
        chroma_bins.append(c_bins['Chroma Values from NNLS'].tolist())
        
        #print(chroma_bins)
    return np.array(chroma_bins)

def midi2chroma(MIDI_file):
    chroma = np.zeros((MIDI_file.shape[0], 12))
    for i, MIDI_frame in enumerate(MIDI_file):
        for j, frame in enumerate(MIDI_frame):
            chroma[i][j % 12] += frame
    return chroma

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

get_NNLS('C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Chroma Datasets/NNLS_Features/Cross-Era_Datasets/cross-era_chroma-nnls')