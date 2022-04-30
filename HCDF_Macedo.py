# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:24:08 2022

@author: HP
"""

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
!pip install py-midi

# import essentia
# from essentia.standard import HPCP

from Read_Symbolic_Notation import *
from HPCP_Vectors import *

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

def harmonic_change(chroma: list, window_size: int=2048, symbolic: bool=True, 
                         sigma: int = 5, dist: str = 'euclidean'):
    chroma = np.array(chroma).transpose()
    centroid_vector = tonalIntervalSpace(chroma, symbolic=symbolic)

    # blur
    centroid_vector_blurred = gaussian_blur(centroid_vector, sigma)

    # harmonic distance and calculate peaks
    harmonic_function = distance_calc(centroid_vector_blurred, dist)

    changes, hcdf_changes = get_peaks_hcdf(harmonic_function, window_size, symbolic)

    return changes, hcdf_changes, harmonic_function

changes, hcdf_changes, harmonic_function = harmonic_change(chroma=chroma_list, symbolic=True, sigma=28, dist='euclidean')
changes

changes_ground_truth = np.array([m['time'] for m in midi_matrix])
changes_ground_truth

f_measure, precision, recall = mir_eval.onset.f_measure(changes_ground_truth, changes, window=31.218) #same window than Harte
f_measure, precision, recall