# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:24:04 2022

@author: José Macedo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.spatial.distance import cosine, euclidean
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
import pickle
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd
from unidecode import unidecode
from vampy import *

from Read_Symbolic_Notation import *
from HPCP_Vectors import *
from HCDF_Macedo import *

#METEDOLOGY:
    #1 - Read input symbolic files 
    #2 - Create Pitch Class Profile Vector (the same as Chroma Vector, but this is for audio)
    #3 - Create a class for everything related to TIS and TIV
    #4 - Calculate Distance Calculation for each case (cosine and euclidean calculation)
    #5 - Calculate HCDF
    #6 - Obtain results
    
# path = "C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/NEW_BPS-FH_Dataset"
# time = time_info(path)
# chords = chord_info(path)

chroma_path = "C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Chroma_Datasets/NNLS_Features/Cross-Era_Datasets/cross-era_chroma-nnls"
#NNLS = get_NNLS(chroma_path)

#3 - We will need a class for everything related with TIS/TIV, largely based on TIVlib (Ramires, António, et al. "TIV. lib: an open-source library for the tonal description of musical audio." arXiv preprint arXiv:2008.11529 (2020).)
class TonalIntervalVector:
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
    
    # @classmethod
    # def from_pcp(cls, pcp):
    #     """
    #     Get TIVs from pcp, as the original method
    #     :param pcp: 12xN vector containing N pcps
    #     :return: TIVCollection object
    #     """
    #     if pcp.shape[0] != 12:
    #         raise TypeError("Vector is not compatible with PCP")
    #     fft = np.fft.rfft(pcp, n=12, axis=0)
    #     if fft.ndim == 1:
    #         fft = fft[:, np.newaxis]
    #     energy = fft[0, :] + epsilon
    #     vector = fft[1:7, :]
    #     vector = ((vector / energy) * np.array(cls.weights)[:, np.newaxis])
    #     return cls([TIV(energy[i], vector[:, i]) for i in range(len(energy))])
    
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
            distance = TonalIntervalVector.euclidean(tiv_array[i + 1], tiv_array[i])
            results.append(distance)
        return results

    @classmethod
    def euclidean(cls, tiv1, tiv2):
        return np.linalg.norm(tiv1.vector - tiv2.vector)

    @classmethod
    def cosine(cls, tiv1, tiv2):
        a = np.concatenate((tiv1.vector.real, tiv1.vector.imag), axis=0)
        b = np.concatenate((tiv2.vector.real, tiv2.vector.imag), axis=0)
        if all_zero(a) or all_zero(b):
            distance_computed = euclidean(a, b)
        else:
            distance_computed = cosine(a, b)
        return distance_computed

#Now we will need to take information from TIS. So we will need some additional functions
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

def TonalIntervalSpace(chroma, symbolic=True):
    centroid_vector = []
    for i in range(0, chroma.shape[1]):
        each_chroma = [chroma[j][i] for j in range(0, chroma.shape[0])]
        # print(each_chroma)
        if all_zero(each_chroma):
            centroid = [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]
        else:
            tonal = TonalIntervalVector.from_pcp(each_chroma, symbolic)
            centroid = tonal.get_vector()
        centroid_vector.append(centroid)
    return real_imag(np.array(centroid_vector))

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
            
beatles = mirdata.initialize('beatles')
#beatles.download()
#beatles.validate()
track = beatles.load_tracks()
#beatles.choice_track().chords
matrix_MIDI = Info_MIDI(track, 28).read_file(path) #We have to discover the best sigma value (for audio is 28)
mat = list(matrix_MIDI.values())
midi_mat = mat[0] + mat[1] + mat[2] + mat[3]
midi_mat.shape

np.set_printoptions(threshold=sys.maxsize)

chroma_mat = midi2chroma(midi_mat)

changes, hcdf_changes, harmonic_function = harmonic_change(chroma=chroma_mat, symbolic=True,
                         sigma=28, dist='euclidean')
changes

changes_ground_truth = np.array([c['time'] for c in track.chords])
changes_ground_truth

f_measure, precision, recall = mir_eval.onset.f_measure(changes_ground_truth, changes, window=31.218) #same window than Harte
f_measure, precision, recall