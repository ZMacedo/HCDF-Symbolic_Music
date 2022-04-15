<<<<<<< HEAD
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
from TIVlib import TIV
import glob
import csv
import json
import librosa
import pickle
import mir_eval
import pandas as pd
from unidecode import unidecode
from vampy import *
# import essentia
# from essentia.standard import HPCP

from Read_Symbolic_Notation import *
from HPCP_Vectors import *

def get_harmonic_change(filename: str, name_file: str, hpss: bool = False, tonal_model: str = 'TIV2',
                        chroma: str = 'nnls',
                        blur: str = 'full', sigma: int = 11, log_compresion: str = 'none', dist: str = 'euclidean'):
    """
        Computes Harmonic Change Detection Function

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        name_file: str
            name of the file that is being computed

        hpss : bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11

        distance: str optional
            type of distance measure used. Types can be "euclidean" for euclidean distance and "cosine" for cosine distance. Default "euclidean".


        Returns
        -------
        list
            harmonic changes (the peaks) on the song detected
        list
            HCDF function values
        number
            windows size
    """
    # audio
    y, sr = audio(filename, name_file, hpss, get_parameters_chroma(chroma)["sr"])

    # chroma
    doce_bins_tuned_chroma = chromagram(hpss, name_file, y, sr, chroma)

    # tonal_model
    centroid_vector = tonal_centroid_transform(hpss, chroma, name_file, y, sr, tonal_model, doce_bins_tuned_chroma)

    # blur
    centroid_vector_blurred = gaussian_blur(hpss, chroma, tonal_model, name_file, centroid_vector, log_compresion, blur,
                                            sigma)

    # harmonic distance and calculate peaks
    harmonic_function = distance_calc(centroid_vector_blurred, dist)
    windows_size = centroids_per_second(y, sr, centroid_vector_blurred)
    changes, centroid_changes = get_peaks_hcdf(harmonic_function, centroid_vector_blurred, 0, windows_size,
                                                centroid_vector)

    return changes, harmonic_function, windows_size, numpy.array(centroid_changes)

def harmonic_change(filename: str, name_file: str, hpss: bool = False, tonal_model: str = 'TIV2', chroma: str = 'cqt',
                    blur: str = 'full', sigma: int = 11, log_compresion: str = 'none', distance: str = 'euclidean'):
    """
        Wrapper of harmonic change detection function for save all results for future same calculations. If parameterization
        have been computed before HCDF is not computed.

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        name_file: str
            name of the file that is being computed

        hpss : bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11

        distance: str optional
            type of distance measure used. Types can be "euclidean" for euclidean distance and "cosine" for cosine distance. Default "euclidean".


        Returns
        -------
        list
            harmonic changes (the peaks) on the song detected
        list
            HCDF function values
        number
            windows size
    """
    centroid_changes = []
    check_parameters(chroma, blur, tonal_model, log_compresion, distance)

    name_harmonic_change = get_name_harmonic_change(name_file, hpss, tonal_model, chroma, blur, sigma, log_compresion,
                                                    distance)
    if path.exists(name_harmonic_change):
        dic = load_binary(name_harmonic_change)
    else:
        changes, harmonic_function, windows_size, centroid_changes = get_harmonic_change(filename, name_file, hpss,
                                                                                          tonal_model, chroma,
                                                                                          blur, sigma, log_compresion,
                                                                                          distance)
        dic = {'changes': changes, 'harmonic_function': harmonic_function, 'windows_size': windows_size}

        save_binary(dic, name_harmonic_change)
    return dic['changes'], dic['harmonic_function'], dic['windows_size']

def get_peaks_hcdf(hcdf_function, c, threshold, rate_centroids_second, centroids):
    changes = [0]
    centroid_changes = [[centroids[j][0] for j in range(0, c.shape[0])]]
    last = 0
    for i in range(2, hcdf_function.shape[0] - 1):
        if hcdf_function[i - 1] < hcdf_function[i] and hcdf_function[i + 1] < hcdf_function[i]:
            centroid_changes.append([np.median(centroids[j][last + 1:i - 1]) for j in range(0, c.shape[0])])
            changes.append(i / rate_centroids_second)
            last = i
    return np.array(changes), centroid_changes

# def evaluate_hcdf_across_haydn_op20(sigma=30, distance='euclidean'):
#     f_measure_results = []
#     precision_results = []
#     recall_results = []
#     print("evaluate_hcdf_across_haydn_op20", sigma, distance)
#     for k, t in h20.load_tracks().items():
#         midi_matrixes = Read_midi(t.midi_path, 28).read_file()
#         mat = list(midi_matrixes.values())
#         midi_quartet = mat[0] + mat[1] + mat[2] + mat[3]
#         chroma_quartets = midi2chroma(midi_quartet)
        
#         changes, hcdf_changes, harmonic_function = harmonic_change(chroma=chroma_quartets, symbolic=True,
#                             sigma=sigma, dist=distance)

#         changes_ground_truth = np.array([c['time'] for c in t.chords])

#         f_measure, precision, recall = mir_eval.onset.f_measure(changes_ground_truth, changes, window=31.218) #same window than Harte
#         # print(t.title, f_measure, precision, recall)
#         f_measure_results.append(f_measure)
#         precision_results.append(precision)
#         recall_results.append(recall)
#     return np.mean(np.array(f_measure_results)), \
#            np.mean(np.array(precision_results)), \
#            np.mean(np.array(recall_results))

# evaluate_hcdf_across_haydn_op20()

# results_euclidean = {
#     sigma: evaluate_hcdf_across_haydn_op20(sigma=sigma, distance='euclidean') 
#     for sigma in range(1, 52, 5)
# }

# def tune_sigma_plot(evaluation_result):
#     sigma_list = []; type_metric = []; metrics = []
#     for s, v in evaluation_result.items():
#         f, p, r = v
#         # f measure
#         sigma_list.append(s)
#         type_metric.append("F_score")
#         metrics.append(f)
#         # Precision
#         sigma_list.append(s)
#         type_metric.append("Precision")
#         metrics.append(p)
#         # Recall
#         sigma_list.append(s)
#         type_metric.append("Recall")
#         metrics.append(r)
#     df_dict = {
#         "sigma": sigma_list,
#         "metric": type_metric,
#         "value": metrics
#     }

#     df = pd.DataFrame(df_dict)
#     fig = px.line(df, x="sigma", y="value", color="metric", render_mode="svg")
#     fig.show()

# """Tuning sigma gaussian hyperparameter for HCDF with euclidean distance."""

# tune_sigma_plot(results_euclidean)

# """Results are better segmenting the chord boundaries that the current aproaches for chord recgnition. With a sigma=20 all the metrics computed across the Haydn op 20 dataset are greater than 70%."""

# results_cosine = {
#     sigma: evaluate_hcdf_across_haydn_op20(sigma=sigma, distance='cosine') 
#     for sigma in range(1, 52, 5)
# }
# tune_sigma_plot(results_cosine)

# """The perrformance of HCDF with the cosine distance is a bit worse than using the euclidean distance."""
=======
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
from TIVlib import TIV
import glob
import csv
import json
import librosa
import pickle
import mir_eval
import pandas as pd
from unidecode import unidecode
from vampy import *
# import essentia
# from essentia.standard import HPCP

from Read_Symbolic_Notation import *
from HPCP_Vectors import *

def get_harmonic_change(filename: str, name_file: str, hpss: bool = False, tonal_model: str = 'TIV2',
                        chroma: str = 'nnls',
                        blur: str = 'full', sigma: int = 11, log_compresion: str = 'none', dist: str = 'euclidean'):
    """
        Computes Harmonic Change Detection Function

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        name_file: str
            name of the file that is being computed

        hpss : bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11

        distance: str optional
            type of distance measure used. Types can be "euclidean" for euclidean distance and "cosine" for cosine distance. Default "euclidean".


        Returns
        -------
        list
            harmonic changes (the peaks) on the song detected
        list
            HCDF function values
        number
            windows size
    """
    # audio
    y, sr = audio(filename, name_file, hpss, get_parameters_chroma(chroma)["sr"])

    # chroma
    doce_bins_tuned_chroma = chromagram(hpss, name_file, y, sr, chroma)

    # tonal_model
    centroid_vector = tonal_centroid_transform(hpss, chroma, name_file, y, sr, tonal_model, doce_bins_tuned_chroma)

    # blur
    centroid_vector_blurred = gaussian_blur(hpss, chroma, tonal_model, name_file, centroid_vector, log_compresion, blur,
                                            sigma)

    # harmonic distance and calculate peaks
    harmonic_function = distance_calc(centroid_vector_blurred, dist)
    windows_size = centroids_per_second(y, sr, centroid_vector_blurred)
    changes, centroid_changes = get_peaks_hcdf(harmonic_function, centroid_vector_blurred, 0, windows_size,
                                                centroid_vector)

    return changes, harmonic_function, windows_size, numpy.array(centroid_changes)

def harmonic_change(filename: str, name_file: str, hpss: bool = False, tonal_model: str = 'TIV2', chroma: str = 'cqt',
                    blur: str = 'full', sigma: int = 11, log_compresion: str = 'none', distance: str = 'euclidean'):
    """
        Wrapper of harmonic change detection function for save all results for future same calculations. If parameterization
        have been computed before HCDF is not computed.

        Parameters
        ----------
        filename: str
                name of the file that is being computed witout format extension

        name_file: str
            name of the file that is being computed

        hpss : bool optional
            true or false depends is harmonic percussive source separation (hpss) block wants to be computed. Default False.

        tonal_model: str optional
            Tonal model block type. "TIV2" for Tonal Interval space focus on audio. "TIV2" for audio. "TIV2_Symb" for symbolic data.
            "tonnetz" for harte centroids aproach. Default TIV2

        chroma: str optional
            "chroma-samplerate-framesize-overlap"
            chroma can be "CQT","NNLS", "STFT", "CENS" or "HPCP"
            samplerate as a number scalar
            frame size as a number scalar
            overlap number that a windows is divided

        sigma: number (scalar > 0) optional
            sigma of gaussian smoothing value. Default 11

        distance: str optional
            type of distance measure used. Types can be "euclidean" for euclidean distance and "cosine" for cosine distance. Default "euclidean".


        Returns
        -------
        list
            harmonic changes (the peaks) on the song detected
        list
            HCDF function values
        number
            windows size
    """
    centroid_changes = []
    check_parameters(chroma, blur, tonal_model, log_compresion, distance)

    name_harmonic_change = get_name_harmonic_change(name_file, hpss, tonal_model, chroma, blur, sigma, log_compresion,
                                                    distance)
    if path.exists(name_harmonic_change):
        dic = load_binary(name_harmonic_change)
    else:
        changes, harmonic_function, windows_size, centroid_changes = get_harmonic_change(filename, name_file, hpss,
                                                                                          tonal_model, chroma,
                                                                                          blur, sigma, log_compresion,
                                                                                          distance)
        dic = {'changes': changes, 'harmonic_function': harmonic_function, 'windows_size': windows_size}

        save_binary(dic, name_harmonic_change)
    return dic['changes'], dic['harmonic_function'], dic['windows_size']

def get_peaks_hcdf(hcdf_function, c, threshold, rate_centroids_second, centroids):
    changes = [0]
    centroid_changes = [[centroids[j][0] for j in range(0, c.shape[0])]]
    last = 0
    for i in range(2, hcdf_function.shape[0] - 1):
        if hcdf_function[i - 1] < hcdf_function[i] and hcdf_function[i + 1] < hcdf_function[i]:
            centroid_changes.append([np.median(centroids[j][last + 1:i - 1]) for j in range(0, c.shape[0])])
            changes.append(i / rate_centroids_second)
            last = i
    return np.array(changes), centroid_changes

>>>>>>> a5479a3e3c66eb2dc1d199376857bda380d39572
