from music21 import *
from collections import Counter
import plotly.express as px
import sys
import numpy as np
import glob
import pretty_midi
from scipy import spatial
import mir_eval
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter

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

np.set_printoptions(threshold=sys.maxsize)

#Auxilary Functions

#Auxiliary TIV Functions:
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

def harmonic_change(chroma: list, window_size: int=2048, symbolic: bool=True, sigma: int = 5, dist: str = 'euclidean'):
    chroma = np.array(chroma).transpose()
    centroid_vector = tonalIntervalSpace(chroma, symbolic=True)

    # Blur
    centroid_vector_blurred = gaussian_blur(centroid_vector, sigma)

    # Harmonic Distance Calculation - Euclidean or Cosine
    harmonic_function = distance_calc(centroid_vector_blurred, dist)

    changes, hcdf_changes = get_peaks_hcdf(harmonic_function, window_size, symbolic=True)

    return changes, hcdf_changes, harmonic_function

#Auxiliary HCDF Functions:

#By Pedro Ramoneda in "Harmonic Change Detection from Musical Audio"
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

#Distance Calculation (Euclidean and Cosine)
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

#Reduction Functions:
# Just to order list of TIV_Redux
def sort_dist(dist):
    return dist[1]

def dist_calc(pitches_chord, dist):
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

#NOW WITHOUT DUPLICATED NOTES ON THE CHORD:
def non_duplicated_stream(stream):
  for chord in stream.flatten().getElementsByClass('Chord'):
      chord.removeRedundantPitchNames(inPlace=True)
  return stream
    
#The aim is to only show the notes that are going to be cut by the reduction step
#(Taking into account that the representation of each chord are only up to 3 notes maximum)
def TIV_Redux(pitches_chord, dist):
    dist_list = dist_calc(pitches_chord, dist)
    dist_list.sort(key=sort_dist)
    dist_reduced = dist_list[3:] #only show the notes that are going to be cut, reducing the chord
    return dist_reduced

#Reduction Process
def info_chord(stream):
  midi_pitches = []
  duration_pitches = []
  offset_pitches = []

  for chord in stream.flatten().getElementsByClass('Chord'):
    midi_pitches.append([n.pitch.midi for n in chord.notes])
    duration_pitches.append([n.quarterLength for n in chord.notes])
    offset_pitches.append([chord.offset for n in chord])
  
  print(midi_pitches)
  print(duration_pitches)
  print(offset_pitches)
  return midi_pitches, duration_pitches, offset_pitches

def red_notes_quantifier(lst):
  reduced_notes_lst = []
  for s in lst:
    red_notes = TIV_Redux(s, spatial.distance.euclidean)
    e = [x[0] for x in red_notes]
    n_name = [note.Note(s[i]).nameWithOctave for i in e]
    if n_name:
        reduced_notes_lst.append(n_name)
  return reduced_notes_lst

def times_info(offset_pitches):
  times_offset = []
  for value in offset_pitches:
     if len(value) >= 4:
         times_offset.append(value[0])
  return times_offset

def stream_painted_notes(stream1 ,notes_lst, times_offset):
  final_reduced_list = zip(notes_lst, times_offset)
  zipped_reduced_list = list(final_reduced_list)
  zipped_reduced_list.sort(key=lambda x: x[1])

  for i in range(len(zipped_reduced_list)):
    for j in range(len(zipped_reduced_list[i][0])):
      if len(zipped_reduced_list[i][0]) > 0:
        note_zipped = note.Note(zipped_reduced_list[i][0][j])
        note_zipped.style.color = 'red'
        stream1.insert(zipped_reduced_list[i][1], note_zipped)

  stream1 = stream1.flat
  return stream1

def reduced_stream(stream):
  for chord in stream.flatten().getElementsByClass('Chord'):
    chord_new = chord
    n = [n.pitch.midi for n in chord_new.notes]
    n_pitch = [nt for nt in chord_new.pitches] #All pitches from each chord
    red_chord = TIV_Redux(n, spatial.distance.euclidean) #Putting TIV on the equation
    e = [x[0] for x in red_chord]
    n_to_cut = [n_pitch[i] for i in e] #The pitches of notes that are going to be cutted from the chord
    for element in n_to_cut:
      if element in n_pitch:
        for i in chord_new.pitches:
          if i == element:
            chord_new.remove(i)
  stream.replace(chord,chord_new)
  return stream

#Functions for computing HCDF
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

    # Read Midi File
    midi_vector = pretty_midi.PrettyMIDI(file, resolution, initial_tempo=120)
    
    # Compute chroma
    chroma_vector = midi_vector.get_chroma(resolution).transpose()
    
    # Predicted harmonic changes
    changes, hcdf_changes, harmonic_function = harmonic_change(chroma=chroma_vector, symbolic=True, sigma=sigma, dist = distance)
    changes = changes / resolution
    
    # Ground truth harmonic changes
    changes_ground_truth = hcdf_changes_gt(csv_file)
    
    #Plot
    #plt.figure(figsize=(10, 7))
    #plt.plot(hcdf_changes)
    #plt.vlines(x=changes_ground_truth, ymin=0, ymax=max(hcdf_changes), colors='green')
    #plt.title('Changes_GT / Changes')
    
    # Evaluation
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

def compute_hcdf(lst1, lst2, distance, sigma, resolution):
    f_sc_results = []
    prec_results = []
    rec_results = []
    for file, file2 in zip(lst1, lst2):
        hcdf = HCDF(file, file2, sigma=sigma, distance=distance, resolution=resolution)
        f_sc_results.append(hcdf[0])
        prec_results.append(hcdf[1])
        rec_results.append(hcdf[2])

    return np.mean(np.array(f_sc_results)), np.mean(np.array(prec_results)), np.mean(np.array(rec_results))

def results(lst1,lst2, resolution):
    for file, file2 in zip(lst1, lst2):
        results_euclidean = {
            sigma: HCDF(file, file2, sigma=sigma, distance='Euclidean',resolution = resolution) for sigma in range(0, 50, 10)}
        results_cosine = {
            sigma: HCDF(file, file2, sigma=sigma, distance='Cosine',resolution = resolution) for sigma in range(0, 50, 10)}
    return results_euclidean, results_cosine

# # HCDF in BPS Dataset
path_score_BPS = './Chordify/BPS'
file_list_BPS = glob.glob(path_score_BPS + '/*.mid')
file_csv_BPS = glob.glob(path_score_BPS + '/*.xlsx')

lst1_bps = list()
lst2_bps = list()
for file in file_list_BPS:
    lst1_bps.append(file)
for file in file_csv_BPS:
    lst2_bps.append(file)

print("BPS - Euclidean")
f_sc_bps_e, p_bps_e, r_bps_e = compute_hcdf(lst1_bps,lst2_bps, 'Euclidean', 10,resolution = 28)
print(f_sc_bps_e, p_bps_e, r_bps_e)
print("BPS - Cosine")
f_sc_bps_c, p_bps_c, r_bps_c = compute_hcdf(lst1_bps,lst2_bps, 'Cosine', 10,resolution = 28)
print(f_sc_bps_c, p_bps_c, r_bps_c)

results_euclidean_BPS, results_cosine_BPS = results(lst1_bps,lst2_bps,resolution = 28)

tune_sigma_plot(results_euclidean_BPS)
tune_sigma_plot(results_cosine_BPS)


# # # HCDF in Tavern Dataset
# # TAVERN consists of three types of files for each musical phrase for each annotator (A and B)
path_Tavern = './Chordify/Tavern'
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
print("Beethoven with Annotator A - Euclidean")
f_sc_beethovenA_e, p_beethovenA_e, r_beethovenA_e = compute_hcdf(lst_midi_beethoven,lst_csv_beethovenA, 'Euclidean', 10,resolution = 28)
print(f_sc_beethovenA_e, p_beethovenA_e, r_beethovenA_e)
print("Beethoven with Annotator A - Cosine")
f_sc_beethovenA_c, p_beethovenA_c, r_beethovenA_c = compute_hcdf(lst_midi_beethoven,lst_csv_beethovenA, 'Cosine', 10,resolution = 28)
print(f_sc_beethovenA_c, p_beethovenA_c, r_beethovenA_c)

results_euclidean_TAVERN_Beethoven_A, results_cosine_TAVERN_Beethoven_A= results(lst_midi_beethoven,lst_csv_beethovenA,resolution = 28)

tune_sigma_plot(results_cosine_TAVERN_Beethoven_A)
tune_sigma_plot(results_cosine_TAVERN_Beethoven_A)

# #Beethoven with Annotator B
print("Beethoven with Annotator B - Euclidean")
f_sc_beethovenB_e, p_beethovenB_e, r_beethovenB_e = compute_hcdf(lst_midi_beethoven,lst_csv_beethovenB, 'Euclidean', 10,resolution = 28)
print(f_sc_beethovenB_e, p_beethovenB_e, r_beethovenB_e)
print("Beethoven with Annotator B - Cosine")
f_sc_beethovenB_c, p_beethovenB_c, r_beethovenB_c = compute_hcdf(lst_midi_beethoven,lst_csv_beethovenB, 'Cosine', 10,resolution = 28)
print(f_sc_beethovenB_c, p_beethovenB_c, r_beethovenB_c)

results_euclidean_TAVERN_Beethoven_B, results_cosine_TAVERN_Beethoven_B= results(lst_midi_beethoven,lst_csv_beethovenB,resolution = 28)

tune_sigma_plot(results_cosine_TAVERN_Beethoven_B)
tune_sigma_plot(results_cosine_TAVERN_Beethoven_B)

# #Mozart with Annotator A
print("Mozart with Annotator A - Euclidean")
f_sc_mozartA_e, p_mozartA_e, r_mozartA_e = compute_hcdf(lst_midi_mozart,lst_csv_mozartA, 'Euclidean', 10,resolution = 28)
print(f_sc_mozartA_e, p_mozartA_e, r_mozartA_e)
print("Mozart with Annotator A - Cosine")
f_sc_mozartA_c, p_mozartA_c, r_mozartA_c = compute_hcdf(lst_midi_mozart,lst_csv_mozartA, 'Cosine', 10,resolution = 28)
print(f_sc_mozartA_c, p_mozartA_c, r_mozartA_c)

results_euclidean_TAVERN_MozartA, results_cosine_TAVERN_MozartA= results(lst_midi_mozart,lst_csv_mozartA,resolution = 28)

tune_sigma_plot(results_euclidean_TAVERN_MozartA)
tune_sigma_plot(results_cosine_TAVERN_MozartA)

# #Mozart with Annotator B
print("Mozart with Annotator B - Euclidean")
f_sc_mozartB_e, p_mozartB_e, r_mozartB_e = compute_hcdf(lst_midi_mozart,lst_csv_mozartB, 'Euclidean', 10,resolution = 28)
print(f_sc_mozartB_e, p_mozartB_e, r_mozartB_e)
print("Mozart with Annotator B - Cosine")
f_sc_mozartB_c, p_mozartB_c, r_mozartB_c = compute_hcdf(lst_midi_mozart,lst_csv_mozartB, 'Cosine', 10,resolution = 28)
print(f_sc_mozartB_c, p_mozartB_c, r_mozartB_c)

results_euclidean_TAVERN_MozartB, results_cosine_TAVERN_MozartB= results(lst_midi_mozart,lst_csv_mozartB,resolution = 28)

tune_sigma_plot(results_euclidean_TAVERN_MozartB)
tune_sigma_plot(results_cosine_TAVERN_MozartB)


# # # HCDF in Bach's Preludes (First Book of Well Tempered Clavier Preludes)
path_Bach_Preludes = './Chordify/Bach_Preludes'
midi_bach = list()
csv_bach = list()
for file in glob.glob(path_Bach_Preludes + './*.mid'):
    midi_bach.append(file)
for file in glob.glob(path_Bach_Preludes + './*.csv'):
    csv_bach.append(file)

print("Bach Preludes - Euclidean")
f_sc_bach_preludes_e, p_bach_preludes_e, r_bach_preludes_e = compute_hcdf(midi_bach,csv_bach, 'Euclidean', 10,resolution = 28)
print(f_sc_bach_preludes_e, p_bach_preludes_e, r_bach_preludes_e)
print("Bach Preludes - Cosine")
f_sc_bach_preludes_c, p_bach_preludes_c, r_bach_preludes_c = compute_hcdf(midi_bach,csv_bach, 'Cosine', 10,resolution = 28)
print(f_sc_bach_preludes_c, p_bach_preludes_c, r_bach_preludes_c)

results_euclidean_Bach_Prelude, results_cosine_Bach_Prelude= results(midi_bach,csv_bach,resolution = 28)

tune_sigma_plot(results_euclidean_Bach_Prelude)
tune_sigma_plot(results_cosine_Bach_Prelude)


# # # HCDF with Beethoven Quartets (ABC Dataset)
path_ABC_Beethoven_Quartets = './Chordify/ABC_(Beethoven_Quartets)'
midi_beeQ = list()
csv_beeQ = list()
for file in glob.glob(path_ABC_Beethoven_Quartets + './*.mid'):
    midi_beeQ.append(file)
for file in glob.glob(path_ABC_Beethoven_Quartets + './*.csv'):
    csv_beeQ.append(file)

print("Beethoven Quartets (ABC) - Euclidean")
f_sc_beeQ_e, p_beeQ_e, r_beeQ_e = compute_hcdf(midi_beeQ,csv_beeQ, 'Euclidean', 10,resolution = 28)
print(f_sc_beeQ_e, p_beeQ_e, r_beeQ_e)
print("Beethoven Quartets (ABC) - Cosine")
f_sc_beeQ_c, p_beeQ_c, r_beeQ_c = compute_hcdf(midi_beeQ,csv_beeQ, 'Cosine', 10,resolution = 28)
print(f_sc_beeQ_c, p_beeQ_c, r_beeQ_c)

results_euclidean_Beethoven_Quartets, results_cosine_Beethoven_Quartets = results(midi_beeQ,csv_beeQ,resolution = 28)

tune_sigma_plot(results_euclidean_Beethoven_Quartets)
tune_sigma_plot(results_cosine_Beethoven_Quartets)