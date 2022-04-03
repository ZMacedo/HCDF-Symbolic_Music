# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import csv
from mido import MidiFile
import os
import py_midicsv as pm
import numpy as np
from music21 import *
from unidecode import unidecode
np.seterr(all='raise')

dict1 = {1: 'beats.xlsx', 2: 'chords.xlsx', 3: 'dBeats.xlsx', 4: 'notes.xlsx', 5: 'phrases.xlsx'}

# 1 - Read symbolic inputs (or .csv files with annotations)
# def load_dataset():
#     rootdir = 'C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/NEW_BPS-FH_Dataset'
#     for subdir, dirs, files in os.walk(rootdir):
#         for file in files:
#             convert_to_csv(os.path.join(subdir, file), str(file))
            
#             print(os.path.join(subdir, file))
        
#     #     folders = []
#     #     for element in directory_folders:
#     #         folders.append(element)
#     #         #print(folders)
#     # # for roots, dirs, files in os.walk(path):
#     #     for file in folders:
#     #         with open('chords.csv') as csv_file:
#     #             # print(file)
#     #             csv_r = csv.reader(csv_file)
#     #             csv_list.append(csv_r)
#                 # print(csv_r)

#                 # PARA USO DO 'notes.csv'
#                 # df = pd.read_excel('chords.csv', header=[0,1,2,3,4,5,6])
#                 # df.columns = df.columns.map(','.join) #Concatenate by ',' the fields name
#                 # df = df.rename_axis('Onset').reset_index() #reset and rename index
#                 # df2 = pd.melt(df, id_vars=list(df.columns)[0], value_vars=list(df.columns)[1:], value_name='Measure')
#                 # df2[['Offset', 'Key', 'Degree', 'Quality', 'Inversion' , 'RNN']] = df2['Onset'].str.split(',',expand=True) #Split using ',' as delimeter
#                 # df2.__delitem__('Onset') #Delete extra field 'onset'

#                 # cols = df2.columns.tolist()
#                 # df2 = df2[[cols[0]] + cols[2:] + [cols[1]]]
                
def convert_to_csv(path_to_file, file_name):
            
    if file_name.endswith('xlsx'):
        read_file = pd.read_excel(path_to_file, engine='openpyxl')
    elif file_name.endswith('xls'):
        read_file = pd.read_excel(path_to_file)
    elif file_name.endswith('csv'):
        read_file = pd.read_csv(path_to_file)
    else:
        raise Exception("File not supported")
    
    rootdir = 'C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/NEW_BPS-FH_Dataset'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            convert_to_csv(os.path.join(subdir, file), str(file))
            
            print(os.path.join(subdir, file))

    with open(path_to_file, 'r', encoding='unicode_escape') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            print ('line[{}] = {}'.format(i,line))
    #return read_file.to_csv(file_name, index=None, header=True)
    return path_to_file
            
def read_csvfile(path, file_name):
    file = convert_to_csv(path, file_name)
    with open(file, newline='', encoding = 'unicode_escape', errors = 'ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            print(row)
    return file

def time_info(path):
    time = []
    file = convert_to_csv(path, dict1[2])
    print(file)
    with open(file, newline='', encoding = 'unicode_escape', errors = 'ignore'):
        reader = csv.reader(file, delimiter=' ', quotechar='|')
        for row in reader:
            print(row)
            time.append(float(row[0]) + "-" + float(row[1]))
        
        #print(time)
    return np.array(time)


def chord_info(path):
    chords = []
    csv_reader_chords = read_csvfile(path, dict1[2])

    for row in csv_reader_chords:
        chords.append(str(row[2]))
        
        #print(chords)
    return np.array(chords)

def midi2csv(path): 
    for roots, dirs, files in os.walk(path):
        for files in dirs:
            with open("MIDI2CSV_Converted.csv", 'w', newline='', encoding='utf-8-sig') as csv:
                w = csv.writer(csv)
                for file in files:
                    current = os.path.join(path,file)
                    if files.endswith(".mid"):
                        csv = pm.midi_to_csv(files)
                        midi_notes = os.path.relpath(current,path).split(os.sep)
                        w.writerow(midi_notes)
    
    return csv

def csv2midi(path):
    midi_path = []
    midi_new_path = []
    for roots, dirs, files in os.walk(path):
        for files in dirs:
            if files.startswith("chords"):
                midi_path.append(os.path.join('.', path, files))
                for m in midi_path:
                    midi = pm.csv_to_midi(files)
                with open("CSV2MIDI_Converted.mid", "wb") as output_midi_file:
                    midi_new_path.append(os.path.join('.',path, m))
                    midi_writer = pm.FileWriter(output_midi_file)
                    midi_writer.write(midi)
    
    return midi
    

# #To MIDI files and piano rolls
class MIDI:
    def __init__(self, path, quantization):
        # MIDI Metadata
        self.__path = path
        self.__quantization = quantization

        self.__Ticks = None
        self.__PianoRoll = None
        self.__File = None

    @property
    def quantization(self):
        return self.__quantization

    @property
    def PianoRoll(self):
        return self.__PianoRoll

    @property
    def File(self):
        return self.__File

    def MIDI_INFO(self):
        def Total_Ticks(self):
            midi_file = MidiFile(self.__path)  # Read the file
            Ticks = 0

            # We need to parse the various MIDI files, as well as discover the moments of ticks and its number
            # Cicle "for" based on code from https://mido.readthedocs.io/en/latest/midi_files.html
            for i, track in enumerate(midi_file.tracks):
                print('Track {}: {}'.format(i, track.name))
                counter = 0
                for msg in track:  # To take every information we can about the midi file
                    time_pointer = float(msg.time)
                    counter = counter + time_pointer
                Ticks = max(Ticks, counter)
                print(Ticks)
            self.__Ticks = Ticks

        # Total time of MIDI file
        def Total_Time_MIDI_File(self):
            midi_file = MidiFile(self.__path)
            Ticks_per_Beat = midi_file.ticks_per_beat
            self.__File = int((self.__Ticks / Ticks_per_Beat)
                              * self.__quantization)


time_info('C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/NEW_BPS-FH_Dataset')
