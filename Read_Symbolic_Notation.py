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
    for files in os.walk(path):
        for file in files:
            if file.endswith != '.csv':
                file = convert_to_csv(path, file_name)
            else:
              with open(file, newline='', encoding = 'unicode_escape', errors = 'ignore') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                  print(row)
    return file

path_info = 'C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/NEW_BPS-FH_Dataset'
def time_info(path_info):
    time = []
    file = convert_to_csv(path_info, dict1[2])
    print(file)
    with open(file, newline='', encoding = 'unicode_escape', errors = 'ignore'):
        reader = csv.reader(file, delimiter=' ', quotechar='|')
        for row in reader:
            print(row)
            time.append(float(row[0]) + "-" + float(row[1]))
        
        #print(time)
    return np.array(time)


def chord_info(path_info):
    chords = []
    csv_reader_chords = read_csvfile(path_info, dict1[2])

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
                    if file.endswith(".mid"):
                        csv = pm.midi_to_csv(file)
                        midi_notes = os.path.relpath(current,path).split(os.sep)
                        w.writerow(midi_notes)
    
    return csv

def csv2midi(path):
    midi_path = []
    midi_new_path = []
    final_midi_path = []
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
                    
                    final_midi_path.append(os.path.join('.', midi_new_path, midi_writer))
    
    return midi_writer

path_kern = 'C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/TAVERN_Dataset'
def kern_midiFile(path_kern):
    midi_path = []
    for roots, dirs, files in os.walk(path_kern):
        if dirs == "Beethoven":
            for subdirs in os.walk(dirs):
                for subsubdirs in os.walk(subdirs):
                    if subsubdirs == "Krn":
                        if files.endswith('.mid'):
                            for i, track in enumerate(mid.tracks):
                                mid = mido.MidiFile(files, clip=True)
                                mid.tracks
                                midi_path.append(os.path.join('.',path, file))
        elif dirs == 'Mozart':
            for subdirs in os.walk(dirs):
                if subdirs == "Krn":
                    for files in subdirs:
                        if files.endswith('.mid'):
                            for i, track in enumerate(mid.tracks):
                                mid = mido.MidiFile(files, clip=True)
                                mid.tracks
                                midi_path.append(os.path.join('.',path, file))
    
    return np.array(midi_path)

def kern2midi(path_kern):
    midi_path = []
    for roots, dirs, files in os.walk(path):
        if dirs == "Beethoven":
            for subdirs in os.walk(dirs):
                for subsubdirs in os.walk(subdirs):
                    if subsubdirs == "Krn":
                        if files.endswith('.krn'):
                            midi = ConvertMidi(files)
                            midi_path.append(os.path.join('.',path, midi))
        elif dirs == 'Mozart':
            for subdirs in os.walk(dirs):
                for subsubdirs in subdirs:
                    for subsubdirs in os.walk(subdirs):
                        if subsubdirs == "Krn":
                            if files.endswith('.krn'):
                                midi = ConvertMidi(files)
                                midi_path.append(os.path.join('.',path, midi))
        
    return midi
    
# #To MIDI files and piano rolls
class Info_MIDI(object):
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

    def total_Ticks(self):
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
    def total_Time_MIDI_File(self):
            midi_file = MidiFile(self.__path)
            Ticks_per_Beat = midi_file.ticks_per_beat
            self.__File = int((self.__Ticks / Ticks_per_Beat)
                              * self.__quantization)
            
    def get_pitch_range(self):
            mid = MidiFile(self.__song_path)
            min_pitch = 200
            max_pitch = 0
            for i, track in enumerate(mid.tracks):
                for message in track:
                    if message.type in ['note_on', 'note_off']:
                        pitch = message.note
                        if pitch > max_pitch:
                            max_pitch = pitch
                        if pitch < min_pitch:
                            min_pitch = pitch
            return min_pitch, max_pitch
        
    def read_file(self, path):
            # Read the midi file and return a dictionnary {track_name : pianoroll}
            mid = MidiFile(self.__song_path)
            for roots, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".csv"):
                        mid = csv2midi(path)
                    elif file.endswith(".krn"):
                        mid = kern2midi(path)
            # Tick per beat
            ticks_per_beat = mid.ticks_per_beat

            # Get total time
            self.get_time_file()
            T_pr = self.__T_file
            # Pitch dimension
            N_pr = 128
            pianoroll = {}

            def add_note_to_pr(note_off, notes_on, pr):
                pitch_off, _, time_off = note_off
                # Note off : search for the note in the list of note on,
                # get the start and end time
                # write it in th pr
                match_list = [(ind, item) for (ind, item) in enumerate(notes_on) if item[0] == pitch_off]
                if len(match_list) == 0:
                    print("Try to note off a note that has never been turned on")
                    # Do nothing
                    return

                # Add note to the pr
                pitch, velocity, time_on = match_list[0][1]
                pr[time_on:time_off, pitch] = velocity
                # Remove the note from notes_on
                ind_match = match_list[0][0]
                del notes_on[ind_match]
                return

            # Parse track by track
            counter_unnamed_track = 0
            for i, track in enumerate(mid.tracks):
                # Instanciate the pianoroll
                pr = np.zeros([T_pr, N_pr])
                time_counter = 0
                notes_on = []
                for message in track:

                    ##########################################
                    ##########################################
                    ##########################################
                    # TODO : keep track of tempo information
                    # import re
                    # if re.search("tempo", message.type):
                    #     import pdb; pdb.set_trace()
                    ##########################################
                    ##########################################
                    ##########################################


                    # print message
                    # Time. Must be incremented, whether it is a note on/off or not
                    time = float(message.time)
                    time_counter += time / ticks_per_beat * self.__quantization
                    # Time in pr (mapping)
                    time_pr = int(round(time_counter))
                    # Note on
                    if message.type == 'note_on':
                        # Get pitch
                        pitch = message.note
                        # Get velocity
                        velocity = message.velocity
                        if velocity > 0:
                            notes_on.append((pitch, velocity, time_pr))
                        elif velocity == 0:
                            add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)
                    # Note off
                    elif message.type == 'note_off':
                        pitch = message.note
                        velocity = message.velocity
                        add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)

                # We deal with discrete values ranged between 0 and 127
                #     -> convert to int
                pr = pr.astype(np.int16)
                if np.sum(np.sum(pr)) > 0:
                    name = unidecode(track.name)
                    name = name.rstrip('\x00')
                    if name == u'':
                        name = 'unnamed' + str(counter_unnamed_track)
                        counter_unnamed_track += 1
                    if name in pianoroll.keys():
                        # Take max of the to pianorolls
                        pianoroll[name] = np.maximum(pr, pianoroll[name])
                    else:
                        pianoroll[name] = pr
            return pianoroll