# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:13:17 2022

@author: José Macedo
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

#path = ''

def read_kern(path):
    kern_path = []
    for roots, dirs, files in os.walk(path):
        for files in dirs:
            if files.endswith("krn"):
                file = converter.parse(files)
                kern_path.append(os.path.join('.',path, files))
    
    return kern_path

def kern2midi(path):
    kern_files = read_kern(path)
    midi = ConverterMidi(kern_files)
    
    return midi