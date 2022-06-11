from music21 import *
import glob

path_score_BPS = './Chordify/BPS'
path_Tavern_Beethoven = './Chordify/Tavern/Beethoven'
path_Tavern_Mozart = './Chordify/Tavern/Mozart'
path_Bach_Preludes = './Chordify/Bach_Preludes'
path_ABC_Beethoven_Quartets = './Chordify/ABC_(Beethoven_Quartets)'
path_Haydn20 = './Chordify/Haydn_Op20'

#JUST ONE FILE (OR FOR SOME EXCEPTCIONAL CASES)
#file = './Chordify/BPS/bps_32_01.mid'
#b = converter.parse(file)
#b.show()

#bChords = b.chordify()
#bChords.write('midi', str(file) + "_chordifyMIDI.mid")
#bChords.write('musicxml', str(file) + "_chordifySCORE.mxl")
#bChords.show()

def mxl_chordify(file):
    s = converter.parse(file)
    sChords = s.chordify()
    sChords.write('midi', str(file) + "_chordifyMIDI.mid")
    sChords.write('musicxml', str(file) + "_chordifySCORE.mxl")
    return sChords

#for file in glob.glob(path_score_BPS + './*.mid'):
#    new_file = mxl_chordify(file)

#for file in glob.glob(path_Bach_Preludes + './*.mid'):
#    new_file = mxl_chordify(file)

for file in glob.glob(path_ABC_Beethoven_Quartets + './*.mid'):
    new_file = mxl_chordify(file)

#for file in glob.glob(path_Tavern_Mozart + './*.mid'):
#    new_file = mxl_chordify(file)

#for file in glob.glob(path_Tavern_Beethoven + './*.mid'):
#    new_file = mxl_chordify(file)

#for file in glob.glob(path_Haydn20 + './*.mid'):
#    new_file = mxl_chordify(file)

#for file in glob.glob(path_Haydn20 + './**/*.mid'):
#    new_file = mxl_chordify(file)

#for file in glob.glob(path_Haydn20 + './****/***/**/*.mid'):
#    new_file = mxl_chordify(file)