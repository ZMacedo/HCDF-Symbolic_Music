# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 23:14:15 2022

@author: HP
"""

from anytree import Node, RenderTree
from collections import defaultdict
import os
import pandas as pd

dict1 = {1: 'beats.xlsx', 2: 'chords.xlsx', 3: 'dBeats.xlsx', 4: 'notes.xlsx', 5: 'phrases.xlsx'}

path = 'C:/Users/HP/Downloads/Codigos_Tese/Codigo_ZeMacedo/Datasets/BPS_FH_Dataset'

def symbolic_tree_chord(path):
    file_list = os.listdir(path)
    for file in file_list:
        if file.startswith(dict1[2]) in file_list:
            with open(file, 'r') as csv_file:
                workbook = pd.read_excel(csv_file)
                #workbook = pd.read_excel(file)
                cols = workbook.columns.tolist()
                df1 = workbook[cols[2]]
                df2 = workbook[cols[4]]
                df3 = workbook[cols[6]]
                chord = Node(df1 + df2 + " - " + df3)
                print(chord)
            return chord
            
symbolic_tree_chord(path)

#NOTE: BASIC TREE REPRESENTATION
# udo = Node("Udo")
# marc = Node("Marc", parent=udo)
# lian = Node("Lian", parent=marc)
# dan = Node("Dan", parent=udo)
# jet = Node("Jet", parent=dan)
# jan = Node("Jan", parent=dan)
# joe = Node("Joe", parent=dan)

# print(udo)
# print(joe)

# for pre, fill, node in RenderTree(udo):
#     print("%s%s" % (pre, node.name))

# print(dan.children)



# def rec_dd():
#     """"recursive default dict"""
#     return defaultdict(rec_dd)

# tree = rec_dd()
# for dirs in os.listdir(path):
#     for files in dirs:
#         print(files)
#         if not files.endswith('.csv'):
#             cur_tree = tree['.']
#         else:
#             cur_tree = tree
#         for key in files.split('\\'):
#             cur_tree = cur_tree[key]

#         for d in dirs:
#             cur_tree[d] = rec_dd()
            
# print(cur_tree)

