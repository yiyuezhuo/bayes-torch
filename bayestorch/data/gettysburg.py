# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:59:39 2018

@author: yiyuezhuo
"""

# The data is from Battle of Gettysburg, HPS 

import numpy as np

Confederate = np.array(
      [[24, 48],
       [29, 10],
       [33,  1],
       [35, 28],
       [36, 37],
       [50, 42],
       [60, 39],
       [72, 45]])

Union = np.array([[42, 17],
       [45,  8],
       [51, 23],
       [52, 26],
       [52, 18],
       [53, 31],
       [54, 29],
       [54, 27],
       [55, 31],
       [55, 26],
       [62, 30],
       [62, 28],
       [63, 21],
       [66, 16],
       [67, 17],
       [68, 16],
       [73,  8],
       [74, 12],
       [93,  0]])

Confederate_name = ["Heth's Div",
 "McLaws' Div",
 "Hood's Div",
 "Anderson's Div",
 "Pender's Div",
 "Rodes' Div",
 "Early's Div",
 "Johnson's Div"]

Union_name = ['2nd D (Humphreys)',
 '1st Div (Birney)',
 '2nd Div (Gibbon)',
 '3rd Div (Hays)',
 '1st Div (Caldwell)',
 '3rd Div (Schurz)',
 '2nd Div (Steinwehr)',
 '3rd Div (Doubleday)',
 '1st Div (Barlow)',
 '2nd Div (Robinson)',
 '1st Div (Wadsworth)',
 '2nd Div (Geary)',
 '1st Div (Williams)',
 '2nd Div (Ayres)',
 '1st Div (Barnes)',
 '3rd Div (Crawford)',
 '3rd Div (Newton)',
 '1st Div (Wright)',
 '2nd Div (Howe)']

Confederate_number = np.array([4628, 6762, 6957, 6686, 5080, 5202, 4572, 6012])

Union_number = np.array([4913,
 5008,
 3558,
 3622,
 3303,
 1633,
 2264,
 2922,
 1475,
 1311,
 1697,
 3851,
 4698,
 3990,
 3411,
 2842,
 4729,
 4181,
 3548])