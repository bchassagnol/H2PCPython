# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:16:50 2019

@author: bchassagno
"""

def est_pair(nombre):
    if (nombre%2==0):
        return nombre/2
    
    
liste=[20,50,75]
new_liste=list(map(est_pair,liste))
print(new_liste)